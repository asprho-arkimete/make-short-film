import os
from sys import exception
import time
import traceback
from googletrans.client import Translated
from numpy import insert
try:
    from moviepy import ImageSequenceClip          # moviepy 2.x
except ImportError:
    from moviepy.editor import ImageSequenceClip   # moviepy 1.x

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["DIFFUSERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import logging
import warnings

import torch

for name in ("torch", "torchao", "torch.utils._pytree", "torch.utils.flop_counter"):
    logging.getLogger(name).setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="torch")

import gc
from diffusers import LTX2ImageToVideoPipeline, LTX2VideoTransformer3DModel, BitsAndBytesConfig
from diffusers.utils import encode_video, load_image
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT
from transformers import Gemma3ForConditionalGeneration, Gemma3Processor
from optimum.quanto import freeze, qfloat8, quantize

# Silenzia anche i log informativi di transformers/diffusers
from transformers.utils import logging as hf_logging
from diffusers.utils import logging as df_logging
hf_logging.set_verbosity_error()
df_logging.set_verbosity_error()


import os
import gc
import numpy as np
import torch
from PIL import Image

REPO = "diffusers/LTX-2.3-Diffusers"
GEMMA = "google/gemma-3-12b-it-qat-q4_0-unquantized"
DEVICE = "cuda"
DTYPE = torch.bfloat16

_pipe = None       # cache: i modelli si caricano una volta sola per processo
_pipe_key = None   # (nome_lora, peso) con cui è stata caricata la pipeline


def _lora_file(name):
    """Ritorna il percorso del LoRA oppure None se 'nolora' / non trovato."""
    if not name or name in ('none', 'nolora'):
        return None
    path = os.path.join('lora_ltx', name + '.safetensors')
    if not os.path.exists(path):
        print(f"ATTENZIONE: LoRA non trovato: {path}")
        return None
    return path


def unload_pipeline():
    """Libera la VRAM/RAM."""
    global _pipe, _pipe_key
    _pipe = None
    _pipe_key = None
    gc.collect()
    torch.cuda.empty_cache()


import os
import traceback


def _load_default_transformer(bnb):
    """Transformer predefinito di LTX 2.3 (dal repo), NF4."""
    return LTX2VideoTransformer3DModel.from_pretrained(
        REPO, subfolder="transformer",
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
    )


def _load_transformer(model_name, bnb):
    """Carica il transformer standard o quello personalizzato da model/<nome>.safetensors."""
    if not model_name or model_name == 'ltx2.3':
        print("Carico modello predefinito di LTX 2.3")
        return _load_default_transformer(bnb)

    # accetta sia "nome" sia "nome.safetensors"
    file_name = model_name if model_name.endswith('.safetensors') else f"{model_name}.safetensors"
    path_model = os.path.join('model', file_name)

    try:
        if not os.path.isfile(path_model):
            raise FileNotFoundError(f"File non trovato: {path_model}")
        transformer = LTX2VideoTransformer3DModel.from_single_file(
            path_model,
            config=REPO,
            subfolder="transformer",
            quantization_config=bnb,
            torch_dtype=torch.bfloat16,
        )
        print(f"Carico modello personalizzato: {model_name}")
        return transformer
    except Exception as error:
        traceback.print_exc()
        print(f"Errore: {error} -> carico modello predefinito di LTX 2.3")
        return _load_default_transformer(bnb)


def load_pipeline(lora_name='nolora', lora_weight=0.8, model_name=None):
    """Carica text encoder qfloat8 + transformer NF4 + pipeline (+ LoRA opzionale)."""
    global _pipe, _pipe_key, Modelsltx

    # meglio passare model_name dal thread principale; fallback sul widget
    if model_name is None:
        try:
            model_name = Modelsltx.get()
        except Exception:
            model_name = 'ltx2.3'

    key = (lora_name, lora_weight, model_name)
    if _pipe is not None and _pipe_key == key:
        return _pipe
    if _pipe is not None:
        unload_pipeline()   # LoRA o modello cambiato: ricarico da zero

    # --- Text encoder: qfloat8 ---
    processor = Gemma3Processor.from_pretrained(GEMMA)
    text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
        GEMMA, dtype=DTYPE, low_cpu_mem_usage=True
    )

    # --- Transformer: NF4 con bitsandbytes ---
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    transformer = _load_transformer(model_name, bnb)

    # --- Pipeline ---
    pipe = LTX2ImageToVideoPipeline.from_pretrained(
        REPO, transformer=transformer, text_encoder=text_encoder, dtype=DTYPE,
    )

    # --- LoRA (prima dell'offload, senza fuse) ---
    lora_path = _lora_file(lora_name)
    if lora_path:
        pipe.load_lora_weights(lora_path, adapter_name='lora_ltx')
        pipe.set_adapters(['lora_ltx'], adapter_weights=[lora_weight])

    quantize(pipe.text_encoder, weights=qfloat8)
    freeze(pipe.text_encoder)

    if getattr(pipe, "processor", None) is None:
        pipe.processor = processor

    pipe.enable_model_cpu_offload(device=DEVICE)
    pipe.vae.enable_tiling()

    _pipe, _pipe_key = pipe, key
    return _pipe


def _fit_size(img_w, img_h, long_side):
    """Dimensioni con lo stesso aspect ratio dell'immagine, multipli di 32."""
    if img_w >= img_h:
        w = long_side
        h = long_side * img_h / img_w
    else:
        h = long_side
        w = long_side * img_w / img_h
    w = max(32, int(round(w / 32)) * 32)
    h = max(32, int(round(h / 32)) * 32)
    return w, h


def _to_uint8(frames):
    """Frame float 0-1 (o già uint8) -> array uint8."""
    arr = np.asarray(frames)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).round().astype(np.uint8)
    return arr


def ltx2_3(
    image_path,
    prompt,
    path_lora='nolora',      # nome del file in lora_ltx/ senza estensione
    lora_weight=0.8,
    n_clips=1,
    output_path="ltx2_3_i2v.mp4",
    audio=True,
    width=None,              # se None, calcolata dall'aspect ratio dell'immagine
    height=None,
    long_side=640,           # lato lungo usato quando width/height sono None
    num_frames=41,           # viene portato a 8n+1
    frame_rate=8.0,
    steps=20,
    seed=42,
    guidance_scale=3.0,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
):
    global stop_event
    """Genera n_clips video consecutivi da un'immagine con LTX-2.3. Ritorna la lista dei video."""
    pipe = load_pipeline(path_lora, lora_weight)

    image = load_image(image_path)

    # dimensioni: multipli di 32
    if width is None or height is None:
        width, height = _fit_size(image.size[0], image.size[1], long_side)
    width, height = (width // 32) * 32, (height // 32) * 32

    # frame: deve essere 8n+1
    num_frames = max(9, ((num_frames - 1) // 8) * 8 + 1)

    if audio:
        a_cfg, a_stg, a_mod, a_resc = 7.0, 1.0, 3.0, 0.7
    else:
        a_cfg, a_stg, a_mod, a_resc = 1.0, 0.0, 1.0, 0.0

    generator = torch.Generator("cpu").manual_seed(seed)

    # cartella frame: frames_ltx/<nome_animazione>/
    nome_animazione = 'Animazione' if path_lora in ('none', 'nolora') else path_lora
    frames_dir = os.path.join('frames_ltx', nome_animazione)
    # se la cartella esiste già, creo Animazione_1, Animazione_2, ... (lora1_1, lora1_2, ...)
    n = 1
    while os.path.exists(frames_dir):
        frames_dir = os.path.join('frames_ltx', f"{nome_animazione}_{n}")
        n += 1
    os.makedirs(frames_dir)

    base, ext = os.path.splitext(output_path)
    outputs = []
    current_image = image
    j = 1  # contatore globale dei frame salvati
    # Traduzione prima di caricare il modello, così un errore di rete esce subito
    translator = googletrans.Translator()
    result = asyncio.run(translator.translate(prompt, src='it', dest='en'))
    prompt_en = result.text
    print(f"prompt Inglese di LTX 2.3: {prompt_en}")
    for k in range(1, n_clips+1):
        if stop_event.is_set():
            break
        print(f"LTX2.3 clip {k}/{n_clips}: {width}x{height}, "
              f"{num_frames} frame @ {frame_rate} fps, {steps} step")

        video, audio_out = pipe(
            image=current_image,
            prompt=prompt_en,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=float(frame_rate),
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            stg_scale=1.0,
            modality_scale=3.0,
            guidance_rescale=0.7,
            audio_guidance_scale=a_cfg,
            audio_stg_scale=a_stg,
            audio_modality_scale=a_mod,
            audio_guidance_rescale=a_resc,
            spatio_temporal_guidance_blocks=[28],
            use_cross_timestep=True,
            generator=generator,
            output_type="np",
            return_dict=False,
        )

        # --- salva i frame ---
        frames = _to_uint8(video[0])
        # dalla seconda clip salto il primo frame (uguale all'ultimo della precedente)
        to_save = frames if k == 1 else frames[1:]
        for f in to_save:
            Image.fromarray(f).save(os.path.join(frames_dir, f"f_{j:05d}.png"))
            j += 1
        # la clip successiva parte dall'ultimo frame di questa
        current_image = Image.fromarray(frames[-1])
        del video, audio_out
        gc.collect()
        torch.cuda.empty_cache()

    return outputs


 




import os
import json
import tkinter as tk
from tkinter import ttk, HORIZONTAL, filedialog
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image, ImageTk

EXTS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
COLORS = ['red', 'pink', 'light green', 'light blue']

prompts = []
file_loras1 = []
file_loras2 = []
lora1_path = 'nolora'


def load_lora():
    """Carica i preset dal JSON e i file LoRA dalla cartella 'lora'."""
    prompts.clear()
    file_loras1.clear()
    file_loras2.clear()

    with open("presets_prompts.json", "r", encoding="utf-8") as f:
        presets = json.load(f)

    azioni = []
    for p in presets:
        azioni.append(p["azione"])
        prompts.append(p["prompt"].replace('\n', ' '))
        file_loras1.append(p["nome_lora1"])
        file_loras2.append(p.get("nome_lora2", ""))
    lora1['values'] = azioni

    modelli = ['nolora']
    if os.path.exists('lora'):
        modelli += [os.path.splitext(f)[0] for f in os.listdir('lora')
                    if f.endswith('.safetensors') or f.endswith('.ckpt')]
    lora2['values'] = modelli


def seleziona_lora(event=None):
    global lora1_path
    i = lora1.current()
    if i < 0:
        return
    text_flux.delete("1.0", "end")          # se text_flux è un tk.Text usa: delete("1.0", "end")
    text_flux.insert("1.0", prompts[i])     # se è un tk.Text usa: insert("1.0", prompts[i])
    lora1_path = file_loras1[i]
    if file_loras2[i]:
        lora2.set(file_loras2[i])
    print(f"path_lora1 selezionato {lora1_path}")


# Funzione per caricare i modelli LTX
def load_models():
    modelli = ['ltx2.3']
    if os.path.exists('model'):
        modelli += [os.path.splitext(f)[0] for f in os.listdir('model')
                    if f.endswith('.safetensors') or f.endswith('.ckpt')]
    Modelsltx['values'] = modelli


# Funzione per caricare i Lora per LTX
def load_lora_ltx(event=None):
    modelli = ['nolora']
    if os.path.exists('lora_ltx'):
        modelli += [os.path.splitext(f)[0] for f in os.listdir('lora_ltx')
                    if f.endswith('.safetensors') or f.endswith('.ckpt')]
    loraltx['values'] = modelli


window = TkinterDnD.Tk()
window.title("Video Generator")
window.geometry("1024x600")
window.resizable(False, False)

# Contenitore riga superiore
top_frame = tk.Frame(window)
top_frame.grid(row=0, column=0, sticky="nw")

# Frame FLUX 2
frame_flux2 = tk.LabelFrame(top_frame, text="FLUX 2", padx=10, pady=5)
frame_flux2.pack(side="left", anchor="n", padx=10, pady=10)



lab_step = tk.Label(frame_flux2, text='Flux Steps')
lab_step.grid(row=0, column=0, pady=2, sticky="w")

steps = tk.Scale(frame_flux2, from_=1, to=50, orient=HORIZONTAL, length=150)
steps.grid(row=1, column=0, pady=2, sticky="w")
steps.set(8)

lab_lora1 = tk.Label(frame_flux2, text='Lora 1')
lab_lora1.grid(row=0, column=1, padx=10)

lora1 = ttk.Combobox(frame_flux2, values=[], width=15, state="readonly")
lora1.grid(row=1, column=1, padx=10)
lora1.set('nolora')
lora1.bind("<<ComboboxSelected>>", seleziona_lora)

lab_lora2 = tk.Label(frame_flux2, text='Lora 2')
lab_lora2.grid(row=0, column=2, padx=10)

lora2 = ttk.Combobox(frame_flux2, values=[], width=15, state="readonly")
lora2.grid(row=1, column=2, padx=10)
lora2.set('nolora')

# Carica preset e LoRA una sola volta all'avvio
# (va chiamata DOPO aver creato lora1 e lora2, e dopo aver creato text_flux)
load_lora()

import torch
from diffusers import Flux2KleinPipeline
import googletrans
import asyncio

LORA_DIR = './lora'


def lora_path(name):
    """Ritorna il path della lora se esiste, altrimenti None."""
    if not name or name == 'nolora':
        return None
    if not name.endswith('.safetensors'):
        name += '.safetensors'
    p = os.path.join(LORA_DIR, name)
    return p if os.path.exists(p) else None


def resize_keep_ratio(img, max_side):
    """Ridimensiona mantenendo le proporzioni, lati multipli di 16."""
    w, h = img.size
    if w >= h:
        nw, nh = max_side, (max_side * h) // w
    else:
        nh, nw = max_side, (max_side * w) // h
    nw, nh = max(16, nw // 16 * 16), max(16, nh // 16 * 16)
    return img.resize((nw, nh), Image.Resampling.BICUBIC)

import re
from functools import lru_cache

import torch
from transformers import MarianMTModel, MarianTokenizer

_MT_NAME = "Helsinki-NLP/opus-mt-it-en"
_mt_tok = None
_mt_model = None


def _load_mt():
    global _mt_tok, _mt_model
    if _mt_model is None:
        _mt_tok = MarianTokenizer.from_pretrained(_MT_NAME)
        _mt_model = MarianMTModel.from_pretrained(_MT_NAME).eval()   # resta su CPU


@lru_cache(maxsize=256)
def traduci_it_en(testo: str) -> str:
    """Traduzione italiano -> inglese offline. Se qualcosa va storto restituisce il testo originale."""
    testo = testo.strip()
    if not testo:
        return testo
    try:
        _load_mt()
        # spezza in frasi/righe per non superare la lunghezza massima del modello
        parti = [p for p in re.split(r'(?<=[.!?])\s+|\n+', testo) if p.strip()]
        with torch.no_grad():
            batch = _mt_tok(parti, return_tensors="pt", padding=True, truncation=True, max_length=256)
            out = _mt_model.generate(**batch, num_beams=4, max_new_tokens=256)
        return " ".join(_mt_tok.batch_decode(out, skip_special_tokens=True))
    except Exception as e:
        print(f"Traduzione locale fallita ({e}), uso il testo originale")
        return testo



def flux2(prompt_it, image_paths, n_steps, lora_names):
    global lora1
    print("crea Image con Flux 2")
    device = "cuda"
    dtype = torch.bfloat16

    # Traduzione prima di caricare il modello, così un errore di rete esce subito
    prompt_en = traduci_it_en(prompt_it)
    print(f"prompt Inglese: {prompt_en}")

    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-9B", dtype=dtype)
    try:
        # LoRA: si impostano tutti gli adapter in una sola chiamata
        adapters, weights = [], []
        for i, name in enumerate(lora_names, start=1):
            p = lora_path(name)
            if p:
                pipe.load_lora_weights(p, adapter_name=f'lora{i}')
                adapters.append(f'lora{i}')
                weights.append(0.8)
        if adapters:
            pipe.set_adapters(adapters, adapter_weights=weights)

        quantize(pipe.transformer, weights=qfloat8)
        freeze(pipe.transformer)
        quantize(pipe.text_encoder, weights=qfloat8)
        freeze(pipe.text_encoder)

        pipe.enable_model_cpu_offload()

        # Immagini di input valide
        valid = [p for p in image_paths if p and os.path.exists(p)]
        max_side = 256 if len(valid) > 2 else 512
        images = []
        for k, p in enumerate(valid, start=1):
            print(f"photo {k}: {p}")
            img = Image.open(p).convert('RGB')
            images.append(resize_keep_ratio(img, max_side))

        kwargs = {}
        if len(images) == 1:
            kwargs['image'] = images[0]
        elif len(images) > 1:
            kwargs['image'] = images      # più riferimenti = lista
        
        gw, gh = 1024, 1024
        if valid and "rimuovi vestiti" in lora1.get():
            w, h = Image.open(valid[0]).size
            if w >= h:
                gh = (1024 * h) // w
            else:
                gw = (1024 * w) // h
            # FLUX vuole lati multipli di 16
            gw = max(16, (gw // 16) * 16)
            gh = max(16, (gh // 16) * 16)
        print(f"generate width:{gw},generate height:{gh}")
        image = pipe(
            prompt=prompt_en,
            height=gh,
            width=gw,
            guidance_scale=1.0,
            num_inference_steps=n_steps,
            generator=torch.Generator(device=device).manual_seed(0),
            **kwargs
        ).images[0]

        # Crea la cartella se non esiste e incrementa k finché il file esiste
        os.makedirs("Frames", exist_ok=True)
        k = 1
        out = os.path.join("Frames", f"Frame {k}.png")
        while os.path.exists(out):
            k += 1
            out = os.path.join("Frames", f"Frame {k}.png")
        image.save(out)
        print(f"salvata: {out}")
        return out
    finally:
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

import threading as T
def on_generate_flux2():
    global lora_path
    # I widget si leggono nel thread principale, poi si passa tutto al worker
    prompt_it = text_flux.get('1.0', tk.END).strip()
    if not prompt_it:
        print("Prompt vuoto")
        return
    args = (prompt_it, list(path_images), int(steps.get()),
            [lora1_path, lora2.get()])

    generate_flux2.config(state='disabled')

    def worker():
        try:
            flux2(*args)
        except Exception:
            traceback.print_exc()
        finally:
            window.after(0, lambda: generate_flux2.config(state='normal'))

    T.Thread(target=worker, daemon=True).start()


generate_flux2 = tk.Button(frame_flux2, text='Genera FLUX2', width=15,
                           bg='light green', command=on_generate_flux2)
generate_flux2.grid(row=1, column=3, padx=15)
# Inizializza valori Lora
load_lora()

# Path esterni: path_images[0] = immagine 1, ecc.
path_images = [None] * 4
photo_refs = [None] * 4   # evita che Tk cancelli le immagini

frame_canvas = tk.LabelFrame(top_frame, text="Images Input", padx=10, pady=5)
frame_canvas.pack(side="left", anchor="n", padx=10, pady=10)








def set_image(idx, path):
    canvas = canvases[idx]
    try:
        img = Image.open(path).convert('RGB')
    except Exception as e:
        print(f"Impossibile aprire {path}: {e}")
        return
    img.thumbnail((128, 128), Image.Resampling.BICUBIC)  # mantiene le proporzioni
    photo = ImageTk.PhotoImage(img)
    canvas.delete('all')
    canvas.create_image(64, 64, image=photo)
    photo_refs[idx] = photo
    path_images[idx] = path


def clear_image(idx):
    canvases[idx].delete('all')
    photo_refs[idx] = None
    path_images[idx] = None


def on_drop(event, idx):
    files = window.tk.splitlist(event.data)   # gestisce path con spazi ({...})
    if files and files[0].lower().endswith(EXTS):
        set_image(idx, files[0])


def on_click(event, idx):
    path = filedialog.askopenfilename(
        filetypes=[("Immagini", "*.png *.jpg *.jpeg *.webp *.bmp")])
    if path:
        set_image(idx, path)


canvases = []
for i, color in enumerate(COLORS):
    c = tk.Canvas(frame_canvas, width=128, height=128, bg=color)
    c.grid(row=i // 2, column=i % 2, sticky='n')
    c.create_text(64, 64, text=f'Riferimento {i+1}', fill='black')
    c.drop_target_register(DND_FILES)
    c.dnd_bind('<<Drop>>', lambda e, idx=i: on_drop(e, idx))
    c.bind('<Double-Button-1>', lambda e, idx=i: on_click(e, idx))  # fallback senza drag
    c.bind('<Button-3>', lambda e, idx=i: clear_image(idx))         # tasto destro = svuota
    canvases.append(c)

# Frame LTX 2.3
frame_ltx23 = tk.LabelFrame(window, text="LTX 2.3", padx=10, pady=5)
frame_ltx23.grid(row=1, column=0, padx=10, pady=10, sticky="nw")

lab_modelltx = tk.Label(frame_ltx23, text='Modello')
lab_modelltx.grid(row=0, column=0, sticky="w")
Modelsltx = ttk.Combobox(frame_ltx23, values=[], width=20)
Modelsltx.grid(row=1, column=0, padx=5, pady=3, sticky="w")
Modelsltx.set('ltx2.3')
Modelsltx.bind("<Button-1>", lambda e: load_models())
load_models()

lab_stepltx = tk.Label(frame_ltx23, text='Steps')
lab_stepltx.grid(row=0, column=1, sticky="w")
stepsltx = tk.Scale(frame_ltx23, from_=1, to=50, orient=HORIZONTAL, length=100)
stepsltx.grid(row=1, column=1, padx=5, pady=3, sticky="w")
stepsltx.set(8)

lab_loraltx = tk.Label(frame_ltx23, text='Lora')
lab_loraltx.grid(row=0, column=2, sticky="w")

os.makedirs('lora_ltx', exist_ok=True)
loraltx = ttk.Combobox(frame_ltx23, values=[], width=15)
loraltx.grid(row=1, column=2, padx=5, pady=3, sticky="w")
loraltx.set('nolora')
load_lora_ltx()
loraltx.bind("<Button-1>", load_lora_ltx)

import math
import threading as T
import traceback

MAX_CLIPS = 30   # con 30 fps e 20 s servono 600 frame -> 24 clip da 25 frame

# --- FPS ---
lab_fps = tk.Label(frame_ltx23, text='FPS')
lab_fps.grid(row=0, column=3, sticky="w")
fps = ttk.Combobox(
    frame_ltx23, 
    values=['12', '15', '24', '25', '30', '50', '60'], 
    width=10
)
fps.grid(row=1, column=3, padx=5, pady=3, sticky="w")
fps.set('15')

# --- Numero Frames (frames per singola clip) ---
lab_n_frames = tk.Label(frame_ltx23, text='Numero Frames')
lab_n_frames.grid(row=0, column=4, sticky="w")
n_frames = ttk.Combobox(frame_ltx23,values=['9', '17', '25', '33', '41', '49', '57', '65', '73', '81', '89', '97', '105', '113', '121'],width=10, state="readonly")
n_frames.grid(row=1, column=4, padx=5, pady=3, sticky="w")
n_frames.set('81')

# --- Frames Totali ---
lab_n_frames_totali = tk.Label(frame_ltx23, text='Frames Totali')
lab_n_frames_totali.grid(row=0, column=5, sticky="w")
frames_totali = ttk.Combobox(frame_ltx23, values=[], width=10)
frames_totali.grid(row=1, column=5, padx=5, pady=3, sticky="w")

# --- Clips Totali ---
lab_clips_totali = tk.Label(frame_ltx23, text='Clips Totali')
lab_clips_totali.grid(row=0, column=6, sticky="w")
clips_totali = ttk.Combobox(frame_ltx23, values=[str(i) for i in range(1, MAX_CLIPS + 1)],
                            width=10)
clips_totali.grid(row=1, column=6, padx=5, pady=3, sticky="w")

# --- Secondi ---
Lab_second = tk.Label(frame_ltx23, text='Secondi')
Lab_second.grid(row=0, column=7, padx=5, pady=3, sticky="w")
second = ttk.Combobox(frame_ltx23, width=5, values=['5','8','10', '15', '20'], state="readonly")
second.grid(row=1, column=7, padx=5, pady=3, sticky="w")
second.set('5')

# --- Durata effettiva (row 0, per non sovrapporsi al bottone in row 1) ---
lab_durata = tk.Label(frame_ltx23, text='')
lab_durata.grid(row=0, column=8, padx=5, pady=3, sticky="w")


# ---------------------------------------------------------------
# Helper
# ---------------------------------------------------------------
def _get_int(widget, default=None):
    try:
        v = int(widget.get())
        return v if v > 0 else default
    except (ValueError, tk.TclError):
        return default


def get_frames_per_clip():
    """Frames per singola clip (usata anche dalla generazione)."""
    return _get_int(n_frames, 1)


def get_effective_fpc():
    """Frames effettivi aggiunti da ogni clip tenendo conto dell'overlap (1 frame in meno)."""
    fpc = get_frames_per_clip()
    return max(1, fpc - 1)


def aggiorna_lista_frames_totali():
    """Valori della combo frames_totali tenendo conto dell'overlap."""
    fpc = get_frames_per_clip()
    eff_fpc = get_effective_fpc()
    # Formula totale: (fpc - 1) * i + 1
    frames_totali['values'] = [str(eff_fpc * i + 1) for i in range(1, MAX_CLIPS + 1)]


def aggiorna_durata():
    f = _get_int(fps)
    tot = _get_int(frames_totali)
    lab_durata.config(text=f"≈ {tot / f:.1f} s" if (f and tot) else "")


def ricalcola_da_secondi():
    """secondi * fps -> frames richiesti -> clips necessarie -> frames totali."""
    f = _get_int(fps)
    s = _get_int(second)
    fpc = _get_int(n_frames)
    if not (f and s and fpc):
        return

    frames_richiesti = f * s
    eff_fpc = max(1, fpc - 1)

    # Formula per il calcolo delle clip necessarie con overlap
    if frames_richiesti <= fpc:
        clips = 1
    else:
        clips = math.ceil((frames_richiesti - 1) / eff_fpc)

    clips = max(1, min(clips, MAX_CLIPS))
    totale_frames = eff_fpc * clips + 1

    print(f"frames richiesti ({f}*{s}): {frames_richiesti} -> {clips} clips da {fpc} frames (effettivi: {totale_frames})")

    aggiorna_lista_frames_totali()
    clips_totali.set(str(clips))
    frames_totali.set(str(totale_frames))
    aggiorna_durata()


f_second = ricalcola_da_secondi   # alias per il vecchio nome


# ---------------------------------------------------------------
# Eventi
# ---------------------------------------------------------------
def on_change_fps(event=None):
    ricalcola_da_secondi()


def on_change_second(event=None):
    ricalcola_da_secondi()


def on_change_n_frames(event=None):
    ricalcola_da_secondi()


def on_select_frames_totali(event=None):
    """Frames totali scelti -> calcola clips tenendo conto dell'overlap."""
    totale = _get_int(frames_totali)
    fpc = get_frames_per_clip()
    eff_fpc = get_effective_fpc()
    if not totale:
        return
    
    if totale <= fpc:
        clips = 1
    else:
        clips = math.ceil((totale - 1) / eff_fpc)

    clips = max(1, min(clips, MAX_CLIPS))
    clips_totali.set(str(clips))
    frames_totali.set(str(eff_fpc * clips + 1))   # allinea al valore corretto
    aggiorna_durata()


def on_select_clips_totali(event=None):
    """Clips scelte -> frames totali = (fpc - 1) * clips + 1."""
    clips = _get_int(clips_totali)
    fpc = get_frames_per_clip()
    eff_fpc = get_effective_fpc()
    if not clips:
        return
    clips = min(clips, MAX_CLIPS)
    clips_totali.set(str(clips))
    frames_totali.set(str(eff_fpc * clips + 1))
    aggiorna_durata()


fps.bind('<<ComboboxSelected>>', on_change_fps)
fps.bind('<Return>', on_change_fps)                       # fps è editabile a mano
fps.bind('<FocusOut>', on_change_fps)
second.bind('<<ComboboxSelected>>', on_change_second)
n_frames.bind('<<ComboboxSelected>>', on_change_n_frames)
frames_totali.bind('<<ComboboxSelected>>', on_select_frames_totali)
frames_totali.bind('<Return>', on_select_frames_totali)   # editabile a mano
clips_totali.bind('<<ComboboxSelected>>', on_select_clips_totali)
clips_totali.bind('<Return>', on_select_clips_totali)

# --- Valori iniziali ---
ricalcola_da_secondi()   # 15 fps * 5 s = 75 frames -> 1 clip da 81


# ---------------------------------------------------------------
# Generazione LTX 2.3
# ---------------------------------------------------------------
import math

def calcola_long_side(num_frames, ref_frames=10, ref_side=512,
                      multiple=32, min_side=256, max_side=768):
    """Mantiene circa costante frame * lato^2 rispetto al riferimento che funziona
    sulla 4060 Ti 16GB (25 frame a 640), poi arrotonda per difetto a multipli di 32."""
    side = ref_side * math.sqrt(ref_frames / num_frames)
    side = int(side // multiple * multiple)
    return max(min_side, min(side, max_side))


def run_ltx2_3(image_select, prompt, lora_name, n_clips, num_frames, frame_rate, steps):
    """Gira in un thread separato: NON legge widget Tk."""
    long_side = calcola_long_side(num_frames)
    print(f"risoluzione video (lato lungo): {long_side}, frames per clip: {num_frames}")
    try:
        ltx2_3(
            image_select,
            prompt=prompt,
            path_lora=lora_name,          # nome file in lora_ltx/ senza estensione
            lora_weight=0.8,
            n_clips=n_clips,
            output_path="ltx2_3_i2v.mp4",
            audio=False,
            width=None,
            height=None,
            long_side=long_side,          # prima era fisso a 480
            num_frames=num_frames,
            frame_rate=frame_rate,
            steps=steps,
            seed=42,
            guidance_scale=3.0,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        )
    except Exception:
        traceback.print_exc()
    finally:
        # riattiva il bottone dal thread principale
        window.after(0, lambda: generate_ltx2.config(state='normal'))


def on_generate_ltx2():
    """Legge i widget (thread principale) e avvia il worker."""
    image_select = next((p for p in path_images if p and os.path.exists(p)), None)
    if image_select is None:
        print("Carica almeno un'immagine in Images Input")
        return

    prompt = text_ltx.get('1.0', tk.END).strip()
    if not prompt:
        print("Il prompt LTX 2.3 è vuoto")
        return

    try:
        n_clips = int(clips_totali.get())
        frame_rate = float(fps.get())
        steps = int(stepsltx.get())
    except ValueError:
        print("Clips Totali, FPS o Steps non validi")
        return

    generate_ltx2.config(state='disabled')
    T.Thread(
        target=run_ltx2_3,
        args=(image_select, prompt, loraltx.get(), n_clips,
              get_frames_per_clip(), frame_rate, steps),
        daemon=True,
    ).start()


generate_ltx2 = tk.Button(frame_ltx23, text="Genera LTX 2.3", width=15,
                          bg='light blue', command=on_generate_ltx2)
generate_ltx2.grid(row=1, column=8, padx=15, pady=3)

frame_prompts = tk.LabelFrame(window, text="PROMPTS", padx=10, pady=5)
frame_prompts.grid(row=2, column=0, padx=10, pady=10, sticky="nw")

lab_prompt1=tk.Label(frame_prompts,text='PROMPT FLUX 2')
lab_prompt1.grid(row=0,column=0,sticky='n')

text_flux=tk.Text(frame_prompts,width=50,height=5)
text_flux.grid(row=1,column=0,padx=5)

lab_prompt2=tk.Label(frame_prompts,text='PROMPT LTX 2.3')
lab_prompt2.grid(row=0,column=1,sticky='n')
text_ltx=tk.Text(frame_prompts,width=50,height=5)
text_ltx.grid(row=1,column=1,padx=5)

# CREA SCENE
import os
import shutil

lista_parametri = None
finestra_scene = None          # <-- nuova
prompts_flux2_selezionati = []
prompts_ltx_selezionati = []
stop_event=None
vd_sc=True

from tkinter import filedialog, simpledialog, messagebox

def scrivifile():
    #scrivifile
    elenco = []
    for e in range(lista_parametri.size()):
        dizionario = {
            "riga_listbox": lista_parametri.get(e),
            "prompt_flux_completo": prompts_flux2_selezionati[e],
            "prompt_ltx_completo": prompts_ltx_selezionati[e]
        }
        elenco.append(dizionario)
        print(f"Dizionario {e}: {dizionario}|Scritto... ")

    with open('database.json', 'w', encoding='utf-8') as f:
        json.dump(elenco, f, ensure_ascii=False, indent=4)
    print("Scrittura File Completata")


def f_aggiungi():
    global path_images, text_flux, lora1_path, lora2, text_ltx, loraltx, n_frames, clips_totali, lista_parametri
    os.makedirs("Input_images", exist_ok=True)

    # FLUX2: Path1 | Path2 | Path3 | Path4 | Prompt | lora1 | lora2
    flux_paths = []
    for p in path_images:
        if p and os.path.exists(p):
            try:
                shutil.copyfile(p, os.path.join("Input_images", os.path.basename(p)))
            except shutil.SameFileError:
                pass
            flux_paths.append(os.path.basename(p))
        else:
            flux_paths.append("None")

    prompt_flux = text_flux.get("1.0", tk.END).strip().replace("\n", " ")
    lora1_name = os.path.basename(lora1_path) if lora1_path else "None"
    lora2_name = lora2.get() if lora2 else "None"

    riga_flux = (
        f"{flux_paths[0]}| {flux_paths[1]}| {flux_paths[2]}| {flux_paths[3]}| "
        f"{prompt_flux[:10]}| {lora1_name}| {lora2_name}"
    )

    # LTX: prompt_ltx | lora_ltx | n_frames | clips
    prompt_ltx = text_ltx.get("1.0", tk.END).strip().replace("\n", " ")
    lora_ltx_val = loraltx.get() if loraltx else "None"
    n_frames_val = n_frames.get() if n_frames else "None"
    clips_val = clips_totali.get() if clips_totali else "None"
    riga_ltx = f"{prompt_ltx[:10]}| {lora_ltx_val}| {n_frames_val}| {clips_val}"

    riga_completa = riga_flux + "|" + riga_ltx
    selection = lista_parametri.curselection()
    if selection:
        # sovrascrive l'elemento selezionato
        idx = selection[0]
        lista_parametri.delete(idx)
        lista_parametri.insert(idx, riga_completa)
        prompts_flux2_selezionati[idx] = prompt_flux
        prompts_ltx_selezionati[idx] = prompt_ltx
    else:
        # aggiunge in coda
        lista_parametri.insert(tk.END, riga_completa)
        prompts_flux2_selezionati.append(prompt_flux)
        prompts_ltx_selezionati.append(prompt_ltx)

    # deseleziona e aggiorna
    lista_parametri.selection_clear(0, tk.END)
    lista_parametri.update_idletasks()

    finestra_scene.bind('<Button-3>', lambda e: lista_parametri.selection_clear(0, tk.END))

    # Print all FLUX2 and LTX parameters stored so far
    print("Parametri FLUX2 memorizzati:")
    for i, flux in enumerate(prompts_flux2_selezionati):
        print(f"{i+1}: {flux}")
    print("Parametri LTX memorizzati:")
    for i, ltx in enumerate(prompts_ltx_selezionati):
        print(f"{i+1}: {ltx}")
    # Also print the latest inputs (expanded)
    print(f"path1: {flux_paths[0]}")
    print(f"path2: {flux_paths[1]}")
    print(f"path3: {flux_paths[2]}")
    print(f"path4: {flux_paths[3]}")
    print(f"prompt_flux: {prompt_flux}")
    print(f"lora1: {lora1_name}")
    print(f"lora2: {lora2_name}")
    print(f"prompt_ltx: {prompt_ltx}")
    print(f"lora_ltx: {lora_ltx_val}")
    print(f"n_frames: {n_frames_val}")
    print(f"clips: {clips_val}")
    scrivifile()

import yt_dlp 

def leggidatabase():
    if not os.path.exists('database.json'):
        return

    try:
        with open('database.json', 'r', encoding='utf-8') as f:
            elenco = json.load(f)
    except (json.JSONDecodeError, OSError):
        print("database.json non leggibile")
        return

    # svuota listbox e array prima di ricaricare, per non duplicare le righe
    lista_parametri.delete(0, tk.END)
    prompts_flux2_selezionati.clear()
    prompts_ltx_selezionati.clear()

    for dizionario in elenco:
        lista_parametri.insert(tk.END, dizionario["riga_listbox"])
        prompts_flux2_selezionati.append(dizionario["prompt_flux_completo"])
        prompts_ltx_selezionati.append(dizionario["prompt_ltx_completo"])

    print(f"Lettura File Completata: {len(elenco)} righe")

def stile_bottone(btn, colore, hover):
        btn.config(bg=colore, fg='white', activebackground=hover, activeforeground='white',
                   font=('Segoe UI', 10, 'bold'), relief='flat', bd=0,
                   padx=14, pady=6, cursor='hand2')
        btn.bind('<Enter>', lambda e: btn.config(bg=hover))
        btn.bind('<Leave>', lambda e: btn.config(bg=colore))
        
from tqdm import tqdm
def crea_scene():
    global lista_parametri, finestra_scene

    # se la finestra è già aperta, la porto in primo piano invece di crearne un'altra
    if finestra_scene is not None and finestra_scene.winfo_exists():
        finestra_scene.lift()
        return

    new_form_scene = tk.Toplevel()
    finestra_scene = new_form_scene
    new_form_scene.title("Scene")
    new_form_scene.geometry("1024x900")
    new_form_scene.resizable(False, False)
    new_form_scene.lift()
    frame_buttons=tk.Frame(new_form_scene)
    frame_buttons.grid(row=0, column=0, sticky="wn", padx=8, pady=8)
    aggiungi = tk.Button(frame_buttons, text='Aggiungi', command=f_aggiungi)
    stile_bottone(aggiungi, '#4caf50', '#3d8b40')
    aggiungi.grid(row=0, column=0, sticky='wn', padx=(0, 10))

    def select_ch():
            # Prendi i valori delle selezioni
            images_selected = images_value.get()
            video_selected = video_value.get()

            # Se nessuno dei due è selezionato, forzarne uno (images) e mostra messaggio
            if not images_selected and not video_selected:
                images_value.set(True)
                button_crea_scene.config(text='Crea Immagini Scena')
            elif images_selected and video_selected:
                button_crea_scene.config(text='Crea Immagini||Video Scena')
            elif images_selected:
                button_crea_scene.config(text='Crea Immagini Scena')
            elif video_selected:
                button_crea_scene.config(text='Crea Video Scena')
            button_crea_scene.update()

    images_value = tk.BooleanVar()
    ch_images = tk.Checkbutton(frame_buttons, text="Images", variable=images_value, command=select_ch)
    ch_images.grid(row=0, column=1, sticky="wn", padx=4)
    
    images_value.set(True)

    video_value = tk.BooleanVar()
    ch_video = tk.Checkbutton(frame_buttons, text="Video", variable=video_value, command=select_ch)
    ch_video.grid(row=0, column=2, sticky="wn", padx=(4, 10))

    

    def F_crea_scena():
        global lista_parametri,prompts_flux2_selezionati,prompts_ltx_selezionati
        # Prendi i valori delle selezioni
        images_selected = images_value.get()
        video_selected = video_value.get()
        
        if images_selected:
            print("Creo immagini nella lista")
            paths_salvati=[]
            # leggo la listbox qui, nel thread principale
            righe = lista_parametri.get(0, tk.END)

            def worker_scena():
                try:
                    # estrai tutti parametri
                    for k, elemento in enumerate(tqdm(righe, desc='Generazione Frames')):
                        path_img1, path_img2, path_img3, path_img4, prompt_flux_breve, plora1, plora2, prompt_ltx_breve, lora_ltx, n_frames, n_clip = [x.strip() for x in elemento.split('|')]
            
                        prompt_flux_completo=prompts_flux2_selezionati[k]
                        print("##### STAMPA PARAMETRI flux2 #####")
                        print(f"path image1: {path_img1}")
                        print(f"path image2: {path_img2}")
                        print(f"path image3: {path_img3}")
                        print(f"path image4: {path_img4}")
                        print(f"prompt_flux corto: {prompt_flux_breve}")
                        print(f"prompt flux completo: {prompt_flux_completo}")
                        print(f"lora1 flux: {plora1}")
                        print(f"lora2 flux: {plora2}")

                        lista_images=[os.path.join('./Input_images',p) for p in (path_img1,path_img2,path_img3,path_img4) if p!="None"]

                        args = (prompt_flux_completo, list(lista_images), int(8),
                        [os.path.join('./lora',p) for p in (plora1,plora2) if p!="None"])

                        path_salvato=flux2(*args)
                        paths_salvati.append(path_salvato)

                    with open("frames_flux_generati.json", "w", encoding='utf-8') as f:
                        json.dump(paths_salvati, f, ensure_ascii=False, indent=4)
                except Exception:
                    traceback.print_exc()
                finally:
                    window.after(0, lambda: generate_flux2.config(state='normal'))

            T.Thread(target=worker_scena, daemon=True).start()

        if video_selected:
            print("Creo video nella lista")
            frames_begin = []
            if os.path.exists("frames_flux_generati.json"):
                with open("frames_flux_generati.json", "r", encoding="utf-8") as f:
                    frames_begin = json.load(f)
            else:
                # Inserisce solo i file PNG nella cartella Frames e li ordina
                frames_begin = [os.path.join("Frames", fname) 
                                for fname in sorted(os.listdir("Frames")) if fname.lower().endswith('.png')]

            # leggo la listbox qui, nel thread principale
            righe = lista_parametri.get(0, tk.END)

            #Avvia funzione ltx
            def run_ltx2_3(image_select, prompt, lora_name, n_clips, num_frames, frame_rate, steps):
                """Gira in un thread separato: NON legge widget Tk."""
                long_side = calcola_long_side(num_frames)
                print(f"risoluzione video (lato lungo): {long_side}, frames per clip: {num_frames}")
                ltx2_3(
                    image_select,
                    prompt=prompt,
                    path_lora=lora_name,          # nome file in lora_ltx/ senza estensione
                    lora_weight=0.8,
                    n_clips=n_clips,
                    output_path="ltx2_3_i2v.mp4",
                    audio=False,
                    width=None,
                    height=None,
                    long_side=long_side,          # prima era fisso a 480
                    num_frames=num_frames,
                    frame_rate=frame_rate,
                    steps=steps,
                    seed=42,
                    guidance_scale=3.0,
                    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                )

            def worker_video():
                global stop_event
                try:
                    for k, elemento in enumerate(tqdm(righe, desc='Generazione Video')):
                        if stop_event.is_set():
                            print("Generazione video fermata dall'utente")
                            break
                        path_img1,path_img2,path_img3,path_img4,prompt_flux_breve,plora1,plora2,prompt_ltx_breve,lora_ltx,n_frames,n_clip = [x.strip() for x in elemento.split('|')]
                        prompt_ltx_completo = prompts_ltx_selezionati[k]

                        # cartella frame: frames_ltx/<nome_animazione>/
                        lora_name = 'none' if lora_ltx.lower() in ('none', 'nolora') else lora_ltx
                        nome_animazione = 'Animazione' if lora_name == 'none' else lora_name
                        frames_dir = os.path.join('frames_ltx', nome_animazione)
                        if os.path.exists(frames_dir):
                            print(f"Cartella {frames_dir} già esistente: salto l' Animazione {nome_animazione}")
                            continue

                        print("##### STAMPA PARAMETRI LTX #####")
                        print(f"path image1: {path_img1}")
                        print(f"prompt_ltx corto: {prompt_ltx_breve}")
                        print(f"prompt ltx completo: {prompt_ltx_completo}")
                        print(f"lora1 ltx: {lora_ltx}")
                        print(f"numero frames: {n_frames}")
                        print(f"numero clips: {n_clip}")
                        print(f"Frame begin: {frames_begin[k]}")

                        run_ltx2_3(image_select=frames_begin[k], prompt=prompt_ltx_completo, lora_name=lora_name, n_clips=int(n_clip), num_frames=int(n_frames), frame_rate=int(15), steps=int(8))
                except Exception:
                    traceback.print_exc()
                finally:
                    # riattiva il bottone dal thread principale
                    window.after(0, lambda: generate_ltx2.config(state='normal'))

            process_video= T.Thread(target=worker_video, daemon=True).start()

    button_crea_scene=tk.Button(frame_buttons, text="Crea Immagini Scena", width=26, command=F_crea_scena)
    stile_bottone(button_crea_scene, '#4569f8', '#2f4fd1')
    button_crea_scene.grid(row=0, column=3, sticky="wn", padx=(0, 10))
    def f_stop():
        global stop_event
        stop_event.set()
        print("Stop richiesto: la generazione si ferma al termine del passaggio in corso")

    stoppa_crea_scene=tk.Button(frame_buttons, text="Ferma Generazione Animazioni", command=f_stop)
    stile_bottone(stoppa_crea_scene, '#e5534b', '#c0392b')
    stoppa_crea_scene.grid(row=0, column=4, sticky="wn")

    
    def f_vedi_scene():
        global lista_parametri
        global vd_sc
        print("seleziono le cartelle delle scene e le aggiungo alla listbox")
        if vd_sc:
            lista_parametri.delete(0, tk.END)
            if os.path.isdir('frames_ltx'):
                for d in sorted(os.listdir('frames_ltx')):
                    if os.path.isdir(os.path.join('frames_ltx', d)):
                        lista_parametri.insert(tk.END, d)
            lista_parametri.update_idletasks()
            vd_sc=False
            stile_bottone(vedi_scene, '#f59e0b', 'light green')
            vedi_scene.config(text='Carica Dati')
        else:
            leggidatabase()   # svuota e ricarica la listbox dal database
            lista_parametri.update_idletasks()
            vd_sc=True
            stile_bottone(vedi_scene, '#f59e0b', '#d97706')
            vedi_scene.config(text="Vedi Scene")

    vedi_scene=tk.Button(frame_buttons, text="Vedi Scene", command=f_vedi_scene)
    stile_bottone(vedi_scene, '#f59e0b', '#d97706')
    vedi_scene.grid(row=0, column=5, sticky="wn", padx=(10, 0))


    lab_info = tk.Label(
        new_form_scene,
        text=(
            "FLux2_Parametri: 'Path1 | Path2 | Path3 | Path4 | Prompt | lora1 | lora2' "
            "LTX_Parametri: 'prompt_ltx | lora_ltx | n_frames | clips'"
        )
    )

    lab_info.grid(row=1, column=0, sticky='wn')
    lista_parametri = tk.Listbox(new_form_scene, width=150, height=30, exportselection=False)
    lista_parametri.grid(row=2, column=0, sticky='nw')
    sb_x = tk.Scrollbar(new_form_scene, orient="horizontal", command=lista_parametri.xview)
    sb_y = tk.Scrollbar(new_form_scene, orient="vertical", command=lista_parametri.yview)
    lista_parametri.configure(xscrollcommand=sb_x.set, yscrollcommand=sb_y.set)
    sb_x.grid(row=3, column=0, sticky="ew")
    sb_y.grid(row=2, column=1, sticky="ns")
    


   


    def seleziona_elemento(event=None):
        selezione = lista_parametri.curselection()
        if not selezione:
            return
        ind = selezione[0]
        text_flux.delete("1.0", tk.END)
        text_flux.insert("1.0", prompts_flux2_selezionati[ind])
        text_ltx.delete("1.0", tk.END)
        text_ltx.insert("1.0", prompts_ltx_selezionati[ind])
        text_flux.update_idletasks()
        text_ltx.update_idletasks()

    lista_parametri.bind('<<ListboxSelect>>', seleziona_elemento)

    button_side=tk.Frame(new_form_scene)
    button_side.grid(row=2, column=1, sticky="wn")

    def f_elimina():
        # Elimina elemento selezionato dalla lista
        selezione = lista_parametri.curselection()
        if selezione:
            idx = selezione[0]
            lista_parametri.delete(idx)
            del prompts_flux2_selezionati[idx]
            del prompts_ltx_selezionati[idx]
            scrivifile()

    button_elimina = tk.Button(button_side, width=10, text='Elimina', bg='#a234ec', command=f_elimina)
    button_elimina.grid(row=0, column=0, sticky="wn")

    def f_su():
        # Sposta elemento selezionato verso l'alto
        selezione = lista_parametri.curselection()
        if selezione and selezione[0] > 0:
            idx = selezione[0]
            valore = lista_parametri.get(idx)
            lista_parametri.delete(idx)
            lista_parametri.insert(idx-1, valore)
            lista_parametri.selection_set(idx-1)
            prompts_flux2_selezionati[idx-1], prompts_flux2_selezionati[idx] = prompts_flux2_selezionati[idx], prompts_flux2_selezionati[idx-1]
            prompts_ltx_selezionati[idx-1], prompts_ltx_selezionati[idx] = prompts_ltx_selezionati[idx], prompts_ltx_selezionati[idx-1]
            scrivifile()

    button_su = tk.Button(button_side, width=10, text='Su', bg='light pink', command=f_su)
    button_su.grid(row=1, column=0, sticky="wn")

    def f_giù():
        # Sposta elemento selezionato verso il basso
        selezione = lista_parametri.curselection()
        if selezione and selezione[0] < lista_parametri.size() - 1:
            idx = selezione[0]
            valore = lista_parametri.get(idx)
            lista_parametri.delete(idx)
            lista_parametri.insert(idx+1, valore)
            lista_parametri.selection_set(idx+1)
            prompts_flux2_selezionati[idx+1], prompts_flux2_selezionati[idx] = prompts_flux2_selezionati[idx], prompts_flux2_selezionati[idx+1]
            prompts_ltx_selezionati[idx+1], prompts_ltx_selezionati[idx] = prompts_ltx_selezionati[idx], prompts_ltx_selezionati[idx+1]
            scrivifile()

    button_giù = tk.Button(button_side, width=10, text='Giù', bg='light yellow', command=f_giù)
    button_giù.grid(row=2, column=0, sticky="wn")
    
    path_audio=None
    def f_rendering():
        global vd_sc, lista_parametri, fps
        if vd_sc == False:
            print("rendering frames")
            os.makedirs("video_scene", exist_ok=True)
            frames = []
            # raccoglie i frame di tutte le cartelle, dalla prima all'ultima della lista
            for d in lista_parametri.get(0, tk.END):
                cartella = os.path.join('frames_ltx', d)
                frames += [os.path.join(cartella, f) for f in sorted(os.listdir(cartella))
                           if f.lower().endswith('.png')]
            if not frames:
                print("Nessun frame trovato")
                return
            clip = ImageSequenceClip(frames, fps=int(fps.get()))
            out = os.path.join("video_scene", "scena_completa.mp4")

            if path_audio is None:
                clip.write_videofile(out, codec='libx264', audio=False)
            else:
                from moviepy import AudioFileClip
                audio_clip = AudioFileClip(path_audio)
                durata = min(audio_clip.duration, clip.duration)
                audio_clip = audio_clip.subclipped(0, durata)
                clip = clip.with_audio(audio_clip)
                clip.write_videofile(out, codec='libx264', audio_codec='aac')
                audio_clip.close()

            clip.close()

    rendering=tk.Button(new_form_scene, width=10, text='Rendering', command=f_rendering)
    stile_bottone(rendering, '#8b5cf6', '#7c3aed')
    rendering.grid(row=4, column=0, sticky='nw', padx=8, pady=4)
    
    def f_audio():
        nonlocal path_audio
        import yt_dlp

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        cartella_audio = os.path.join(BASE_DIR, "audio_scaricati")
        os.makedirs(cartella_audio, exist_ok=True)

        # scelta esplicita: evita che Windows risolva un URL come file di cache
        scelta = messagebox.askyesnocancel(
            "Audio",
            "Sì = scegli un file locale\nNo = scarica da URL\nAnnulla = esci",
            parent=new_form_scene
        )
        if scelta is None:
            return

        estensioni_ok = ('.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac', '.mp4', '.mkv')

        # --- file locale ---
        if scelta:
            path = filedialog.askopenfilename(
                title="Seleziona file audio/video",
                filetypes=[("Audio/Video", "*.mp3 *.wav *.m4a *.aac *.ogg *.flac *.mp4 *.mkv")]
            )
            if not path:
                return
            if not path.lower().endswith(estensioni_ok):
                messagebox.showerror("Errore", "File non valido: scegli un file audio/video")
                return
            path_audio = path
            select_audio.config(text='Audio: ' + os.path.basename(path)[:15])
            print(f"Audio selezionato: {path_audio}")
            return

        # --- download da URL ---
        url = simpledialog.askstring("Scarica audio", "Incolla l'URL (es. YouTube):",
                                     parent=new_form_scene)
        if not url:
            return
        url = url.strip()
        print(f"Scarico audio da: {url}")
        print(f"Cartella audio: {cartella_audio}")

        def worker_download():
            risultato = None
            try:
                opzioni = {
                    "format": "bestaudio/best",
                    "outtmpl": os.path.join(cartella_audio, "audio_scena.%(ext)s"),
                    "postprocessors": [{
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "wav",
                    }],
                    "noplaylist": True,
                    "quiet": True,
                }
                with yt_dlp.YoutubeDL(opzioni) as ydl:
                    ydl.download([url])
                risultato = os.path.join(cartella_audio, "audio_scena.wav")
            except Exception:
                traceback.print_exc()

            def aggiorna_ui():
                nonlocal path_audio
                if risultato and os.path.exists(risultato):
                    path_audio = risultato
                    select_audio.config(text='Audio: audio_scena.wav')
                    print(f"Audio scaricato: {risultato}")
                else:
                    messagebox.showerror("Errore", "Download audio non riuscito (vedi console)")

            new_form_scene.after(0, aggiorna_ui)

        T.Thread(target=worker_download, daemon=True).start()

    select_audio = tk.Button(new_form_scene, width=22, text='Seleziona/Scarica Audio',
                             command=f_audio)
    stile_bottone(select_audio, '#0ea5e9', '#0284c7')
    select_audio.grid(row=4, column=0, sticky='nw', padx=(120, 0), pady=4)

    #leggi file database.jons
    leggidatabase()

button_crea_scene=tk.Button(top_frame,text='Crea Scene',bg='light blue',padx=10, pady=5,command=crea_scene)
button_crea_scene.pack(side="left", anchor="n", padx=10, pady=10)


window.mainloop()
