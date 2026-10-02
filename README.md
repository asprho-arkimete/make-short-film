# Make Short Film

Crea cortometraggi e video con **FLUX.2 9B** (frame iniziale) e **LTX 2.3 quantizzato (NVFP4)**.

- Sequenza di frame personalizzabile, da 9 a 121 frame per clip, una clip alla volta, per limitare l'uso della VRAM a 8-10 GB.
- App (immagini/video) ottimizzata al massimo per funzionare su schede video di fascia bassa.
- Crea i fotogrammi chiave con FLUX.2 (con LoRA) e poi li anima generando le clip.

## Prerequisiti (per chi parte da zero)

Installa questi programmi prima di iniziare:

| Programma | A cosa serve | Download |
|---|---|---|
| **Python** (3.10 consigliato) | Esegue l'app. Durante l'installazione spunta **"Add Python to PATH"** | https://www.python.org/downloads/ |
| **Git** | Scarica il progetto con `git clone` | https://git-scm.com/downloads |
| **Anaconda**  Terminale avvia comandi, gestisce gli ambienti virtuali | https://www.anaconda.com/download |
| **Cursor** |Editor di codice con AI, utile per modificare i file | https://cursor.com/download |

Per scaricare l'audio da URL serve anche **FFmpeg** nel PATH di sistema: https://ffmpeg.org/download.html

{
 UNA VOLTA ISTALLATO I 4 PROGRAMMI USA ANACONDA (TERMINALE) PER ESEGUIRE I COMMANDI SOTTO. 
}

## Installazione

```bash
git clone https://github.com/asprho-arkimete/make-short-film.git
cd make-short-film
python -m venv vmake
vmake\Scripts\activate
pip install -r requirements.txt
```

## LoRA

- **FLUX.2**: https://huggingface.co/Asprho/megalora/tree/main
- **LTX 2.3**: https://civitai.com/models/2535622/ltx-23-enhancers?modelVersionId=2849716
- Altri LoRA da scaricare sono elencati su Civitai e nel file `lora_ltx.txt`.

## Modelli

I modelli base di LTX 2.3 e gli altri modelli necessari si scaricano automaticamente. Se preferisci un altro modello base per LTX, puoi trovarlo su Civitai: l'app permette di aggiungerlo.

## Avvio

```bash
python ltx.py
```

Maggiori dettagli nel [README](https://github.com/asprho-arkimete/make-short-film/blob/main/README.md).




