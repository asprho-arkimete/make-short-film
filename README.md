Crea cortometraggi e video con flux 2 9B  (begin frame) e LTX 2.3 quatitizzato(nfp4) con sequenza personalizzabile di frames da 9 a 121 per 
Clip alla volta per limitare uso della Vram a 8/10GB ; app ( images/video) ottimizzata  al massimo per funzionare su schede fascia bassa. 
Crea fotogrammi chiavi con flux 2 con  lora e crea le clip per animarla.
clona repo: git clone https://github.com/asprho-arkimete/make-short-film.git
cd make-short-film
crea ambiente virtuale: python -m venv vmake
attiva: cd vmake\Scripts - activate

pip install -r requirements.txt

scarica le lora per flux: https://huggingface.co/Asprho/megalora/tree/main
scarica lora per ltx 2.3 : https://civitai.com/models/2535622/ltx-23-enhancers?modelVersionId=2849716
su civitai e nel file lora_ltx.txt; trovi altri lora da scaricare,



