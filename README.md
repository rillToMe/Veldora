# Veldora - README

## Siapa yang bikin
- **Author:** rillToMe

## Ringkas tujuan
Pipeline end‑to‑end buat belajar bikin **model bahasa kecil** dari nol:

1. Ekstrak dump Wikipedia → `wiki.jsonl` + `data.txt`  
2. Latih **SentencePiece** tokenizer (`velora.model/.vocab`)  
3. Encode korpus → `tokens.bin`  
4. **Train Velora Mini (PyTorch)**  
5. Coba **generate** teks

---

## Prasyarat
- **Python 3.11** (disarankan)
- **GPU**: RTX 3050 6GB (atau setara)
- **Windows + VS Code** (atau OS lain oke)

### Buat & aktifkan venv (Windows PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install library
```powershell
pip install -U rich ftfy sentencepiece tqdm
# PyTorch CUDA (pilih sesuai CUDA kamu; contoh CUDA 12.1):
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

---

## Struktur folder 
```
Velora/
├─ data/                         # taruh dump
├─ outputs/
│  ├─ wiki-mini/                 # wiki.jsonl + data.txt 
│  ├─ packed/                    # tokens.bin + meta.json 
│  └─ checkpoints/               # model .pt per run 
├─ wiki_tokenizer.py    # extract + train tokenizer
├─ prep_tokens.py                # encode data.txt -> tokens.bin
├─ train_veldora_small.py            # training Velora Mini
└─ gen_sample.py                 # sampling/generate teks
```

---

## Urutan run (dari nol)

### 1) Extract + Train Tokenizer
(Otomatis bikin `outputs/wiki-mini/data.txt` dan `outputs/velora.{model,vocab}` — atur `--prefix` kalau perlu)
```powershell
python pipeline_wiki2tokenizer.py --prefix "outputs/velora"
# tes cepat (mis. hanya 2000 artikel):
# python pipeline_wiki2tokenizer.py --sample 2000 --prefix "outputs/velora_test"
```

### 2) Pack korpus → tokens.bin
(Bikin `outputs/packed/{tokens.bin, meta.json}`)
```powershell
python prep_tokens.py
```

### 3) Train Velora (mini)
(Checkpoint tersimpan di `outputs/checkpoints/velora-YYYYMMDD-HHMM/`)
```powershell
python train_veldora_small.py
```

### 4) Generate teks dari checkpoint terbaik
```powershell
python gen_sample.py
```

---

## Catatan cepat
- Kalau hanya mau **ulang training** (data/tokenizer sama), **lewati** langkah 1–2.
- Jika OOM: turunkan `MICRO_BSZ` atau `CTX_LEN` di `train_gpt_small.py`.
- `gen_sample.py` memakai `outputs/velora.model` dan checkpoint terbaru di `outputs/checkpoints/`.

---

## Library utama
- **PyTorch** (training & inference)
- **SentencePiece** (tokenizer)
- **Rich** (UI progress & warna)
- **ftfy** (perbaikan teks)
- **tqdm** (progress bar encode)

---

## Versi & penamaan
- Model: **Velora-42M**, **Velora-125M** (ke depan)
- Rilis: **Velora v0.1 “Alpha”**, **v0.2 “Beta”**, **v1.0 “Stable”**
- Folder run: `outputs/checkpoints/velora-YYYYMMDD-HHMM/`
