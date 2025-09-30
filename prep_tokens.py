import json, numpy as np
from pathlib import Path
import sentencepiece as spm
from tqdm import tqdm

BASE = Path(__file__).resolve().parent
DATA_TXT  = BASE / "outputs" / "wiki-mini" / "data.txt"
SPM_MODEL = BASE / "outputs" / "veldora_tokenizer.model"
OUT_DIR   = BASE / "outputs" / "packed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOK_BIN   = OUT_DIR / "tokens_veldora.bin"
META_JSON = OUT_DIR / "meta_veldora.json"

def main():
    if not DATA_TXT.exists():
        raise SystemExit(f"data.txt not found: {DATA_TXT}")
    if not SPM_MODEL.exists():
        raise SystemExit(f"tokenizer model not found: {SPM_MODEL}")

    sp = spm.SentencePieceProcessor(model_file=str(SPM_MODEL))
    vocab_size = sp.get_piece_size()
    nl_ids = sp.encode("\n", out_type=int)

    if TOK_BIN.exists():
        TOK_BIN.unlink()
    buf = []

    with open(DATA_TXT, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Encoding"):
            s = line.strip()
            if not s: continue
            ids = sp.encode(s, out_type=int) + nl_ids
            buf.extend(ids)
            if len(buf) >= 2_000_000:
                arr = np.asarray(buf, dtype=np.uint32)
                prev = (TOK_BIN.stat().st_size // 4) if TOK_BIN.exists() else 0
                mm = np.memmap(TOK_BIN, dtype=np.uint32, mode=("r+" if TOK_BIN.exists() else "w+"), shape=(prev+arr.size,))
                mm[prev:prev+arr.size] = arr; mm.flush(); del mm
                buf.clear()

    if buf:
        arr = np.asarray(buf, dtype=np.uint32)
        prev = (TOK_BIN.stat().st_size // 4) if TOK_BIN.exists() else 0
        mm = np.memmap(TOK_BIN, dtype=np.uint32, mode=("r+" if TOK_BIN.exists() else "w+"), shape=(prev+arr.size,))
        mm[prev:prev+arr.size] = arr; mm.flush(); del mm

    meta = {"vocab_size": int(vocab_size), "spm_model": str(SPM_MODEL), "dtype": "uint32"}
    META_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Done ->", TOK_BIN, "| meta ->", META_JSON)

if __name__ == "__main__":
    main()
