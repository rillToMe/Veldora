import torch, sentencepiece as spm
from train_veldora_small import TinyGPT, CTX_LEN, D_MODEL, N_LAYER, N_HEAD, VOCAB_SIZE
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CKPT = BASE_DIR / "outputs" / "checkpoints" / "best-val.pt"
SPM  = BASE_DIR / "outputs" / "mytokenizer.model"

sp = spm.SentencePieceProcessor(model_file=SPM)
m = TinyGPT(VOCAB_SIZE, D_MODEL, N_LAYER, N_HEAD, CTX_LEN).cuda().eval()
state = torch.load(CKPT, map_location="cuda"); m.load_state_dict(state["model"])

prompt = "Halo, siapa nama kamu?"
x = torch.tensor([sp.encode(prompt, out_type=int)], device="cuda")
y = m.generate(x, max_new_tokens=120, temperature=0.9, top_k=80)[0].tolist()
print(sp.decode(y))
