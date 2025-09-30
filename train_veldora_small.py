import json, math, time, numpy as np
from pathlib import Path
import torch, torch.nn.functional as F
from torch import nn

BASE_DIR     = Path(__file__).resolve().parent
PACKED_DIR   = Path("outputs/packed")
OUT_DIR      = BASE_DIR / "outputs" / "checkpoints"
SEED         = 1337

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
if device == "cuda":
    torch.cuda.manual_seed_all(SEED)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

CTX_LEN  = 512
N_LAYER  = 6
N_HEAD   = 8
D_MODEL  = 512
FFN_MULT = 4
DROPOUT  = 0.0

STEPS        = 35_000       
BSZ_TOKENS   = 8_192        
MICRO_BSZ    = 4             
LR           = 3e-4
WEIGHT_DECAY = 0.1
WARMUP       = 500
VAL_EVERY    = 1_000
SAVE_EVERY   = 2_000
AMP          = (device == "cuda")   
VAL_RATIO    = 0.01


meta = json.loads((PACKED_DIR / "meta_veldora.json").read_text(encoding="utf-8"))
VOCAB_SIZE = int(meta["vocab_size"])

tokens = np.memmap(PACKED_DIR / "tokens_veldora.bin", dtype=np.uint32, mode="r")
N = tokens.shape[0]
VAL_N = max(CTX_LEN * 2048, int(N * VAL_RATIO))
train_tokens = tokens[: N - VAL_N]
val_tokens   = tokens[N - VAL_N :]

def sample_from(arr: np.memmap, ctx_len: int, micro_bsz: int):
    ix = np.random.randint(0, arr.shape[0] - ctx_len - 1, size=(micro_bsz,))
    x = np.stack([arr[i : i + ctx_len] for i in ix])
    y = np.stack([arr[i + 1 : i + 1 + ctx_len] for i in ix])
    x = torch.from_numpy(x.astype(np.int64)).to(device, non_blocking=True)
    y = torch.from_numpy(y.astype(np.int64)).to(device, non_blocking=True)
    return x, y

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x

class MLP(nn.Module):
    def __init__(self, d_model, mult=4, dropout=0.0):
        super().__init__()
        hidden = int(mult * d_model)
        self.fc1 = nn.Linear(d_model, hidden * 2, bias=False)  
        self.fc2 = nn.Linear(hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        a, b = self.fc1(x).chunk(2, dim=-1)
        x = F.silu(a) * b
        x = self.fc2(x)
        return self.drop(x)

class Block(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.mlp = MLP(d_model, mult=FFN_MULT, dropout=dropout)

    def forward(self, x):
        B, T, C = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(B, T, 3, self.n_head, self.head_dim).permute(0, 3, 1, 2, 4)
        q, k, v = qkv.unbind(dim=3)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        attn = attn.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        x = x + self.drop(self.proj(attn))
        x = x + self.mlp(self.norm2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layer, n_head, ctx_len, dropout=0.0):
        super().__init__()
        self.ctx_len = ctx_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, ctx_len, d_model))  
        self.blocks = nn.ModuleList([Block(d_model, n_head, dropout) for _ in range(n_layer)])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
    def forward(self, idx):
        B, T = idx.size()
        x = self.tok_emb(idx) + self.pos_emb[:, :T, :]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x)
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=0.8, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.ctx_len:]
            logits = self(idx_cond)[:, -1, :] / max(temperature, 1e-6)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_tok), dim=1)
        return idx

model = TinyGPT(VOCAB_SIZE, D_MODEL, N_LAYER, N_HEAD, CTX_LEN, dropout=DROPOUT).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Device: {device} | Params: {n_params/1e6:.2f}M | Vocab: {VOCAB_SIZE} | CTX {CTX_LEN}")

OUT_DIR.mkdir(parents=True, exist_ok=True)

opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=WEIGHT_DECAY)
scaler = torch.amp.GradScaler(device if AMP else "cpu")

tokens_per_micro = MICRO_BSZ * CTX_LEN
accum_steps = max(1, BSZ_TOKENS // tokens_per_micro)
print(f"MICRO_BSZ={MICRO_BSZ}, accum_steps={accum_steps}, effective_tokens/step={accum_steps * tokens_per_micro}")

def lr_schedule(step: int):
    if step < WARMUP:
        return LR * step / max(1, WARMUP)
    progress = (step - WARMUP) / max(1, STEPS - WARMUP)
    return 0.1 * LR + 0.9 * LR * 0.5 * (1.0 + math.cos(math.pi * progress))

@torch.no_grad()
def eval_val(iters=40):
    model.eval()
    losses = []
    with torch.amp.autocast(device_type=device if AMP else "cpu"):
        for _ in range(iters):
            x, y = sample_from(val_tokens, CTX_LEN, MICRO_BSZ)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
            losses.append(loss.item())
    model.train()
    m = sum(losses) / len(losses)
    return m, math.exp(m)

def save_ckpt(step, tag=None):
    path = OUT_DIR / (f"step{step:06d}.pt" if tag is None else f"{tag}.pt")
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "cfg": {
            "VOCAB_SIZE": VOCAB_SIZE, "CTX_LEN": CTX_LEN,
            "N_LAYER": N_LAYER, "N_HEAD": N_HEAD, "D_MODEL": D_MODEL,
            "FFN_MULT": FFN_MULT, "DROPOUT": DROPOUT
        }
    }, path)
    print("saved", path)

model.train()
t0 = time.time()
best_val = float("inf")

for step in range(1, STEPS + 1):
    opt.zero_grad(set_to_none=True)
    loss_acc = 0.0

    for _ in range(accum_steps):
        x, y = sample_from(train_tokens, CTX_LEN, MICRO_BSZ)
        with torch.amp.autocast(device_type=device if AMP else "cpu"):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
            loss = loss / accum_steps  
        scaler.scale(loss).backward()
        loss_acc += loss.item()

    new_lr = lr_schedule(step)
    for g in opt.param_groups:
        g["lr"] = new_lr

    scaler.step(opt)
    scaler.update()

    if step % 50 == 0:
        dt = time.time() - t0
        tok_per_sec = (accum_steps * tokens_per_micro) / max(1e-6, dt)
        print(f"step {step:6d}/{STEPS} | lr {new_lr:.6f} | loss {loss_acc:.4f} | {tok_per_sec:,.0f} tok/s | {dt:.1f}s")
        t0 = time.time()

    if step % VAL_EVERY == 0 or step == 1:
        vloss, ppl = eval_val(iters=40)
        print(f"[val] step {step:6d} | loss {vloss:.4f} | ppl {ppl:.1f}")
        if vloss < best_val:
            best_val = vloss
            save_ckpt(step, tag="best-val")

    if step % SAVE_EVERY == 0 or step == STEPS:
        save_ckpt(step)
