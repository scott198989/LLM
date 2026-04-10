"""
GPT-style language model training script.
Loads preprocessed token data from data/processed/ (output of preprocess.py).

Usage:
    python scripts/train.py
    python scripts/train.py --processed_dir data/processed --batch_size 8
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dataset import build_dataloaders

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class Config:
    # Model — ~30M params at GPT-2 vocab (50258)
    # n_layer=6, n_head=8, n_embd=512: ~18.9M block params + 50258*512=25.7M embed = ~44M total
    # For exactly ~30M: keep n_layer=4 or reduce n_embd to 384
    # vocab_size / block_size are filled in from tokenizer_info.json at runtime
    vocab_size   = 50258      # GPT-2 vocab + <|sep|>; overridden from processed data
    n_layer      = 6
    n_head       = 8
    n_embd       = 512
    block_size   = 512
    dropout      = 0.1

    # Training — tuned for RTX 2050 4GB VRAM
    batch_size   = 16         # reduce to 8 if OOM
    lr           = 3e-4
    weight_decay = 0.1
    max_epochs   = 5
    grad_clip    = 1.0
    eval_interval= 200        # steps between val loss checks

    # Paths
    processed_dir = "data/processed"
    ckpt_dir      = "models/checkpoints"
    log_dir       = "logs"

    # Hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


cfg = Config()

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.c_attn  = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj  = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.drop    = nn.Dropout(cfg.dropout)
        self.n_head  = cfg.n_head
        self.n_embd  = cfg.n_embd

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        def reshape(t):
            return t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q, k, v = reshape(q), reshape(k), reshape(v)
        # Flash attention when available (PyTorch 2+)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.c_proj(out))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp  = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop    = nn.Dropout(cfg.dropout)
        self.blocks  = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f    = nn.LayerNorm(cfg.n_embd)
        self.head    = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Weight tying
        self.tok_emb.weight = self.head.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x   = self.drop(tok + pos)
        x   = self.blocks(x)
        x   = self.ln_f(x)
        logits = self.head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.pos_emb.num_embeddings:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(model, val_loader, device, dtype):
    model.eval()
    losses = []
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
            _, loss = model(x, y)
        losses.append(loss.item())
        if len(losses) >= 50:   # cap eval batches to keep it fast
            break
    return sum(losses) / len(losses)


def train(args):
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    print(f"Device : {cfg.device}")
    if cfg.device == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # Data — load from processed directory
    train_loader, val_loader, info = build_dataloaders(
        processed_dir=args.processed_dir,
        batch_size=cfg.batch_size,
    )
    cfg.vocab_size = info["vocab_size"]
    cfg.block_size = info["block_size"]
    print(f"Vocab  : {cfg.vocab_size:,}")
    print(f"Train batches: {len(train_loader):,}  Val batches: {len(val_loader):,}")

    # Model
    model = GPT(cfg).to(cfg.device)
    n_params = count_params(model)
    print(f"Params : {n_params:,}  (~{n_params/1e6:.1f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler    = torch.cuda.amp.GradScaler(enabled=(cfg.device == "cuda"))

    # Try to load GPT-2 tokenizer for generation samples
    try:
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|sep|>"]})
    except Exception:
        tokenizer = None

    log_path = os.path.join(cfg.log_dir, "train_log.jsonl")
    step = 0
    best_val = float("inf")

    for epoch in range(cfg.max_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.max_epochs}")
        for x, y in pbar:
            x, y = x.to(cfg.device), y.to(cfg.device)
            with torch.autocast(device_type=cfg.device, dtype=cfg.dtype, enabled=(cfg.device == "cuda")):
                _, loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            pbar.set_postfix(loss=f"{loss.item():.4f}", step=step)

            if step % cfg.eval_interval == 0 and step > 0:
                val_loss = evaluate(model, val_loader, cfg.device, cfg.dtype)
                print(f"\n  [step {step}] val_loss={val_loss:.4f}")
                with open(log_path, "a") as f:
                    import json
                    f.write(json.dumps({"step": step, "val_loss": val_loss}) + "\n")
                model.train()

            step += 1

        # Epoch-end eval + checkpoint
        val_loss = evaluate(model, val_loader, cfg.device, cfg.dtype)
        print(f"\nEpoch {epoch+1} complete — val_loss={val_loss:.4f}")

        tag = "best" if val_loss < best_val else f"epoch_{epoch+1}"
        if val_loss < best_val:
            best_val = val_loss
        ckpt_path = os.path.join(cfg.ckpt_dir, f"{tag}.pt")
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg": vars(cfg),
            "epoch": epoch + 1,
            "step": step,
            "val_loss": val_loss,
        }, ckpt_path)
        print(f"Saved: {ckpt_path}")

        # Generate a sample
        model.eval()
        eot_id = info.get("eot_token_id", 50256)
        ctx = torch.tensor([[eot_id]], dtype=torch.long, device=cfg.device)
        sample_ids = model.generate(ctx, max_new_tokens=150, temperature=0.8, top_k=40)
        if tokenizer:
            decoded = tokenizer.decode(sample_ids[0].tolist(), skip_special_tokens=False)
        else:
            decoded = str(sample_ids[0].tolist()[:40]) + " ..."
        print(f"\n--- Generation sample ---\n{decoded[:400]}\n-------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default=cfg.processed_dir)
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    args = parser.parse_args()
    cfg.batch_size = args.batch_size
    train(args)
