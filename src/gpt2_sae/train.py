"""Train a TopK SAE on streamed GPT-2 activations.

Writes to <out_root>/<name>/:
  metrics.jsonl   one line per log step
  ckpt.pt         latest SAE + optimizer state, for --resume
  sae.pt          final SAE weights
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

from gpt2_sae.activations import activation_batches, load_model
from gpt2_sae.config import RunConfig
from gpt2_sae.sae import TopKSAE


def lr_at(step: int, cfg: RunConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    decay_start = int(cfg.total_steps * (1 - cfg.decay_frac))
    if step < decay_start:
        return cfg.lr
    return cfg.lr * max(0.0, (cfg.total_steps - step) / (cfg.total_steps - decay_start))


def save_ckpt(path: Path, sae: TopKSAE, opt: torch.optim.Optimizer, step: int) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({"sae": sae.state_dict(), "opt": opt.state_dict(), "step": step}, tmp)
    tmp.replace(path)


def train(cfg: RunConfig, resume: bool = False) -> Path:
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = cfg.resolve_device()
    out = cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "ckpt.pt"

    model = load_model(device)
    sae = TopKSAE(cfg.sae).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr, betas=(0.9, 0.999))

    start = 0
    if resume and ckpt_path.exists():
        blob = torch.load(ckpt_path, map_location=device, weights_only=True)
        sae.load_state_dict(blob["sae"])
        opt.load_state_dict(blob["opt"])
        start = blob["step"]
        print(f"resumed at step {start}")
    elif (out / "metrics.jsonl").exists():
        (out / "metrics.jsonl").unlink()

    batches = activation_batches(
        model,
        hook=cfg.hook,
        layer=cfg.layer,
        seq_len=cfg.seq_len,
        model_batch_rows=cfg.model_batch_rows,
        batch_size=cfg.batch_size,
        buffer_size=cfg.buffer_size,
        skip_tokens=start * cfg.batch_size,
    )

    if start == 0:
        first = next(batches)
        # Scale inputs so the mean norm is sqrt(d_in). This makes lr and init width-independent.
        scale = first.norm(dim=-1).mean() / math.sqrt(cfg.sae.d_in)
        sae.input_scale.copy_(scale)
        sae.init_b_pre(first / scale)
        print(f"input scale {scale.item():.3f}")

    sae.train()
    log = (out / "metrics.jsonl").open("a")
    t0, tokens_since = time.time(), 0
    step = start
    for step in range(start, cfg.total_steps):
        x = next(batches, None)
        if x is None:
            print("data ran out")
            break
        for group in opt.param_groups:
            group["lr"] = lr_at(step, cfg)

        out_ = sae(x / sae.input_scale)
        opt.zero_grad(set_to_none=True)
        out_.loss.backward()
        sae.remove_parallel_grads()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg.grad_clip)
        opt.step()
        sae.normalize_decoder()
        tokens_since += len(x)

        if step % cfg.log_every == 0 or step == cfg.total_steps - 1:
            elapsed = time.time() - t0
            row = {
                "step": step,
                "tokens": (step + 1) * cfg.batch_size,
                "mse": round(out_.mse.item(), 5),
                "aux": round(out_.aux.item(), 5),
                "dead_frac": round(sae.dead_fraction(), 5),
                "lr": lr_at(step, cfg),
                "tok_per_s": round(tokens_since / max(elapsed, 1e-9)),
            }
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(
                f"step {step:>6}/{cfg.total_steps}  mse {row['mse']:.4f}  aux {row['aux']:.4f}  "
                f"dead {row['dead_frac']:.3f}  {row['tok_per_s']:,} tok/s",
                flush=True,
            )
            t0, tokens_since = time.time(), 0

        if step > start and step % cfg.ckpt_every == 0:
            save_ckpt(ckpt_path, sae, opt, step + 1)

    save_ckpt(ckpt_path, sae, opt, step + 1)
    final = out / "sae.pt"
    sae.save(final)
    print(f"saved {final}")
    return final
