"""Collect per-feature statistics over held-out text.

For every latent this records:
  - the top-K activating token positions (value and global token index)
  - how often it fires, and a histogram of log10 activation values
  - the output tokens its decoder direction promotes and suppresses (logit lens: W_dec @ W_U)

Token ids for all processed rows are kept, so contexts can be rebuilt later without re-reading
the dataset.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from tqdm import tqdm
from transformer_lens import HookedTransformer

from gpt2_sae.activations import token_batches
from gpt2_sae.evaluate import Dictionary

HIST_BINS = 40
HIST_LO, HIST_HI = -3.0, 2.0  # log10 activation range covered by the histogram


@torch.no_grad()
def collect(
    model: HookedTransformer,
    sae: Dictionary,
    *,
    hook: str,
    layer: int,
    n_tokens: int,
    top_k: int = 20,
    seq_len: int = 128,
    batch_rows: int = 32,
) -> dict[str, Tensor]:
    device = model.cfg.device
    d = sae.d_sae
    top_vals = torch.zeros(d, top_k, device=device)
    top_idx = torch.full((d, top_k), -1, dtype=torch.long, device=device)
    counts = torch.zeros(d, dtype=torch.long, device=device)
    hist = torch.zeros(d * HIST_BINS, dtype=torch.long, device=device)
    rows: list[Tensor] = []
    body = seq_len - 1
    n_batches = max(1, n_tokens // (batch_rows * body))

    batches = token_batches(model, "eval", seq_len, batch_rows)
    for b, tokens in tqdm(zip(range(n_batches), batches, strict=False), total=n_batches):
        rows.append(tokens)
        _, cache = model.run_with_cache(
            tokens.to(device), names_filter=hook, stop_at_layer=layer + 1
        )
        x = cache[hook][:, 1:].reshape(-1, model.cfg.d_model).float()
        z = sae.encode(x)  # (n, d)

        active = z > 0
        counts += active.sum(0)
        tok_i, lat_i = active.nonzero(as_tuple=True)
        vals = z[tok_i, lat_i].log10().clamp(HIST_LO, HIST_HI - 1e-6)
        bins = ((vals - HIST_LO) / (HIST_HI - HIST_LO) * HIST_BINS).long()
        hist.index_add_(0, lat_i * HIST_BINS + bins, torch.ones_like(bins))

        # Merge this batch's top-K per latent into the running top-K.
        k = min(top_k, len(z))
        bv, bi = z.T.topk(k, dim=-1)
        bi = bi + b * batch_rows * body  # global index = row * body + position
        merged_v = torch.cat([top_vals, bv], dim=-1)
        merged_i = torch.cat([top_idx, bi], dim=-1)
        top_vals, order = merged_v.topk(top_k, dim=-1)
        top_idx = merged_i.gather(-1, order)

    top_idx[top_vals <= 0] = -1
    W_dec = sae.W_dec.float().to(device)
    W_dec = W_dec / W_dec.norm(dim=-1, keepdim=True)
    logits = W_dec @ model.W_U  # (d, vocab)
    up_v, up_i = logits.topk(10, dim=-1)
    down_v, down_i = (-logits).topk(10, dim=-1)

    return {
        "top_vals": top_vals.cpu(),
        "top_idx": top_idx.cpu(),
        "counts": counts.cpu(),
        "hist": hist.view(d, HIST_BINS).cpu(),
        "tokens": torch.cat(rows).to(torch.int32),  # (n_rows, seq_len), BOS first
        "n_tokens": torch.tensor(len(rows) * batch_rows * body),
        "logit_up": up_i.cpu(),
        "logit_up_vals": up_v.cpu(),
        "logit_down": down_i.cpu(),
        "logit_down_vals": (-down_v).cpu(),
    }


def save(stats: dict[str, Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, path)


@torch.no_grad()
def context_acts(
    model: HookedTransformer,
    sae: Dictionary,
    tokens: Tensor,
    *,
    hook: str,
    layer: int,
    latent: int,
) -> Tensor:
    """Activations of one latent at every non-BOS position of `tokens` (rows, seq_len)."""
    _, cache = model.run_with_cache(
        tokens.to(model.cfg.device).long(), names_filter=hook, stop_at_layer=layer + 1
    )
    x = cache[hook][:, 1:]
    z = sae.encode(x.reshape(-1, x.shape[-1]).float())[:, latent]
    return z.view(x.shape[0], x.shape[1]).cpu()


def locate(stats: dict[str, Tensor], global_idx: int) -> tuple[int, int]:
    """Map a global token index to (row, position), where position counts from after BOS."""
    body = stats["tokens"].shape[1] - 1
    return global_idx // body, global_idx % body


def window(
    stats: dict[str, Tensor], global_idx: int, before: int = 24, after: int = 4
) -> tuple[list[int], int]:
    """Token ids around one activating position, and the index of that position in the list."""
    row, pos = locate(stats, global_idx)
    ids = stats["tokens"][row, 1:].tolist()
    lo = max(0, pos - before)
    return ids[lo : pos + after + 1], pos - lo
