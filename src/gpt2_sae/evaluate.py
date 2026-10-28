"""Evaluate an SAE on held-out OpenWebText.

Metrics:
  fvu           fraction of variance unexplained: sum ||x - x_hat||^2 / sum ||x - mean(x)||^2
  l0            mean number of active latents per token
  dead_frac     fraction of latents that never fire on the eval tokens
  ce_clean      GPT-2's next-token loss with no intervention
  ce_spliced    loss with the hooked activation replaced by the SAE reconstruction
  ce_zero       loss with the hooked activation set to zero
  ce_recovered  (ce_zero - ce_spliced) / (ce_zero - ce_clean). 1.0 means no loss from the SAE.
"""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor
from transformer_lens import HookedTransformer

from gpt2_sae.activations import token_batches
from gpt2_sae.sae import TopKSAE


class Dictionary(Protocol):
    """Anything that maps activations to latents and back, in the model's own units."""

    d_sae: int

    def encode(self, x: Tensor) -> Tensor: ...
    def decode(self, z: Tensor) -> Tensor: ...


class OurSAE:
    """Wraps TopKSAE so callers don't need to handle the input scale."""

    def __init__(self, sae: TopKSAE) -> None:
        self.sae = sae.eval()
        self.d_sae = sae.cfg.d_sae

    def encode(self, x: Tensor) -> Tensor:
        return self.sae.encode(x / self.sae.input_scale)

    def decode(self, z: Tensor) -> Tensor:
        return self.sae.decode(z) * self.sae.input_scale

    @property
    def W_dec(self) -> Tensor:
        return self.sae.W_dec


@torch.no_grad()
def evaluate(
    model: HookedTransformer,
    sae: Dictionary,
    *,
    hook: str,
    seq_len: int = 128,
    batch_rows: int = 32,
    n_batches: int = 50,
) -> dict[str, float]:
    device = model.cfg.device
    fired = torch.zeros(sae.d_sae, dtype=torch.bool, device=device)
    sums = dict.fromkeys(("err", "var", "l0", "ce_clean", "ce_spliced", "ce_zero"), 0.0)
    n_tokens = 0

    def splice(acts: Tensor, hook: object) -> Tensor:
        body = acts[:, 1:]
        z = sae.encode(body.reshape(-1, body.shape[-1]))
        x_hat = sae.decode(z).reshape(body.shape)
        # Keep BOS as is. Its activation is an outlier the SAE never trained on.
        return torch.cat([acts[:, :1], x_hat.to(acts.dtype)], dim=1)

    def zero(acts: Tensor, hook: object) -> Tensor:
        return torch.cat([acts[:, :1], torch.zeros_like(acts[:, 1:])], dim=1)

    batches = token_batches(model, "eval", seq_len, batch_rows)
    for _, tokens in zip(range(n_batches), batches, strict=False):
        tokens = tokens.to(device)
        ce_clean, cache = model.run_with_cache(tokens, return_type="loss", names_filter=hook)
        x = cache[hook][:, 1:].reshape(-1, model.cfg.d_model).float()
        z = sae.encode(x)
        x_hat = sae.decode(z)
        fired |= (z > 0).any(0)
        sums["err"] += (x - x_hat).pow(2).sum().item()
        sums["var"] += (x - x.mean(0)).pow(2).sum().item()
        sums["l0"] += (z > 0).sum().item()
        n = len(x)
        n_tokens += n
        sums["ce_clean"] += ce_clean.item() * n
        for name, fn in (("ce_spliced", splice), ("ce_zero", zero)):
            loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(hook, fn)])
            sums[name] += loss.item() * n

    ce = {k: sums[k] / n_tokens for k in ("ce_clean", "ce_spliced", "ce_zero")}
    return {
        "fvu": sums["err"] / sums["var"],
        "l0": sums["l0"] / n_tokens,
        "dead_frac": 1 - fired.float().mean().item(),
        **ce,
        "ce_recovered": (ce["ce_zero"] - ce["ce_spliced"]) / (ce["ce_zero"] - ce["ce_clean"]),
        "n_tokens": n_tokens,
    }
