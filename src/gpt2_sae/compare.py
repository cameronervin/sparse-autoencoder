"""Compare our SAE with the pretrained gpt2-small-res-jb SAE for the same hook.

Both SAEs get the same held-out eval tokens. Features are matched by decoder cosine
similarity: for each of our latents, find the jb latent whose decoder direction is closest.
"""

from __future__ import annotations

import torch
from torch import Tensor

JB_RELEASE = "gpt2-small-res-jb"


class JbSAE:
    """Adapter so the SAELens SAE fits the evaluate.Dictionary protocol."""

    def __init__(self, hook: str, device: str) -> None:
        from sae_lens import SAE

        self.sae = SAE.from_pretrained(JB_RELEASE, hook, device=device).eval()
        self.d_sae = self.sae.cfg.d_sae

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        return self.sae.encode(x)

    @torch.no_grad()
    def decode(self, z: Tensor) -> Tensor:
        return self.sae.decode(z)

    @property
    def W_dec(self) -> Tensor:
        return self.sae.W_dec


@torch.no_grad()
def best_matches(ours: Tensor, theirs: Tensor, chunk: int = 2048) -> tuple[Tensor, Tensor]:
    """For each row of `ours`, the max cosine similarity to any row of `theirs`, and its index."""
    a = torch.nn.functional.normalize(ours.float(), dim=-1)
    b = torch.nn.functional.normalize(theirs.float().to(a.device), dim=-1)
    best_cos, best_idx = [], []
    for i in range(0, len(a), chunk):
        sims = a[i : i + chunk] @ b.T
        c, j = sims.max(dim=-1)
        best_cos.append(c)
        best_idx.append(j)
    return torch.cat(best_cos).cpu(), torch.cat(best_idx).cpu()
