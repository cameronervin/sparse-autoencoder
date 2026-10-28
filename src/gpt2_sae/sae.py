"""TopK sparse autoencoder.

Follows Gao et al. (2024), "Scaling and evaluating sparse autoencoders".

Each input keeps only its k largest latent activations, so the L0 is fixed at k and no L1
penalty is needed. Dead latents get a gradient from the auxiliary loss (aux-k).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SAEConfig:
    d_in: int = 768
    d_sae: int = 24576
    k: int = 32
    # Latents in the aux-k loss. Gao et al. use d_in / 2.
    k_aux: int = 384
    aux_coef: float = 1 / 32
    # A latent is dead if it has not fired in this many tokens.
    dead_after_tokens: int = 10_000_000


@dataclass
class SAEOutput:
    x_hat: Tensor
    latents: Tensor  # (batch, d_sae), dense with k nonzeros per row
    mse: Tensor  # normalized: ||x - x_hat||^2 / ||x - mean(x)||^2
    aux: Tensor
    loss: Tensor


class TopKSAE(nn.Module):
    def __init__(self, cfg: SAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.b_pre = nn.Parameter(torch.zeros(cfg.d_in))
        self.W_enc = nn.Parameter(torch.empty(cfg.d_in, cfg.d_sae))
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_sae))
        self.W_dec = nn.Parameter(torch.empty(cfg.d_sae, cfg.d_in))
        # Tokens since each latent last fired. Used to find dead latents.
        self.register_buffer("since_fired", torch.zeros(cfg.d_sae, dtype=torch.long))
        # Activations are divided by this before encoding, so E[||x||] = sqrt(d_in).
        self.register_buffer("input_scale", torch.ones(()))
        self.since_fired: Tensor
        self.input_scale: Tensor

        nn.init.kaiming_uniform_(self.W_dec)
        self.normalize_decoder()
        # Tied init: the encoder starts as the decoder's transpose.
        with torch.no_grad():
            self.W_enc.copy_(self.W_dec.T)

    # -- forward pass --------------------------------------------------------------------------

    def pre_acts(self, x: Tensor) -> Tensor:
        return (x - self.b_pre) @ self.W_enc + self.b_enc

    def _topk(self, pre: Tensor) -> Tensor:
        vals, idx = pre.topk(self.cfg.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, idx, vals.relu())

    def encode(self, x: Tensor) -> Tensor:
        """Return dense latents with at most k nonzeros per row."""
        return self._topk(self.pre_acts(x))

    def decode(self, latents: Tensor) -> Tensor:
        return latents @ self.W_dec + self.b_pre

    def forward(self, x: Tensor) -> SAEOutput:
        pre = self.pre_acts(x)
        latents = self._topk(pre)
        x_hat = self.decode(latents)

        var = (x - x.mean(0)).pow(2).sum(-1).mean()
        mse = (x - x_hat).pow(2).sum(-1).mean() / var
        aux = self._aux_loss(x, x_hat, pre)
        if self.training:
            self._update_dead(latents, x.shape[0])
        return SAEOutput(x_hat, latents, mse, aux, mse + self.cfg.aux_coef * aux)

    def _aux_loss(self, x: Tensor, x_hat: Tensor, pre: Tensor) -> Tensor:
        """Reconstruct the residual error using only the dead latents."""
        dead = self.since_fired >= self.cfg.dead_after_tokens
        n_dead = int(dead.sum())
        if n_dead == 0:
            return x.new_zeros(())
        k_aux = min(self.cfg.k_aux, n_dead)
        masked = pre.masked_fill(~dead, float("-inf"))
        vals, idx = masked.topk(k_aux, dim=-1)
        aux_latents = torch.zeros_like(pre).scatter_(-1, idx, vals.relu())
        err = (x - x_hat).detach()
        err_hat = aux_latents @ self.W_dec
        aux = (err - err_hat).pow(2).sum(-1).mean() / err.pow(2).sum(-1).mean()
        return aux.nan_to_num(0.0)

    @torch.no_grad()
    def _update_dead(self, latents: Tensor, n_tokens: int) -> None:
        fired = (latents > 0).any(0)
        self.since_fired += n_tokens
        self.since_fired[fired] = 0

    # -- decoder constraint --------------------------------------------------------------------

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.W_dec.div_(self.W_dec.norm(dim=-1, keepdim=True).clamp_min(1e-8))

    @torch.no_grad()
    def remove_parallel_grads(self) -> None:
        """Drop the gradient component along each decoder row, so Adam steps stay on the sphere."""
        if self.W_dec.grad is None:
            return
        parallel = (self.W_dec.grad * self.W_dec).sum(-1, keepdim=True)
        self.W_dec.grad.sub_(parallel * self.W_dec)

    @torch.no_grad()
    def init_b_pre(self, x: Tensor) -> None:
        """Start the pre-encoder bias at the data mean."""
        self.b_pre.copy_(x.mean(0))

    def dead_fraction(self) -> float:
        return float((self.since_fired >= self.cfg.dead_after_tokens).float().mean())

    # -- save / load ---------------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": asdict(self.cfg), "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path: Path, device: str | torch.device = "cpu") -> TopKSAE:
        blob = torch.load(path, map_location=device, weights_only=True)
        sae = cls(SAEConfig(**blob["cfg"]))
        sae.load_state_dict(blob["state"])
        return sae.to(device)
