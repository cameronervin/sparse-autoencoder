"""Matplotlib figures for the README."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch import Tensor


def training_curves(metrics_path: Path, out: Path) -> None:
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    tokens = [r["tokens"] / 1e6 for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(tokens, [r["mse"] for r in rows])
    axes[0].set(xlabel="tokens (M)", ylabel="normalized MSE", yscale="log", title="Reconstruction")
    axes[1].plot(tokens, [r["dead_frac"] for r in rows])
    axes[1].set(xlabel="tokens (M)", ylabel="fraction", title="Dead latents")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def match_histogram(cos: Tensor, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(cos.numpy(), bins=50)
    ax.set(
        xlabel="max cosine similarity to a jb latent",
        ylabel="our latents",
        title="Decoder direction match",
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
