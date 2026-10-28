"""Run configs. Each YAML file in configs/ maps onto RunConfig."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import torch
import yaml

from gpt2_sae.sae import SAEConfig


@dataclass(frozen=True)
class RunConfig:
    name: str
    sae: SAEConfig = field(default_factory=SAEConfig)
    layer: int = 6
    device: str = "auto"
    # Data
    seq_len: int = 128
    model_batch_rows: int = 64
    buffer_size: int = 262_144
    # Optimization
    total_tokens: int = 50_000_000
    batch_size: int = 4096
    lr: float = 2e-4
    warmup_steps: int = 1000
    decay_frac: float = 0.2  # linear decay to zero over the last 20% of steps
    grad_clip: float = 1.0
    # Evaluation and analysis
    eval_batches: int = 50  # x 32 rows x 127 tokens
    feature_tokens: int = 5_000_000
    n_label: int = 200
    n_export_extra: int = 800
    # Bookkeeping
    log_every: int = 50
    ckpt_every: int = 2000
    out_root: str = "artifacts"
    seed: int = 0

    @property
    def hook(self) -> str:
        return f"blocks.{self.layer}.hook_resid_pre"

    @property
    def out_dir(self) -> Path:
        return Path(self.out_root) / self.name

    @property
    def total_steps(self) -> int:
        return self.total_tokens // self.batch_size

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        raw = yaml.safe_load(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
        sae = SAEConfig(**raw.pop("sae", {}))
        return cls(sae=sae, **raw)
