"""Stream OpenWebText through GPT-2 and serve shuffled residual-stream activations.

Nothing is cached to disk. Documents are tokenized, packed into fixed-length rows that start
with BOS, run through the model, and the hooked activations go into a shuffle buffer.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from datasets import load_dataset
from torch import Tensor
from transformer_lens import HookedTransformer

DATASET = "Skylion007/openwebtext"
# Every 100th document is held out for evaluation. The rest is for training.
EVAL_EVERY = 100


def load_model(device: str) -> HookedTransformer:
    model = HookedTransformer.from_pretrained("gpt2", device=device)
    model.eval()
    return model


def documents(split: str) -> Iterator[str]:
    ds = load_dataset(DATASET, split="train", streaming=True)
    for i, ex in enumerate(ds):
        held_out = i % EVAL_EVERY == 0
        if held_out == (split == "eval"):
            yield ex["text"]


def token_rows(model: HookedTransformer, split: str, seq_len: int) -> Iterator[Tensor]:
    """Yield rows of seq_len tokens: BOS followed by seq_len - 1 tokens of packed text."""
    bos = model.tokenizer.bos_token_id
    body = seq_len - 1
    pending: list[int] = []
    for text in documents(split):
        pending.extend(model.tokenizer.encode(text))
        pending.append(model.tokenizer.eos_token_id)
        while len(pending) >= body:
            yield torch.tensor([bos, *pending[:body]])
            pending = pending[body:]


def token_batches(
    model: HookedTransformer, split: str, seq_len: int, batch_rows: int, skip_rows: int = 0
) -> Iterator[Tensor]:
    rows: list[Tensor] = []
    for i, row in enumerate(token_rows(model, split, seq_len)):
        if i < skip_rows:
            continue
        rows.append(row)
        if len(rows) == batch_rows:
            yield torch.stack(rows)
            rows = []


@torch.no_grad()
def hook_acts(model: HookedTransformer, tokens: Tensor, hook: str, layer: int) -> Tensor:
    """Activations at `hook` for every non-BOS position, flattened to (n_tokens, d_model)."""
    _, cache = model.run_with_cache(
        tokens.to(model.cfg.device), names_filter=hook, stop_at_layer=layer + 1
    )
    acts = cache[hook][:, 1:]  # drop BOS: its norm is far larger than any other position
    return acts.reshape(-1, acts.shape[-1]).float()


def activation_batches(
    model: HookedTransformer,
    *,
    hook: str,
    layer: int,
    split: str = "train",
    seq_len: int = 128,
    model_batch_rows: int = 64,
    batch_size: int = 4096,
    buffer_size: int = 262_144,
    skip_tokens: int = 0,
) -> Iterator[Tensor]:
    """Yield shuffled (batch_size, d_model) activation batches until the data runs out.

    The buffer refills to buffer_size, shuffles, then serves batches until it is half empty.
    That keeps consecutive batches from coming from the same few documents.
    skip_tokens drops that many activations' worth of rows first, so a resumed run sees new text.
    """
    device = model.cfg.device
    skip_rows = skip_tokens // (seq_len - 1)
    tokens = token_batches(model, split, seq_len, model_batch_rows, skip_rows)
    buf = torch.empty(0, model.cfg.d_model, device=device)
    exhausted = False
    while True:
        chunks = [buf]
        n = len(buf)
        while n < buffer_size and not exhausted:
            batch = next(tokens, None)
            if batch is None:
                exhausted = True
                break
            acts = hook_acts(model, batch, hook, layer)
            chunks.append(acts)
            n += len(acts)
        buf = torch.cat(chunks)
        buf = buf[torch.randperm(len(buf), device=device)]
        keep = 0 if exhausted else buffer_size // 2
        while len(buf) - batch_size >= keep:
            yield buf[:batch_size]
            buf = buf[batch_size:]
        if exhausted:
            return
