"""Steer GPT-2 by adding a feature's decoder direction to the residual stream.

The added vector is coef * max_act * direction, where max_act is the feature's largest
activation on the eval text. So coef = 1 adds the feature as strongly as it ever fires
naturally. The vector is added at every position after BOS, during the prompt and generation.
"""

from __future__ import annotations

import torch
from torch import Tensor
from transformer_lens import HookedTransformer

PROMPTS = [
    "I think the most important thing is",
    "Yesterday I went to the",
    "The new report says that",
]


def steering_vector(W_dec: Tensor, feature: int, max_act: float, scale: float = 1.0) -> Tensor:
    """Decoder row in the model's units. `scale` undoes any input normalization of the SAE."""
    return W_dec[feature].float() * max_act * scale


@torch.no_grad()
def generate(
    model: HookedTransformer,
    prompt: str,
    *,
    hook: str,
    vector: Tensor | None,
    coef: float,
    max_new_tokens: int = 40,
    seed: int = 0,
) -> str:
    def add(acts: Tensor, hook: object) -> Tensor:
        acts[:, 1:] += coef * vector.to(acts.device, acts.dtype)
        return acts

    torch.manual_seed(seed)
    hooks = [(hook, add)] if vector is not None and coef != 0 else []
    with model.hooks(fwd_hooks=hooks):
        out = model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_p=0.9,
            use_past_kv_cache=False,  # the hook must see every position on every step
            verbose=False,
        )
    return out[len(prompt) :] if isinstance(out, str) else out


def steer_feature(
    model: HookedTransformer,
    vector: Tensor,
    *,
    hook: str,
    coefs: tuple[float, ...] = (0.0, 1.0, 2.0),
    prompts: list[str] = PROMPTS,
    seed: int = 0,
) -> list[dict]:
    rows = []
    for prompt in prompts:
        for coef in coefs:
            text = generate(model, prompt, hook=hook, vector=vector, coef=coef, seed=seed)
            rows.append({"prompt": prompt, "coef": coef, "text": text})
    return rows
