"""Label features with Claude Haiku, from their top-activating contexts and logit-lens tokens.

Needs ANTHROPIC_API_KEY (or another credential the Anthropic SDK can find). Without one,
labels come from configs/hand_labels.yaml instead.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
import yaml
from pydantic import BaseModel, Field
from torch import Tensor
from tqdm import tqdm

from gpt2_sae.features import window

MODEL = "claude-haiku-5-5"

SYSTEM = """You label features of a sparse autoencoder trained on GPT-2 small's residual stream.
You get text snippets where one feature fires. In each snippet the token where it fires most
is wrapped in [[ ]]. You also get the output tokens the feature pushes the model toward.
Give a short label (2 to 6 words) for what the feature detects, a one-sentence description,
and a confidence from 0 to 1. Use low confidence when the snippets have no clear pattern."""


class Label(BaseModel):
    label: str = Field(description="2 to 6 words")
    description: str = Field(description="one sentence")
    confidence: float = Field(ge=0, le=1)


def live_features(stats: dict[str, Tensor], min_count: int = 20) -> list[int]:
    return (stats["counts"] >= min_count).nonzero().flatten().tolist()


def sample_features(stats: dict[str, Tensor], n: int, seed: int = 0) -> list[int]:
    """A random sample of live features, so the labeled set is not cherry-picked."""
    live = live_features(stats)
    return sorted(random.Random(seed).sample(live, min(n, len(live))))


def prompt_for(stats: dict[str, Tensor], tokenizer, feature: int, n_snippets: int = 10) -> str:
    lines = []
    for idx, val in zip(stats["top_idx"][feature], stats["top_vals"][feature], strict=True):
        if idx < 0 or len(lines) == n_snippets:
            break
        ids, at = window(stats, int(idx))
        text = (
            tokenizer.decode(ids[:at])
            + "[["
            + tokenizer.decode(ids[at : at + 1])
            + "]]"
            + tokenizer.decode(ids[at + 1 :])
        )
        lines.append(f"({val:.1f}) {text!r}")
    promoted = [tokenizer.decode([t]) for t in stats["logit_up"][feature].tolist()]
    return (
        "Snippets (activation in parentheses):\n"
        + "\n".join(lines)
        + f"\n\nPromoted output tokens: {promoted!r}"
    )


def label_features(
    stats: dict[str, Tensor], tokenizer, features: list[int], out_path: Path
) -> dict[int, dict]:
    """Label each feature, appending results to out_path (JSONL) so a rerun skips done ones."""
    import anthropic

    client = anthropic.Anthropic()
    done: dict[int, dict] = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            row = json.loads(line)
            done[row["feature"]] = row
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as f:
        for feature in tqdm([x for x in features if x not in done]):
            response = client.messages.parse(
                model=MODEL,
                max_tokens=1024,
                output_config={"effort": "low"},
                system=SYSTEM,
                messages=[{"role": "user", "content": prompt_for(stats, tokenizer, feature)}],
                output_format=Label,
            )
            if response.stop_reason == "refusal" or response.parsed_output is None:
                continue
            row = {"feature": feature, "source": MODEL, **response.parsed_output.model_dump()}
            f.write(json.dumps(row) + "\n")
            f.flush()
            done[feature] = row
    return done


def hand_labels(path: Path) -> dict[int, dict]:
    """configs/hand_labels.yaml maps feature id -> {label, description}."""
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    return {
        int(k): {"feature": int(k), "source": "hand", "confidence": 1.0, **v}
        for k, v in raw.items()
    }


def load_labels(labels_path: Path, hand_path: Path) -> dict[int, dict]:
    """Model labels, overridden by hand labels where both exist."""
    labels: dict[int, dict] = {}
    if labels_path.exists():
        for line in labels_path.read_text().splitlines():
            row = json.loads(line)
            labels[row["feature"]] = row
    labels.update(hand_labels(hand_path))
    return labels


def load_stats(path: Path) -> dict[str, Tensor]:
    return torch.load(path, weights_only=True)
