"""Write the JSON files the static feature browser in site/ reads.

site/data/
  summary.json            eval metrics, training curve, match histogram, steering samples
  index.json              one entry per exported feature, for search and the list view
  features/<id>.json      contexts with per-token activations, histogram, logit tokens, match
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch import Tensor
from transformer_lens import HookedTransformer

from gpt2_sae.evaluate import Dictionary
from gpt2_sae.features import HIST_BINS, HIST_HI, HIST_LO, locate

NEURONPEDIA = "https://www.neuronpedia.org/gpt2-small/6-res-jb/{}"


def pick_features(
    stats: dict[str, Tensor], labels: dict[int, dict], n_extra: int, seed: int = 0
) -> list[int]:
    """All labeled features plus a random sample of other live ones."""
    live = set((stats["counts"] >= 20).nonzero().flatten().tolist())
    chosen = set(labels) & live
    rest = sorted(live - chosen)
    chosen |= set(random.Random(seed).sample(rest, min(n_extra, len(rest))))
    return sorted(chosen)


@torch.no_grad()
def context_records(
    model: HookedTransformer,
    sae: Dictionary,
    stats: dict[str, Tensor],
    features: list[int],
    *,
    hook: str,
    layer: int,
    n_contexts: int = 10,
    before: int = 24,
    after: int = 6,
    batch_rows: int = 16,
) -> dict[int, list[dict]]:
    """For each feature, its top contexts as token strings with that feature's activation per token.

    Rows needed by any feature are run through the model once, in batches.
    """
    tok = model.tokenizer
    col = {f: j for j, f in enumerate(features)}
    need: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    out: dict[int, list[dict | None]] = {f: [] for f in features}
    for f in features:
        for slot, idx in enumerate(stats["top_idx"][f, :n_contexts].tolist()):
            if idx < 0:
                break
            row, pos = locate(stats, idx)
            need[row].append((f, slot, max(0, pos - before), pos + after + 1))
            out[f].append(None)

    rows = sorted(need)
    feat_idx = torch.tensor(features, device=model.cfg.device)
    for i in range(0, len(rows), batch_rows):
        chunk = rows[i : i + batch_rows]
        tokens = stats["tokens"][chunk].long().to(model.cfg.device)
        _, cache = model.run_with_cache(tokens, names_filter=hook, stop_at_layer=layer + 1)
        x = cache[hook][:, 1:]
        z = sae.encode(x.reshape(-1, x.shape[-1]).float())[:, feat_idx]
        z = z.view(len(chunk), x.shape[1], len(features)).cpu()
        for r, row in enumerate(chunk):
            ids = stats["tokens"][row, 1:].tolist()
            for f, slot, lo, hi in need[row]:
                acts = z[r, lo:hi, col[f]].tolist()
                out[f][slot] = {
                    "tokens": [tok.decode([t]) for t in ids[lo:hi]],
                    "acts": [round(a, 3) for a in acts],
                }
    return {f: [c for c in ctxs if c is not None] for f, ctxs in out.items()}


def write_site_data(
    out_dir: Path,
    model: HookedTransformer,
    sae: Dictionary,
    stats: dict[str, Tensor],
    labels: dict[int, dict],
    features: list[int],
    *,
    hook: str,
    layer: int,
    summary: dict,
    matches: tuple[Tensor, Tensor] | None = None,
) -> None:
    tok = model.tokenizer
    feat_dir = out_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    contexts = context_records(model, sae, stats, features, hook=hook, layer=layer)
    n_tokens = int(stats["n_tokens"])
    bin_edges = torch.linspace(HIST_LO, HIST_HI, HIST_BINS + 1).tolist()

    index = []
    for f in features:
        lab = labels.get(f, {})
        density = int(stats["counts"][f]) / n_tokens
        rec = {
            "id": f,
            "label": lab.get("label"),
            "description": lab.get("description"),
            "label_source": lab.get("source"),
            "confidence": lab.get("confidence"),
            "density": density,
            "max_act": round(float(stats["top_vals"][f, 0]), 3),
            "logit_up": [
                [tok.decode([t]), round(v, 3)]
                for t, v in zip(
                    stats["logit_up"][f].tolist(), stats["logit_up_vals"][f].tolist(), strict=True
                )
            ],
            "logit_down": [
                [tok.decode([t]), round(v, 3)]
                for t, v in zip(
                    stats["logit_down"][f].tolist(),
                    stats["logit_down_vals"][f].tolist(),
                    strict=True,
                )
            ],
            "hist": stats["hist"][f].tolist(),
            "hist_edges": [round(e, 3) for e in bin_edges],
            "contexts": contexts[f],
        }
        if matches is not None:
            cos, idx = matches
            rec["match"] = {
                "cos": round(float(cos[f]), 3),
                "jb_id": int(idx[f]),
                "url": NEURONPEDIA.format(int(idx[f])),
            }
        (feat_dir / f"{f}.json").write_text(json.dumps(rec, separators=(",", ":")))
        index.append(
            {
                k: rec.get(k)
                for k in ("id", "label", "confidence", "density", "max_act", "label_source")
            }
            | {"top_token": rec["logit_up"][0][0], "match_cos": rec.get("match", {}).get("cos")}
        )

    (out_dir / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
