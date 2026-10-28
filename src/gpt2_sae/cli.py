"""Command line entry point: `gpt2-sae <command> --config configs/<run>.yaml`.

Typical order: train -> evaluate -> features -> label -> steer -> export.
Large files go to artifacts/<name>/. Small results for the README go to results/<name>/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gpt2_sae.config import RunConfig

HAND_LABELS = Path("configs/hand_labels.yaml")


def _setup(cfg: RunConfig):
    from gpt2_sae.activations import load_model
    from gpt2_sae.evaluate import OurSAE
    from gpt2_sae.sae import TopKSAE

    device = cfg.resolve_device()
    model = load_model(device)
    sae = OurSAE(TopKSAE.load(cfg.out_dir / "sae.pt", device))
    return device, model, sae


def results_dir(cfg: RunConfig) -> Path:
    path = Path("results") / cfg.name
    path.mkdir(parents=True, exist_ok=True)
    return path


def cmd_train(cfg: RunConfig, args: argparse.Namespace) -> None:
    from gpt2_sae.plots import training_curves
    from gpt2_sae.train import train

    train(cfg, resume=args.resume)
    training_curves(cfg.out_dir / "metrics.jsonl", results_dir(cfg) / "training.png")


def cmd_evaluate(cfg: RunConfig, args: argparse.Namespace) -> None:
    from gpt2_sae.evaluate import evaluate

    device, model, sae = _setup(cfg)
    report = {"ours": evaluate(model, sae, hook=cfg.hook, n_batches=cfg.eval_batches)}
    print("ours", json.dumps(report["ours"], indent=1))
    if args.compare:
        from gpt2_sae.compare import JbSAE, best_matches
        from gpt2_sae.plots import match_histogram

        jb = JbSAE(cfg.hook, device)
        report["jb"] = evaluate(model, jb, hook=cfg.hook, n_batches=cfg.eval_batches)
        print("jb", json.dumps(report["jb"], indent=1))
        cos, idx = best_matches(sae.W_dec, jb.W_dec)
        torch.save({"cos": cos, "idx": idx}, cfg.out_dir / "matches.pt")
        report["match"] = {
            "median_cos": cos.median().item(),
            "frac_above_0.7": (cos > 0.7).float().mean().item(),
            "frac_above_0.9": (cos > 0.9).float().mean().item(),
        }
        print("match", report["match"])
        match_histogram(cos, results_dir(cfg) / "match_hist.png")
    (results_dir(cfg) / "eval.json").write_text(json.dumps(report, indent=1))


def cmd_features(cfg: RunConfig, args: argparse.Namespace) -> None:
    from gpt2_sae.features import collect, save

    _, model, sae = _setup(cfg)
    stats = collect(model, sae, hook=cfg.hook, layer=cfg.layer, n_tokens=cfg.feature_tokens)
    save(stats, cfg.out_dir / "stats.pt")
    live = int((stats["counts"] > 0).sum())
    print(f"{live}/{sae.d_sae} latents fired on {int(stats['n_tokens']):,} tokens")


def cmd_label(cfg: RunConfig, args: argparse.Namespace) -> None:
    from transformer_lens import HookedTransformer

    from gpt2_sae.autointerp import label_features, load_stats, sample_features

    stats = load_stats(cfg.out_dir / "stats.pt")
    tokenizer = HookedTransformer.from_pretrained("gpt2", device="cpu").tokenizer
    features = sample_features(stats, cfg.n_label, seed=cfg.seed)
    labels = label_features(stats, tokenizer, features, cfg.out_dir / "labels.jsonl")
    print(f"{len(labels)} features labeled")


def cmd_steer(cfg: RunConfig, args: argparse.Namespace) -> None:
    from gpt2_sae.autointerp import load_labels, load_stats
    from gpt2_sae.steer import steer_feature, steering_vector

    _, model, sae = _setup(cfg)
    stats = load_stats(cfg.out_dir / "stats.pt")
    labels = load_labels(cfg.out_dir / "labels.jsonl", HAND_LABELS)
    scale = float(sae.sae.input_scale)
    out = []
    for f in args.features:
        max_act = float(stats["top_vals"][f, 0])
        vec = steering_vector(sae.W_dec, f, max_act, scale)
        rows = steer_feature(model, vec, hook=cfg.hook, coefs=tuple(args.coefs))
        out.append({"feature": f, "label": labels.get(f, {}).get("label"), "samples": rows})
        for r in rows:
            print(f"[{f} x{r['coef']}] {r['prompt']}|{r['text']!r}")
    (results_dir(cfg) / "steering.json").write_text(json.dumps(out, indent=1))


def cmd_export(cfg: RunConfig, args: argparse.Namespace) -> None:
    from gpt2_sae.autointerp import load_labels, load_stats
    from gpt2_sae.export import pick_features, write_site_data

    _, model, sae = _setup(cfg)
    stats = load_stats(cfg.out_dir / "stats.pt")
    labels = load_labels(cfg.out_dir / "labels.jsonl", HAND_LABELS)
    features = pick_features(stats, labels, cfg.n_export_extra, seed=cfg.seed)
    res = results_dir(cfg)
    lines = (cfg.out_dir / "metrics.jsonl").read_text().splitlines()
    metrics = [json.loads(line) for line in lines]
    stride = max(1, len(metrics) // 300)
    summary = {
        "run": cfg.name,
        "hook": cfg.hook,
        "d_sae": cfg.sae.d_sae,
        "k": cfg.sae.k,
        "total_tokens": cfg.total_tokens,
        "eval": json.loads((res / "eval.json").read_text())
        if (res / "eval.json").exists()
        else None,
        "training": [{k: m[k] for k in ("tokens", "mse", "dead_frac")} for m in metrics[::stride]],
        "steering": json.loads((res / "steering.json").read_text())
        if (res / "steering.json").exists()
        else [],
        "n_labeled": len(labels),
    }
    matches_path = cfg.out_dir / "matches.pt"
    matches = None
    if matches_path.exists():
        blob = torch.load(matches_path, weights_only=True)
        matches = (blob["cos"], blob["idx"])
        summary["match_hist"] = torch.histc(blob["cos"], bins=40, min=0, max=1).tolist()
    write_site_data(
        Path(args.out),
        model,
        sae,
        stats,
        labels,
        features,
        hook=cfg.hook,
        layer=cfg.layer,
        summary=summary,
        matches=matches,
    )
    print(f"exported {len(features)} features to {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="gpt2-sae")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help)
        p.add_argument("--config", required=True)
        return p

    add("train", "train an SAE").add_argument("--resume", action="store_true")
    add("evaluate", "reconstruction and CE metrics on held-out text").add_argument(
        "--compare", action="store_true", help="also evaluate gpt2-small-res-jb and match features"
    )
    add("features", "collect top contexts and stats per latent")
    add("label", "label a random sample of features with Claude Haiku")
    p = add("steer", "generate text with a feature added to the residual stream")
    p.add_argument("--features", type=int, nargs="+", required=True)
    p.add_argument("--coefs", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    add("export", "write JSON for the feature browser").add_argument("--out", default="site/data")

    args = parser.parse_args()
    cfg = RunConfig.from_yaml(args.config)
    commands = {
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "features": cmd_features,
        "label": cmd_label,
        "steer": cmd_steer,
        "export": cmd_export,
    }
    commands[args.command](cfg, args)


if __name__ == "__main__":
    main()
