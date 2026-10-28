# gpt2-sae

A sparse autoencoder (SAE) trained from scratch on GPT-2 small, plus a browser for the features it learned.

<!-- RESULTS -->

## What an SAE does here

GPT-2 small passes a 768-dimensional vector (the residual stream) from layer to layer. Single dimensions of that vector rarely mean one thing. An SAE learns a larger set of directions, 24,576 here, and rewrites each vector as a sum of a few of them. Each direction is a "feature". Many features turn out to fire on one recognizable thing, such as a token, a topic or a piece of syntax.

This repo trains a TopK SAE ([Gao et al., 2024](https://arxiv.org/abs/2406.04093)) on the residual stream entering layer 6 (`blocks.6.hook_resid_pre`). TopK keeps the k = 32 largest latent activations for each token and sets the rest to zero. That fixes the sparsity directly, so there is no L1 penalty to tune.

## Method

- **Data.** OpenWebText, streamed from Hugging Face. Documents are packed into 128-token rows that start with a BOS token. The BOS position is dropped, because its norm is much larger than any other position's. Every 100th document is held out for evaluation.
- **Activations.** They are computed on the fly with TransformerLens and served from a shuffled buffer, so nothing is cached to disk.
- **SAE.** It has a pre-encoder bias, starts with the encoder tied to the decoder, and keeps decoder rows at unit norm. Inputs are scaled so their mean norm is √768. An auxiliary loss (aux-k) gives a gradient to latents that have not fired in the last 10M tokens.
- **Training.** Adam with lr 2e-4, batch size 4096 tokens, linear warmup, and linear decay over the last 20% of steps. The full run was on one A40 GPU on RunPod. Development ran locally on an M1.
- **Evaluation.** The SAE is measured on held-out text. The main number is *CE loss recovered*: splice the SAE's reconstruction into GPT-2 in place of the real activation, and measure how much of the gap between the clean loss and the zero-ablated loss it closes.
- **Comparison.** The same evaluation runs on the public `gpt2-small-res-jb` SAE for the same hook, loaded with [SAELens](https://github.com/jbloomAus/SAELens). Features are matched by the cosine similarity of their decoder directions.
- **Labels.** A random sample of live features gets labels from an OpenAI model (`gpt-5.6-luna`), which reads each feature's top-activating snippets and the output tokens it promotes. Some labels were checked by hand.
- **Steering.** Adding a feature's decoder direction to the residual stream during generation shows what the feature does, as well as what it detects.

## Repo layout

```
src/gpt2_sae/
  sae.py           TopK SAE
  activations.py   streaming data -> GPT-2 -> shuffled activation batches
  train.py         training loop, checkpoints, metrics.jsonl
  evaluate.py      FVU, L0, dead latents, CE loss recovered
  compare.py       gpt2-small-res-jb adapter and decoder matching
  features.py      top contexts, histograms, logit lens per latent
  autointerp.py    LLM labels (OpenAI)
  steer.py         generation with a feature added
  export.py        JSON for the browser
site/              static feature browser (GitHub Pages)
configs/           run configs
scripts/           RunPod helpers
results/           metrics and plots committed for this README
```

## Reproduce

```bash
uv sync --extra compare --extra label
uv run pytest
uv run gpt2-sae train --config configs/smoke.yaml      # small local check
```

The full run needs a CUDA GPU:

```bash
scripts/pod_sync.sh <ssh-host>                          # copy the repo to a RunPod pod and install
# on the pod, inside tmux:
bash scripts/pod_run.sh configs/full.yaml               # train, evaluate + compare, collect features
# back on the laptop:
scripts/pod_fetch.sh <ssh-host>
uv run gpt2-sae label  --config configs/full.yaml       # needs OPENAI_API_KEY
uv run gpt2-sae steer  --config configs/full.yaml --features <ids>
uv run gpt2-sae export --config configs/full.yaml
python -m http.server -d site                           # view the browser locally
```

## License

MIT
