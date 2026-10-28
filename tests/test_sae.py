from pathlib import Path

import torch

from gpt2_sae.sae import SAEConfig, TopKSAE

SMALL = SAEConfig(d_in=16, d_sae=64, k=4, k_aux=8, dead_after_tokens=100)


def make_sae(cfg: SAEConfig = SMALL) -> TopKSAE:
    torch.manual_seed(0)
    return TopKSAE(cfg)


def test_shapes_and_exactly_k_active() -> None:
    sae = make_sae()
    x = torch.randn(32, SMALL.d_in)
    out = sae(x)
    assert out.x_hat.shape == x.shape
    assert out.latents.shape == (32, SMALL.d_sae)
    assert ((out.latents > 0).sum(-1) <= SMALL.k).all()
    assert (out.latents >= 0).all()


def test_decoder_rows_are_unit_norm() -> None:
    sae = make_sae()
    norms = sae.W_dec.norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms))


def test_parallel_grads_removed() -> None:
    sae = make_sae()
    sae(torch.randn(8, SMALL.d_in)).loss.backward()
    sae.remove_parallel_grads()
    dots = (sae.W_dec.grad * sae.W_dec).sum(-1)
    torch.testing.assert_close(dots, torch.zeros_like(dots), atol=1e-6, rtol=0)


def test_loss_decreases_on_sparse_toy_data() -> None:
    # Data built from 4-sparse combinations of 64 random directions: the SAE should fit it.
    torch.manual_seed(1)
    dirs = torch.nn.functional.normalize(torch.randn(SMALL.d_sae, SMALL.d_in), dim=-1)
    codes = torch.zeros(4096, SMALL.d_sae)
    idx = torch.randint(0, SMALL.d_sae, (4096, SMALL.k))
    codes.scatter_(1, idx, torch.rand(4096, SMALL.k) + 0.5)
    data = codes @ dirs

    sae = make_sae()
    sae.init_b_pre(data)
    opt = torch.optim.Adam(sae.parameters(), lr=3e-3)
    first = None
    for step in range(300):
        batch = data[torch.randint(0, len(data), (256,))]
        out = sae(batch)
        opt.zero_grad()
        out.loss.backward()
        sae.remove_parallel_grads()
        opt.step()
        sae.normalize_decoder()
        if step == 0:
            first = out.mse.item()
    assert first is not None
    assert out.mse.item() < 0.5 * first


def test_dead_latents_tracked_and_aux_loss_active() -> None:
    sae = make_sae()
    sae.since_fired.fill_(SMALL.dead_after_tokens)
    assert sae.dead_fraction() == 1.0
    out = sae(torch.randn(8, SMALL.d_in))
    assert out.aux.item() > 0
    # Latents that fired in this batch are no longer dead.
    assert sae.dead_fraction() < 1.0


def test_save_load_roundtrip(tmp_path: Path) -> None:
    sae = make_sae()
    sae.input_scale.fill_(3.0)
    path = tmp_path / "sae.pt"
    sae.save(path)
    loaded = TopKSAE.load(path)
    x = torch.randn(4, SMALL.d_in)
    sae.eval()
    loaded.eval()
    torch.testing.assert_close(sae.encode(x), loaded.encode(x))
    assert loaded.input_scale.item() == 3.0
