"""Feature extraction tests (FEAT_PY: torch + transformers).

CPU, tiny random configs, no downloads:  python tests/test_feats.py
GPU, real weights from the HF cache:      PHYSPROBE_GPU_TESTS=1 python tests/test_feats.py
The two test_gpu_* tests need CUDA and the cached weights; they skip otherwise.
pytest also works; on compute run it as PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider tests/test_feats.py,
otherwise pytest writes assertion-rewrite .pyc files into site-packages on the full home partition.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

try:
    import torch
    import torch.nn.functional as F
    import transformers  # noqa: F401
except ImportError:
    import pytest
    pytest.skip("needs FEAT_PY (torch + transformers)", allow_module_level=True)

from transformers import Dinov2Config, VJEPA2Config  # noqa: E402

from physprobe.feats import encoders, extract  # noqa: E402
from test_io import make_dataset, tmp_root  # noqa: E402

CONTRACT_INFO_KEYS = {"rep", "hf_id", "revision", "mode", "layer", "pooling", "n_frames", "input_res", "sec_per_clip",
                      "git_commit", "code_hash"}
QUIET = lambda s: None  # noqa: E731


class Skip(Exception):
    pass


def _skip(msg):
    if "PYTEST_CURRENT_TEST" in os.environ:
        import pytest
        pytest.skip(msg)
    raise Skip(msg)


def tiny_vjepa(modes=None, dtype=None):
    cfg = VJEPA2Config(patch_size=64, crop_size=256, frames_per_clip=64, tubelet_size=2, hidden_size=32,
                       num_attention_heads=2, num_hidden_layers=24, mlp_ratio=1.0, pred_hidden_size=32,
                       pred_num_attention_heads=2, pred_num_hidden_layers=1, num_pooler_layers=1)
    return encoders.VJEPA2("tiny_vjepa", device="cpu", dtype=dtype or torch.float32, config=cfg, modes=modes)


def tiny_dino():
    cfg = Dinov2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, mlp_ratio=1, image_size=224,
                       patch_size=56)
    return encoders.DINOv2("tiny_dino", device="cpu", dtype=torch.float32, config=cfg)


def rand_clips(b, t, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (b, t, 256, 256, 3), dtype=torch.uint8, generator=g)


def test_vjepa_pool_order():
    b, n_t, s, d = 2, 4, 9, 3
    tok = torch.arange(n_t, dtype=torch.float32).view(1, n_t, 1, 1).expand(b, n_t, s, d).reshape(b, n_t * s, d)
    mean, seq = encoders.vjepa_pool(tok, n_t)
    assert seq.shape == (b, n_t, d) and mean.shape == (b, d)
    assert torch.equal(seq[1, :, 2], torch.arange(n_t, dtype=torch.float32))
    assert torch.allclose(mean, seq.mean(1)) and torch.allclose(mean, tok.mean(1))


def test_dino_frame_feat():
    h = torch.randn(5, 1 + 4, 3)
    f = encoders.dino_frame_feat(h)
    assert f.shape == (5, 6)
    assert torch.equal(f[:, :3], h[:, 0]) and torch.allclose(f[:, 3:], h[:, 1:].mean(1))


def test_vjepa_tiny_encode():
    enc = tiny_vjepa()
    clips = rand_clips(2, 8)
    out = enc.encode(clips, enc.modes)
    assert "static" not in out
    n_t, d = 4, 32
    for m in enc.modes[1:]:
        if m.startswith("clip_seq"):
            assert out[m].shape == (2, n_t * d) and out[m].dtype == np.float16, (m, out[m].shape, out[m].dtype)
        else:
            assert out[m].shape == (2, d) and out[m].dtype == np.float32, (m, out[m].shape, out[m].dtype)
    for sfx in ("", "_L08", "_L16"):
        seq = out["clip_seq" + sfx].astype(np.float32).reshape(2, n_t, d)
        assert np.allclose(seq.mean(1), out["clip_mean" + sfx], atol=2e-3, rtol=2e-3), sfx
    assert not np.allclose(out["clip_mean_L08"], out["clip_mean"])
    enc.static_frames = 8
    f0 = clips[0, 0]
    st = enc.encode_static(torch.stack([f0, f0, clips[1, 0]]))
    assert st.shape == (3, d) and st.dtype == np.float32
    assert np.array_equal(st[0], st[1]) and not np.allclose(st[0], st[2])
    static_clip = f0[None, None].expand(1, 8, -1, -1, -1).contiguous()
    assert np.allclose(enc.encode(static_clip, ["clip_mean"])["clip_mean"][0], st[0], atol=1e-5)
    for m in enc.modes:
        info = enc.mode_info(m)
        assert {"hf_id", "revision", "layer", "pooling", "n_frames", "input_res"} <= set(info), m
    assert enc.mode_info("clip_seq_L08")["layer"] == 8 and enc.mode_info("clip_mean")["layer"] == 24


def test_dino_tiny_encode():
    enc = tiny_dino()
    clips = rand_clips(2, 64, seed=1)
    out = enc.encode(clips, ["last", "clip_mean", "clip_seq"])
    d = 64
    assert out["last"].shape == (2, d) and out["clip_mean"].shape == (2, d) and out["clip_seq"].shape == (2, 16 * d)
    assert out["clip_seq"].dtype == np.float16 and out["clip_mean"].dtype == np.float32
    f = enc._frames(clips.reshape(-1, 256, 256, 3)).reshape(2, 64, d).numpy()
    assert np.allclose(out["last"], f[:, 63], atol=1e-5)
    assert np.allclose(out["clip_mean"], f.mean(1), atol=1e-5)
    assert np.allclose(out["clip_seq"].astype(np.float32), f[:, 0:64:4].reshape(2, -1), atol=2e-3, rtol=2e-3)
    assert np.allclose(enc.encode(clips, ["last"])["last"], out["last"], atol=1e-5)
    assert np.allclose(enc.encode_static(clips[:, 0]), f[:, 0], atol=1e-5)
    assert enc.mode_info("clip_seq")["frame_idx"] == list(range(0, 64, 4))


def test_dino_frame_batch_invariant():
    """Same frame -> same feature whatever the pass: static, last and clip_seq slot 0 of a clip whose frame 63 is a
    copy of frame 0, in passes with different mode sets and batch sizes (every forward sees 64 frames)."""
    enc = tiny_dino()
    calls = []
    orig = enc.model.forward

    def spy(*a, **kw):
        calls.append(kw["pixel_values"].shape[0])
        return orig(*a, **kw)

    enc.model.forward = spy
    clips = rand_clips(3, 64, seed=4)
    clips[:, 63] = clips[:, 0]
    d = 64
    st = enc.encode_static(clips[:, 0])
    a = enc.encode(clips, ["last", "clip_seq"])
    b = enc.encode(clips[:1], ["last", "clip_mean", "clip_seq"])
    assert set(calls) == {enc.frame_chunk}, calls
    for x in (a["last"], a["clip_seq"][:, :d].astype(np.float32)):
        assert np.allclose(x, st, atol=2e-3), np.abs(x - st).max()
    assert np.allclose(b["last"][0], a["last"][0], atol=1e-5)
    assert enc.mode_info("last")["frame_batch"].startswith("64 frames")


def test_static_dedupe_multi_frame0():
    """5 clips with 3 distinct frame 0s: static encodes 3 frames and maps each clip to its own frame 0."""
    with tmp_root() as root:
        io = make_dataset(root, n=5, n_frame0=3)
        ids = [r["id"] for r in io.load_meta("push")]
        enc = tiny_dino()
        assert extract.run("push", enc, modes=["static"], batch=2, log=QUIET) == ["static"]
        got, X, info = io.load_features("push", enc.name, "static")
        assert got == ids and info["n_unique_frame0"] == 3
        f0 = torch.from_numpy(np.stack([np.asarray(io.load_clip("push", i)[0]) for i in ids]))
        ref = enc.encode_static(f0)
        assert np.allclose(X, ref, atol=1e-5)
        assert np.array_equal(X[0], X[3]) and not np.allclose(X[0], X[1])


def test_stale_features_reextracted():
    """A mode is re-extracted when the stored dtype or code_hash differs; --ignore-code-hash skips only the
    code_hash part; partial chunks with another tag are ignored on resume."""
    with tmp_root() as root:
        io = make_dataset(root, n=3)
        enc = tiny_vjepa(modes=("clip_mean", "clip_seq"))
        assert extract.run("push", enc, batch=2, log=QUIET) == ["clip_mean", "clip_seq"]
        assert extract.run("push", enc, batch=2, log=QUIET) == []
        real_hash = extract.code_hash
        try:
            extract.code_hash = lambda: "other"
            assert extract.run("push", enc, batch=2, log=QUIET, ignore_code_hash=True) == []
            log = []
            assert extract.run("push", enc, batch=2, log=log.append) == ["clip_mean", "clip_seq"]
            assert any("stale (code_hash" in s for s in log), log
            assert io.load_features("push", enc.name, "clip_mean")[2]["code_hash"] == "other"
        finally:
            extract.code_hash = real_hash
        enc16 = tiny_vjepa(modes=("clip_mean",), dtype=torch.bfloat16)
        enc16.name = enc.name
        assert extract.run("push", enc16, batch=2, log=QUIET) == ["clip_mean"]
        assert io.load_features("push", enc.name, "clip_mean")[2]["dtype"] == "bfloat16"
        # chunks left by a crashed run with another code_hash are not merged
        orig, calls = enc.encode, [0]

        def flaky(clips, modes):
            calls[0] += 1
            if calls[0] == 2:
                raise RuntimeError("simulated crash")
            return orig(clips, modes)

        enc.encode = flaky
        try:
            extract.code_hash = lambda: "old"
            try:
                extract.run("push", enc, batch=1, chunk=1, overwrite=True, log=QUIET)
                raise AssertionError("expected crash")
            except RuntimeError as e:
                assert "simulated crash" in str(e)
        finally:
            extract.code_hash = real_hash
        assert len(os.listdir(extract.partial_dir("push", enc.name))) == 1
        calls[0] = 10
        log = []
        extract.run("push", enc, batch=1, chunk=1, log=log.append)
        assert calls[0] == 10 + 3, calls
        assert any("ignored 1 partial chunk" in s for s in log), log


class _StubCosmos(torch.nn.Module):
    """Causal 8x8x8 stand-in for encoder.jit: slot 0 = 8x8 mean of frame 0, slot j = 8x8x8 mean of frames
    8j-7..8j; channel c = RGB channel c % 3."""

    def forward(self, x):
        first = F.avg_pool3d(x[:, :, :1], (1, 8, 8))
        rest = F.avg_pool3d(x[:, :, 1:], (8, 8, 8))
        z = torch.cat([first, rest], 2)
        return (z[:, [c % 3 for c in range(16)]],)


def test_cosmos_stub_pad_and_flatten():
    """Cosmos path: 64 frames + frame 63 repeated -> 9 slots; x/127.5 - 1; clip_seq (slot, channel);
    clip_grid4 (channel, slot, row, col); clip_mean = mean of slots."""
    enc = encoders.CosmosCV8("cosmos_stub", device="cpu", dtype=torch.float32, model=_StubCosmos())
    t = torch.arange(64).view(64, 1, 1, 1)
    r = (torch.arange(256) // 64).view(1, 256, 1, 1)
    q = (torch.arange(256) // 64).view(1, 1, 256, 1)
    ch = torch.arange(3).view(1, 1, 1, 3)
    clips = (3 * t + 16 * r + 4 * q + ch).to(torch.uint8)[None]
    out = enc.encode(clips, enc.modes)
    frames = list(range(64)) + [63]
    tbar = np.array([0.0] + [np.mean(frames[8 * j - 7:8 * j + 1]) for j in range(1, 9)])
    cmod = np.arange(16) % 3
    want_grid = (3 * tbar[None, :, None, None] + 16 * np.arange(4)[None, None, :, None]
                 + 4 * np.arange(4)[None, None, None, :] + cmod[:, None, None, None]) / 127.5 - 1
    want_seq = want_grid.mean((2, 3)).T
    assert out["clip_seq"].shape == (1, 9 * 16) and out["clip_grid4"].shape == (1, 16 * 9 * 4 * 4)
    assert np.allclose(out["clip_seq"][0].astype(np.float32).reshape(9, 16), want_seq, atol=2e-3)
    assert np.allclose(out["clip_grid4"][0].astype(np.float32).reshape(16, 9, 4, 4), want_grid, atol=2e-3)
    assert np.allclose(out["clip_mean"][0], want_seq.mean(0), atol=1e-5)
    for m in enc.modes:
        info = enc.mode_info(m)
        assert {"hf_id", "revision", "layer", "pooling", "n_frames", "input_res"} <= set(info), m
    assert enc.mode_info("clip_grid4")["seq_shape"] == [16, 9, 4, 4]
    assert enc.mode_info("clip_seq")["seq_shape"] == [9, 16]


def test_default_dtype():
    assert encoders.default_dtype("vjepa2_l_random") == torch.float32
    assert encoders.default_dtype("dinov2_b_random") == torch.float32
    for name in ("vjepa2_l", "dinov2_b", "cosmos_cv8"):
        assert encoders.default_dtype(name) == torch.bfloat16, name


def _check_files(io, ds, enc, ids):
    for m in enc.modes:
        got, X, info = io.load_features(ds, enc.name, m)
        assert got == ids, m
        assert CONTRACT_INFO_KEYS <= set(info), (m, CONTRACT_INFO_KEYS - set(info))
        assert info["rep"] == enc.name and info["mode"] == m and info["sec_per_clip"] > 0
        assert X.dtype == (np.float16 if m.startswith("clip_seq") else np.float32), (m, X.dtype)
        assert np.isfinite(X).all()
        if m == "static":
            assert info["n_unique_frame0"] == 1 and (X == X[0]).all()
    assert not os.path.exists(extract.partial_dir(ds, enc.name))


def test_extract_end_to_end():
    with tmp_root() as root:
        io = make_dataset(root, n=5)
        ids = [r["id"] for r in io.load_meta("push")]
        venc, denc = tiny_vjepa(), tiny_dino()
        for enc in (venc, denc):
            assert extract.run("push", enc, batch=2, chunk=2, log=QUIET) == list(enc.modes)
            _check_files(io, "push", enc, ids)
        _, X, _ = io.load_features("push", "tiny_vjepa", "clip_seq")
        assert X.shape == (5, 32 * 32)
        _, X, _ = io.load_features("push", "tiny_dino", "clip_seq")
        assert X.shape == (5, 16 * 64)
        assert not np.allclose(X[0], X[1])
        assert extract.run("push", venc, batch=2, log=QUIET) == []
        os.remove(io.paths.features_path("push", "tiny_vjepa", "clip_seq"))
        assert extract.run("push", venc, batch=2, log=QUIET) == ["clip_seq"]
        _, X2, _ = io.load_features("push", "tiny_vjepa", "clip_seq")
        assert X2.shape == (5, 32 * 32)
        assert extract.run("push", denc, limit=3, modes=["last"], log=QUIET) == ["last"]
        assert io.load_features("push", "tiny_dino", "last")[0] == ids[:3]


def test_extract_resume_after_crash():
    with tmp_root() as root:
        io = make_dataset(root, n=5)
        ids = [r["id"] for r in io.load_meta("push")]
        enc = tiny_vjepa(modes=("clip_mean", "clip_seq"))
        ref = {m: enc.encode(torch.from_numpy(np.stack([np.asarray(io.load_clip("push", i)) for i in ids])), [m])[m]
               for m in enc.modes}
        orig, calls = enc.encode, [0]

        def flaky(clips, modes):
            calls[0] += 1
            if calls[0] == 3:
                raise RuntimeError("simulated crash")
            return orig(clips, modes)

        enc.encode = flaky
        try:
            extract.run("push", enc, batch=1, chunk=1, log=QUIET)
            raise AssertionError("expected crash")
        except RuntimeError as e:
            assert "simulated crash" in str(e)
        assert len(os.listdir(extract.partial_dir("push", enc.name))) == 2
        calls[0] = 10
        extract.run("push", enc, batch=1, chunk=1, log=QUIET)
        assert calls[0] == 10 + 3, calls
        for m in enc.modes:
            got, X, info = io.load_features("push", enc.name, m)
            assert got == ids and np.allclose(X.astype(np.float32), ref[m].astype(np.float32), atol=2e-3), m


def _gpu_ready():
    if os.environ.get("PHYSPROBE_GPU_TESTS") != "1":
        _skip("set PHYSPROBE_GPU_TESTS=1")
    if not torch.cuda.is_available():
        _skip("no CUDA")


def test_gpu_vjepa2_real():
    _gpu_ready()
    enc = encoders.get("vjepa2_l")
    clips = rand_clips(1, 64, seed=2)
    with torch.inference_mode():
        out = enc.model(pixel_values_videos=enc._norm(clips), skip_predictor=True, output_hidden_states=True)
    assert out.last_hidden_state.shape == (1, 8192, 1024)
    assert len(out.hidden_states) == 25 and torch.equal(out.hidden_states[24], out.last_hidden_state)
    f = enc.encode(clips, enc.modes)
    for m in enc.modes[1:]:
        assert f[m].shape == ((1, 32 * 1024) if m.startswith("clip_seq") else (1, 1024)), (m, f[m].shape)
    assert enc.encode_static(clips[:, 0]).shape == (1, 1024)


def test_gpu_dinov2_real():
    _gpu_ready()
    enc = encoders.get("dinov2_b")
    clips = rand_clips(2, 64, seed=3)
    f = enc.encode(clips, ["last", "clip_mean", "clip_seq"])
    assert f["last"].shape == (2, 1536) and f["clip_mean"].shape == (2, 1536) and f["clip_seq"].shape == (2, 16 * 1536)
    assert enc.encode_static(clips[:, 0]).shape == (2, 1536)
    # bf16: the same frame must give the same feature in static, last and clip_seq slot 0 (fixed 64-frame batches)
    clips[:, 63] = clips[:, 0]
    st = enc.encode_static(clips[:, 0])
    g = enc.encode(clips, ["last", "clip_seq"])
    for name, x in (("last", g["last"]), ("clip_seq slot 0", g["clip_seq"][:, :1536].astype(np.float32))):
        d = float(np.abs(x - st).max())
        assert d < 1e-2, f"{name} vs static: max|d| = {d}"


if __name__ == "__main__":
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    names = [k for k in sorted(globals()) if k.startswith("test_")]
    failed = 0
    for k in names:
        try:
            globals()[k]()
            print(f"PASS {k}")
        except Skip as e:
            print(f"SKIP {k}: {e}")
        except Exception as e:
            failed += 1
            print(f"FAIL {k}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    print("tests/test_feats.py:", "OK" if not failed else f"{failed} failed")
    sys.exit(1 if failed else 0)
