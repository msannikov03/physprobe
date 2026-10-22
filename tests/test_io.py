"""Minimal io test: write a fake dataset, validate, read back, features round trip, splits, validate checks.
Run: python tests/test_io.py   (or pytest tests/)"""
import contextlib
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

G = 9.81
REST_Z = 0.05


@contextlib.contextmanager
def tmp_root():
    """Temporary PHYSPROBE_ROOT; the previous value (or its absence) is restored on exit."""
    old = os.environ.get("PHYSPROBE_ROOT")
    try:
        with tempfile.TemporaryDirectory() as root:
            os.environ["PHYSPROBE_ROOT"] = root
            yield root
    finally:
        if old is None:
            os.environ.pop("PHYSPROBE_ROOT", None)
        else:
            os.environ["PHYSPROBE_ROOT"] = old


def make_dataset(root, name="push", n=3, break_frame0=False, n_frame0=1):
    """Fake push dataset that passes io.validate: frame 0 = grey with a red square (shared unless
    n_frame0 > 1, then clip k gets variant k % n_frame0), random later frames, fn = m g, x(t) increasing.
    Sets PHYSPROBE_ROOT = root (use inside tmp_root() so it is restored)."""
    os.environ["PHYSPROBE_ROOT"] = root
    from physprobe import io
    ds = io.dataset_dir(name)
    rows = []
    for k in range(n):
        seed = k
        rng = np.random.default_rng(seed)
        m = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
        mu = float(rng.uniform(0.2, 0.8))
        v0 = 1.0 / m
        frame0 = np.full((io.H, io.W, 3), 128, np.uint8)
        frame0[120:137, 20:37] = (220, 30, 30)
        frame0[0, 0, 0] = k % n_frame0
        if break_frame0 and k == n - 1:
            frame0[1, 1, 1] = 0
        clip = np.empty(io.CLIP_SHAPE, np.uint8)
        clip[0] = frame0
        clip[1:] = rng.integers(0, 255, (io.T - 1, io.H, io.W, 3), dtype=np.uint8)
        t = np.arange(io.T) / io.FPS
        v = np.maximum(v0 - mu * G * t, 0.0)
        x = np.cumsum(np.r_[0.0, v[:-1]]) / io.FPS
        contact = np.stack([np.full(io.T, m * G), np.where(v > 0, mu * m * G, 0.0), v], 1).astype(np.float32)
        state = np.zeros(io.STATE_SHAPE, np.float32)
        state[:, 0], state[:, 2], state[:, 3] = x, REST_Z, v
        io.write_clip(ds, f"{k:06d}", clip, contact, state)
        rows.append(dict(id=f"{k:06d}", scenario="push", m=m, mu=mu, J=1.0, v0=v0, x0_m=0.0, seed=seed,
                         split=io.canonical_split(seed), in_frame=True, disp_m=float(x[-1]), moved_px=10,
                         frames_moving=20, t_stop_s=0.4, max_tilt_deg=0.0, max_yaw_deg=0.0, z_pen_max_m=0.0,
                         ok=True))
    io.write_meta(ds, rows)
    io.write_dataset_json(ds, dict(name=name, scenario="push", n=n, dt=1 / 1280, substeps=20,
                                   ranges=dict(m=[0.5, 2.0, "loguniform"], mu=[0.2, 0.8, "uniform"]), J=1.0,
                                   v0_fixed=None, cube_half=0.05, floor_friction=0.01, base_seed=0, g=G,
                                   rest_z_m=REST_Z, split_mode="canonical",
                                   camera=dict(pos=[0, 0, 1], fovy=45, m_per_px_floor=0.004, origin_px=[128, 128]),
                                   mujoco_version="test"))
    return io


def test_roundtrip():
    with tmp_root() as root:
        io = make_dataset(root)
        rep = io.validate("push", verbose=False)
        assert rep["problems"] == [], rep["problems"]
        assert rep["n"] == 3 and rep["ok_frac"] == 1.0
        meta = io.load_meta("push")
        assert [r["id"] for r in meta] == ["000000", "000001", "000002"]
        c = io.load_clip("push", "000001")
        assert c.shape == io.CLIP_SHAPE and c.dtype == np.uint8
        assert io.load_contact("push", "000001").shape == io.CONTACT_SHAPE
        assert io.load_state("push", "000001").shape == io.STATE_SHAPE
        ids = [r["id"] for r in meta]
        X = np.random.default_rng(0).standard_normal((3, 8)).astype(np.float32)
        p = io.save_features("push", "dummy_rep", "clip_mean", ids, X, dict(hf_id="none", layer="last"))
        assert os.path.exists(p)
        ids2, X2, info = io.load_features("push", "dummy_rep", "clip_mean")
        assert ids2 == ids and np.allclose(X, X2) and info["rep"] == "dummy_rep" and info["mode"] == "clip_mean"
        assert info["git_commit"] == io.git_commit() != ""
        s0 = io.split_ids(meta, 0)
        assert s0 == {"train": ids}
        meta_big = [dict(id=f"{k:06d}", seed=k, split=io.canonical_split(k)) for k in range(100)]
        s = io.split_ids(meta_big, 0)
        assert (len(s["train"]), len(s["val"]), len(s["test"])) == (70, 15, 15)
        s1 = io.split_ids(meta_big, 1)
        assert (len(s1["train"]), len(s1["val"]), len(s1["test"])) == (70, 15, 15)
        assert s1 != s and sorted(sum(s1.values(), [])) == sorted(sum(s.values(), []))
        assert io.split_ids(meta_big, 1) == s1


def test_frame0_check():
    with tmp_root() as root:
        io = make_dataset(root, break_frame0=True)
        rep = io.validate("push", verbose=False)
        assert any("frame 0 differs" in p for p in rep["problems"]), rep


def test_env_restored():
    before = os.environ.get("PHYSPROBE_ROOT")
    with tmp_root() as root:
        make_dataset(root, n=1)
        assert os.environ["PHYSPROBE_ROOT"] == root
    assert os.environ.get("PHYSPROBE_ROOT") == before and not os.path.exists(root)


def test_validate_content_checks():
    """Each corruption is reported: ok fraction, fn <= 0, x decreasing, constant frame 0, unchanged last frame,
    non-reproducible m, v0 != J/m, wrong canonical split, state row 0."""
    with tmp_root() as root:
        io = make_dataset(root, n=3)
        ds = io.dataset_dir("push")
        meta = io.load_meta("push")
        meta[0]["ok"] = False
        meta[1]["m"] = meta[1]["m"] * (1 + 1e-6)
        meta[1]["v0"] = 1.0 / meta[1]["m"]
        meta[2]["split"] = "val"
        meta[2]["v0"] = meta[2]["v0"] + 0.1
        io.write_meta(ds, meta)
        c = np.load(os.path.join(ds, "contact_000001.npy"))
        c[5, 0] = 0.0
        np.save(os.path.join(ds, "contact_000001.npy"), c)
        st = np.load(os.path.join(ds, "state_000002.npy"))
        st[40, 0] = st[39, 0] - 1e-3
        np.save(os.path.join(ds, "state_000002.npy"), st)
        clip = np.load(os.path.join(ds, "clip_000000.npy"))
        clip[0] = 7
        clip[-1] = 7
        np.save(os.path.join(ds, "clip_000000.npy"), clip)
        probs = io.validate("push", verbose=False)["problems"]
        for needle in ("ok_frac", "contact fn <= 0", "state x decreases", "frame 0 is constant",
                       "last frame equals frame 0", "do not reproduce", "v0=", "!= canonical", "state row 0"):
            assert any(needle in p for p in probs), (needle, probs)
        assert io.main(["validate", "push"]) == 1


def test_validate_split_fractions():
    """Meta-only dataset with n = 20: every row labelled train -> per-row and fraction problems."""
    with tmp_root():
        from physprobe import io
        ds = io.dataset_dir("slide")
        rows = [dict(id=f"{k:06d}", scenario="slide", seed=k, split="train", ok=True, in_frame=True) for k in range(20)]
        io.write_meta(ds, rows)
        io.write_dataset_json(ds, dict(name="slide", scenario="slide", n=20, ranges={},
                                       camera=dict(pos=[0, 0, 1], fovy=45, m_per_px_floor=0.004)))
        probs = io.validate("slide", verbose=False)["problems"]
        assert any("split val missing" in p for p in probs), probs
        assert any("!= canonical" in p for p in probs), probs
        assert any("camera lacks ['origin_px']" in p for p in probs), probs


if __name__ == "__main__":
    for _name, _fn in [(k, v) for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]:
        _fn()
    print("tests/test_io.py: OK")
