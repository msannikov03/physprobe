"""B1: synthetic `dummy` dataset in CONTRACT v1 format (numpy + stdlib only).

A saturated red 17x17 px square slides on a gray background with analytic
Coulomb-friction kinematics: x(t) = v0 t - 1/2 mu g t^2 until t_stop = v0/(mu g).

Usage:
    python -m physprobe.tools.make_dummy [--n 200] [--dataset dummy] [--scenario push|slide]
        [--base-seed 0] [--J 1.0] [--v0 1.2] [--m-range 0.5 2] [--mu-range 0.2 0.8]
        [--no-fake-enc] [--workers N]

Conventions (resolve contract ambiguities; also written to dataset.json):
- World x = 0 is pixel column 0, x_px = x_m / m_per_px, m_per_px = 1.5/256
  (camera.origin_px = [0, 128]). Pixel column c is the integer coordinate c
  (the square's centre pixel is round(x_px); its centroid equals that column).
- The square's CENTRE starts at x = 30 px (x0_m = 30 * m_per_px); rows 120..136
  (centre row 128). Positions are rounded to the nearest pixel.
- Frame 0 is rendered at rest (x0) and is identical for all clips.
- If the square would leave the frame, its drawing is clamped at the right edge
  (centre column <= 247) and in_frame = ok = False. state/contact stay analytic.
- contact row k = mean over 20 substeps at midpoints t = k/64 + (j + 0.5)/1280.
- meta: scenario = "dummy", extra key dummy_kind = "push" | "slide".
- frames_moving = #frames k with t_k = k/64 < t_stop; disp_m = displacement at the
  last frame (t = 63/64); moved_px = pixel shift of the drawn square, last vs. frame 0.
- Fake encoder (features/<ds>/fake_enc): gray 32x32 (8x8 block mean, [0,1]) per frame,
  mean over 8 temporal bins of 8 frames -> 8192, fixed Gaussian projection (rng 0)
  to 1024, plus N(0, 0.05 * std) noise (rng 1). static.npz = same projection of
  frame 0 repeated over the 8 bins, without noise (identical rows, as for a real
  encoder on identical frame 0).
"""
import argparse
import datetime as _dt
import json
from pathlib import Path
import os
import time
from multiprocessing import Pool

import numpy as np

from physprobe import io

T, H, W = 64, 256, 256
FPS = 64
SUBSTEPS = 20
G = 9.81
M_PER_PX = 1.5 / 256
X0_PX = 30
ROW_C = 128
HALF_PX = 8  # 17 px square
BG = 128
COLOR = np.array([220, 30, 30], np.uint8)
CUBE_HALF = 0.05
FAKE_D = 1024


def sample_params(seed, m_range, mu_range):
    rng = np.random.default_rng(seed)
    m = float(np.exp(rng.uniform(np.log(m_range[0]), np.log(m_range[1]))))
    mu = float(rng.uniform(mu_range[0], mu_range[1]))
    return m, mu


def kinematics(t, v0, a):
    """Displacement and speed at times t (arrays) for decel a until stop."""
    t_stop = v0 / a
    tc = np.minimum(t, t_stop)
    d = v0 * tc - 0.5 * a * tc ** 2
    v = np.maximum(v0 - a * t, 0.0)
    return d, v


def render(cols):
    clip = np.full((T, H, W, 3), BG, np.uint8)
    for k, c in enumerate(cols):
        clip[k, ROW_C - HALF_PX:ROW_C + HALF_PX + 1, c - HALF_PX:c + HALF_PX + 1] = COLOR
    return clip


def gray32(frames):
    """(..., H, W, 3) uint8 -> (..., 32, 32) gray in [0, 1] via 8x8 block mean."""
    g = frames.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32) / 255.0
    return g.reshape(*g.shape[:-2], 32, 8, 32, 8).mean(axis=(-3, -1))


def gray_bins(clip):
    """(T,H,W,3) uint8 -> 8 temporal bins of 32x32 gray -> (8192,) float32."""
    return gray32(clip).reshape(8, 8, 32 * 32).mean(axis=1).reshape(-1)


def make_one(job):
    idx, cfg = job
    seed = cfg["base_seed"] + idx
    cid = f"{idx:06d}"
    m, mu = sample_params(seed, cfg["m_range"], cfg["mu_range"])
    if cfg["kind"] == "push":
        J, v0 = cfg["J"], cfg["J"] / m
    else:
        J, v0 = None, cfg["v0"]
    a = mu * G
    t_stop = v0 / a
    x0_m = X0_PX * M_PER_PX

    tf = np.arange(T) / FPS
    d, v = kinematics(tf, v0, a)
    x_px = np.rint((x0_m + d) / M_PER_PX).astype(int)
    x_px[0] = X0_PX  # frame 0 at rest
    in_frame = bool(x_px.max() + HALF_PX <= W - 1)
    cols = np.minimum(x_px, W - 1 - HALF_PX)

    state = np.zeros((T, 6), np.float32)
    state[:, 0] = x0_m + d
    state[:, 2] = CUBE_HALF
    state[:, 3] = v

    ts = (np.arange(T)[:, None] + (np.arange(SUBSTEPS)[None, :] + 0.5) / SUBSTEPS) / FPS
    _, vs = kinematics(ts, v0, a)
    contact = np.zeros((T, 3), np.float32)
    contact[:, 0] = m * G
    contact[:, 1] = (mu * m * G * (vs > 0)).mean(axis=1)
    contact[:, 2] = vs.mean(axis=1)

    ds_dir = Path(io.dataset_dir(cfg["dataset"]))
    clip_path = ds_dir / f"clip_{cid}.npy"
    if clip_path.exists():
        clip = np.load(clip_path, mmap_mode="r")
    else:
        clip = render(cols)
        io.write_clip(ds_dir, cid, clip, contact, state)
    feat = gray_bins(np.asarray(clip)) if cfg["fake_enc"] else None

    row = {
        "id": cid, "scenario": "dummy", "dummy_kind": cfg["kind"],
        "m": m, "mu": mu, "J": J, "v0": float(v0), "x0_m": x0_m,
        "seed": seed, "split": io.canonical_split(seed),
        "in_frame": in_frame, "disp_m": float(d[-1]),
        "moved_px": int(cols[-1] - cols[0]),
        "frames_moving": int((tf < t_stop).sum()), "t_stop_s": float(t_stop),
        "max_tilt_deg": 0.0, "max_yaw_deg": 0.0, "z_pen_max_m": 0.0,
        "ok": in_frame,
    }
    return row, feat


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--dataset", default=None)
    p.add_argument("--scenario", choices=["push", "slide"], default="push")
    p.add_argument("--base-seed", type=int, default=None)
    p.add_argument("--J", type=float, default=1.0)
    p.add_argument("--v0", type=float, default=1.2)
    p.add_argument("--m-range", type=float, nargs=2, default=[0.5, 2.0])
    p.add_argument("--mu-range", type=float, nargs=2, default=[0.2, 0.8])
    p.add_argument("--no-fake-enc", action="store_true")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = p.parse_args(argv)

    kind = args.scenario
    dataset = args.dataset or ("dummy" if kind == "push" else "dummy_slide")
    base_seed = args.base_seed if args.base_seed is not None else (0 if kind == "push" else 100000)
    cfg = dict(dataset=dataset, kind=kind, base_seed=base_seed, J=args.J, v0=args.v0,
               m_range=args.m_range, mu_range=args.mu_range, fake_enc=not args.no_fake_enc)
    ds_dir = Path(io.dataset_dir(dataset))
    ds_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    jobs = [(i, cfg) for i in range(args.n)]
    with Pool(args.workers) as pool:
        res = pool.map(make_one, jobs, chunksize=1)
    rows = [r for r, _ in res]
    io.write_meta(ds_dir, rows)

    commit = io.git_commit()
    io.write_dataset_json(ds_dir, {
        "name": dataset, "scenario": "dummy", "dummy_kind": kind, "n": args.n,
        "T": T, "H": H, "W": W, "fps": FPS, "dt": 1.0 / (FPS * SUBSTEPS), "substeps": SUBSTEPS,
        "contact_channels": ["fn_N", "ft_N", "slip_mps"],
        "state_channels": ["x", "y", "z", "vx", "vy", "vz"],
        "ranges": {"m": [args.m_range[0], args.m_range[1], "loguniform"],
                   "mu": [args.mu_range[0], args.mu_range[1], "uniform"]},
        "J": args.J if kind == "push" else None,
        "v0_fixed": args.v0 if kind == "slide" else None,
        "base_seed": base_seed,
        "cube_half": CUBE_HALF, "floor_friction": 0.01,
        "camera": {"pos": None, "fovy": None, "m_per_px_floor": M_PER_PX, "origin_px": [0, ROW_C],
                   "note": "x_px = x_m / m_per_px_floor (world x=0 at column 0); "
                           "square centre at x0=30 px, rows 120..136"},
        "square_px": 2 * HALF_PX + 1, "g": G,
        "mujoco_version": "dummy", "git_commit": commit,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
    })

    if cfg["fake_enc"]:
        ids = [r["id"] for r in rows]
        F = np.stack([f for _, f in res]).astype(np.float32)  # (n, 8192)
        P = np.random.default_rng(0).standard_normal((F.shape[1], FAKE_D)).astype(np.float32)
        P /= np.sqrt(F.shape[1])
        Z = F @ P
        Z += np.random.default_rng(1).standard_normal(Z.shape).astype(np.float32) * 0.05 * Z.std()
        f0 = gray32(np.asarray(io.load_clip(dataset, ids[0])[0])).reshape(-1)
        S = np.tile(np.tile(f0, 8) @ P, (len(ids), 1))
        base_info = dict(rep="fake_enc", hf_id=None, revision=None, layer=None, n_frames=T,
                         input_res=[32, 32], sec_per_clip=(time.time() - t0) / max(args.n, 1),
                         git_commit=commit,
                         note="fake: gray 32x32 x 8 temporal bins -> Gaussian proj (rng 0) to 1024")
        io.save_features(dataset, "fake_enc", "clip_mean", ids, Z,
                         dict(base_info, mode="clip_mean", pooling="8 temporal bins, proj + 5% noise"))
        io.save_features(dataset, "fake_enc", "static", ids, S,
                         dict(base_info, mode="static", pooling="frame 0 repeated x8, proj, no noise"))

    ok = np.mean([r["ok"] for r in rows])
    splits = {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}
    print(f"{dataset} ({kind}): n={len(rows)} ok={ok:.1%} splits={splits} "
          f"time={time.time() - t0:.1f}s -> {ds_dir}")


if __name__ == "__main__":
    main()
