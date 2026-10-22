"""Reference baseline feature blocks (PLAN §4.3 B2, CONTRACT v1 §4.1).

    python -m physprobe.baselines.build --dataset D [--workers N]
        [--only pixels,state,contact,track] [--noise-seed 0] [--noise-curve]

Reads data/<D>/ via physprobe.io, writes features/<D>/<rep>/<mode>.npz via
io.save_features (ids in meta order, as strings).

Blocks
  pixels/static     (1024,)  gray 32x32 of frame 0, [0,1]
  pixels/last       (1024,)  gray 32x32 of frame 63
  pixels/clip_seq   (8192,)  8 temporal bins (mean of 8 frames) x gray 32x32, float16
  state/oracle_x    (64,)    x(t) in px, rounded to 1 px            [ORACLE]
  state/oracle_xc   (64,)    x(t) in px, unquantised (float32)      [ORACLE]
                    (the linear x(t) read-out reference: a ceiling for mu, not for
                    log m; the 1-px oracle_x is neither: 32x32 block means keep
                    sub-pixel position)
  state/oracle_fit  (2,)     [log v0, a] fitted on unquantised x(t) [ORACLE]
  contact/raw       (192,)   flattened (64,3) [fn, ft, slip]        [ORACLE]
  contact/realistic (192,)   floored fn/max, ft/max + additive N(0, 0.05), slip>1e-3 flag
  track/clip        (128,)   per-frame centroid (x_px, y_px) of saturated cube
  contact/noise_{000,010,030,100} (192,)  only with --noise-curve: the noise-free
                    realistic representation (floored fn/max, ft/max, slip flag)
                    + additive N(0, p) on every channel and row, p = 0, 0.1, 0.3, 1.0

Realistic contact. After the cube stops, MuJoCo leaves a friction residual of
~1e-8 N proportional to m. Dividing by the per-clip max (= mu*m*g) would turn it
into a 1/mu code that train z-scoring blows up to unit variance (R^2 0.95 / 0.86 on
push from a solver residual). So every |fn|, |ft| below CONTACT_FLOOR_REL * max is
set to 0 BEFORE the per-clip normalisation, and the noise is ADDITIVE (a
multiplicative N(1, s) keeps tiny values tiny). What survives is the stop time
(ft step, slip flag): t_stop = J/(m mu g), worth about R^2 0.55 (log m) / 0.48 (mu).

Noise curve. The noise is relative to the normalised [0, 1] scale, so it cannot
leak m (no absolute force scale is left); raw slip speed |v_xy| = v0 - mu g t is an
exact kinematic oracle and is never included. The same rng seed is used for every
level (same draw, scaled), so the curve is monotone in p by construction.

Only numpy/stdlib here (plus multiprocessing); no torch, no mujoco.
"""
import argparse
import json
from pathlib import Path
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np

from physprobe import io

G = 9.81
T = 64
GRAY_W = np.array([0.299, 0.587, 0.114], dtype=np.float32)
SLIP_THR = 1e-3            # m/s
REALISTIC_NOISE = 0.05     # additive, on the per-clip normalised [0, 1] scale
CONTACT_FLOOR_REL = 1e-3   # |fn|, |ft| < 1e-3 * clip max -> 0 (solver residual after the stop)
NOISE_CURVE = (0.0, 0.10, 0.30, 1.00)
ALL_GROUPS = ("pixels", "state", "contact", "track")
FLOAT16_BLOCKS = {"pixels/clip_seq"}   # CONTRACT v1: X float16 for clip_seq
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


# ----------------------------------------------------------------------------- helpers

def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(io.root()),
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "nogit"
    except Exception:
        return "nogit"


def load_dataset_json(name):
    with open(Path(io.dataset_dir(name)) / "dataset.json") as f:
        return json.load(f)


def gray32(frames):
    """(..., 256, 256, 3) uint8 -> (..., 32, 32) float32 in [0,1] (8x8 block mean).

    ITU-R 601 luma weights, then 8x8 block mean."""
    f = np.asarray(frames)
    *lead, h, w, _ = f.shape
    g = f[..., 0] * np.float32(GRAY_W[0])
    g += f[..., 1] * np.float32(GRAY_W[1])
    g += f[..., 2] * np.float32(GRAY_W[2])
    return g.reshape(*lead, h // 8, 8, w // 8, 8).mean(axis=(-3, -1)) / np.float32(255.0)


def track_centroids(clip, sat_thr=0.4, chroma_thr=60):
    """Per-frame centroid (x=col, y=row) of saturated pixels; (64,2) float32, NaN-free.

    Mask = HSV saturation (max-min)/max > sat_thr AND chroma max-min > chroma_thr
    (both: S alone fires on dark noisy pixels, chroma alone on bright non-grey bg).
    If a frame has no pixels, fall back to chroma > 25; if still none, carry the
    previous centroid (leading misses back-filled from the first hit; if never
    found: image centre). Returns (xy, n_missing_frames).
    """
    c = np.asarray(clip)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    chroma = mx - np.minimum(np.minimum(r, g), b)          # uint8, no overflow
    T_, H, W = mx.shape
    satok = chroma.astype(np.float32) > np.float32(sat_thr) * mx   # (max-min)/max > sat_thr
    m = satok & (chroma > chroma_thr)
    cnt = m.sum(axis=(1, 2))
    if (cnt == 0).any():
        lo = satok & (chroma > 25)
        m[cnt == 0] = lo[cnt == 0]
        cnt = m.sum(axis=(1, 2))
    xs = np.arange(W, dtype=np.float64)
    ys = np.arange(H, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        cx = (m.sum(axis=1) @ xs) / cnt
        cy = (m.sum(axis=2) @ ys) / cnt
    out = np.stack([cx, cy], axis=1).astype(np.float32)
    out[cnt == 0] = np.nan
    miss = np.isnan(out[:, 0])
    n_miss = int(miss.sum())
    if n_miss == T_:
        out[:] = (W / 2, H / 2)
    elif n_miss:
        first = int(np.argmax(~miss))
        out[:first] = out[first]
        for k in range(first + 1, T_):
            if miss[k]:
                out[k] = out[k - 1]
    return out, n_miss


def com_scale(cam):
    """(m_per_px, origin_u) for the cube centre height (CONTRACT v1.1 `_com` pair), else floor."""
    if "m_per_px_com" in cam and "origin_px_com" in cam:
        return float(cam["m_per_px_com"]), float(cam["origin_px_com"][0])
    return float(cam["m_per_px_floor"]), float(cam["origin_px"][0])


def oracle_xc_px(state, cam):
    """state (64,6) -> x in px, unquantised (64,) float32."""
    mpp, x0 = com_scale(cam)
    return (x0 + state[:, 0].astype(np.float64) / mpp).astype(np.float32)


def oracle_x_px(state, cam):
    """state (64,6) -> x in px rounded to 1 px (64,) float32."""
    mpp, x0 = com_scale(cam)
    return np.round(x0 + state[:, 0].astype(np.float64) / mpp).astype(np.float32)


def _stop_traj(t, v0, a):
    ts = v0 / a
    tt = np.minimum(t, ts)
    return v0 * tt - 0.5 * a * tt ** 2


_GRID = {}


def _grid(fps):
    """Coarse (v0, a) grid of stop-model displacement curves, cached per fps."""
    if fps not in _GRID:
        lv = np.linspace(np.log(0.05), np.log(10.0), 120)
        la = np.linspace(np.log(0.1), np.log(40.0), 120)
        V, A = (z.ravel() for z in np.meshgrid(np.exp(lv), np.exp(la), indexing="ij"))
        t = np.arange(T) / fps
        _GRID[fps] = (V, A, _stop_traj(t[None], V[:, None], A[:, None]))
    return _GRID[fps]


def fit_v0_a(x_m, fps=64.0, quantum_m=None):
    """Fit v0 and deceleration a from (quantised) x(t) in metres.

    Model (constant deceleration until stop, then rest):
        x(t) = x0 + v0 tau - a tau^2 / 2,   tau = min(t, v0/a).
    Least squares over all frames (scipy least_squares in (x0, log v0, log a)),
    initialised from the best point of a coarse log-grid. Fitting the plain parabola
    only on the moving frames was tried and is much worse at 1 px quantisation
    (heavy, high-mu cubes move < 8 frames): including the rest phase pins the
    stopping distance d = v0^2/(2a). On synthetic quantised push trajectories:
    corr(log v0) 0.99, corr(a) 0.92 (parabola-on-moving-phase: 0.95 / 0.59).
    Fallbacks: if fewer than 3 frames move or the optimiser fails -> finite
    differences v0 = (x1-x0)*fps, a = v0^2/(2 d_total) ("fd"); no movement at all
    (< 1 quantum) -> v0 = half a quantum per frame, a = v0^2/quantum ("static").
    Returns (v0, a, method) with method in {"lsq", "fd", "static"}.
    """
    x = np.asarray(x_m, np.float64)
    q = quantum_m if quantum_m else 1e-9
    d = x - x[0]
    d_tot = d[-1]
    changed = np.nonzero(np.abs(np.diff(x)) > 0.5 * q)[0]
    if len(changed) == 0 or abs(d_tot) < 0.5 * q:
        v0 = 0.5 * q * fps
        return v0, v0 ** 2 / (2 * 0.5 * q), "static"
    d = d * (1.0 if d_tot > 0 else -1.0)

    def fd():
        v0 = max(d[1] * fps, 0.5 * q * fps)
        return float(v0), float(v0 ** 2 / (2 * max(d[-1], 0.5 * q))), "fd"

    if len(changed) < 2:          # < 3 frames carry motion information
        return fd()
    try:
        from scipy.optimize import least_squares
        V, A, M = _grid(fps)
        n = len(d)
        j = int(((M[:, :n] - d[None]) ** 2).sum(1).argmin())
        t = np.arange(n) / fps
        sol = least_squares(lambda p: p[0] + _stop_traj(t, np.exp(p[1]), np.exp(p[2])) - d,
                            [0.0, np.log(V[j]), np.log(A[j])])
        v0, a = float(np.exp(sol.x[1])), float(np.exp(sol.x[2]))
        if sol.success and np.isfinite(v0) and np.isfinite(a):
            return v0, a, "lsq"
    except Exception:
        pass
    return fd()


def floor_norm(col, floor_rel=CONTACT_FLOOR_REL):
    """|x| < floor_rel * max|col| -> 0, then divide by max|col| (all-zero column -> zeros).

    The floor comes first: dividing a ~1e-8 N solver residual by the clip max keeps it
    as a tiny but perfectly informative (1/mu) number."""
    col = np.asarray(col, np.float64)
    mx = float(np.max(np.abs(col))) if col.size else 0.0
    if not mx > 1e-12:
        return np.zeros_like(col)
    return np.where(np.abs(col) < floor_rel * mx, 0.0, col) / mx


def realistic_base(contact):
    """(64,3) [fn, ft, slip] -> (64,3) float64 [floor_norm(fn), floor_norm(ft),
    slip > 1 mm/s flag], noise-free. No absolute force scale and no raw slip speed."""
    c = np.asarray(contact, np.float64)
    out = np.empty((c.shape[0], 3), np.float64)
    out[:, 0] = floor_norm(c[:, 0])
    out[:, 1] = floor_norm(c[:, 1])
    out[:, 2] = c[:, 2] > SLIP_THR
    return out


def realistic_contact(contact, sigma, rng):
    """(64,3) [fn, ft, slip] -> (64,3) float32 contact/realistic:
    realistic_base (floored, per-clip max-normalised fn, ft; slip flag) with ADDITIVE
    N(0, sigma) on fn and ft (sigma on the normalised scale; the flag is kept binary)."""
    r = realistic_base(contact)
    r[:, :2] += sigma * rng.standard_normal((r.shape[0], 2))
    return r.astype(np.float32)


def noisy_contact(contact, p, rng):
    """(64,3) -> (64,3) float32 contact/noise_<p>: realistic_base + additive N(0, p) on
    every channel (fn/max, ft/max, slip flag) and every row, independently.

    The noise is on the normalised scale, so its amplitude cannot leak m; raw slip speed
    is never included (it is an exact kinematic oracle)."""
    r = realistic_base(contact)
    r += float(p) * rng.standard_normal(r.shape)
    return r.astype(np.float32)


# ----------------------------------------------------------------------------- worker

_W = {}


def _init(cfg):
    _W.update(cfg)


def _process(cid):
    cfg = _W
    name, groups = cfg["dataset"], cfg["groups"]
    out, tm = {}, {}
    seed = cfg["noise_seed"] + int(cid)

    if "pixels" in groups or "track" in groups:
        t0 = time.perf_counter()
        clip = np.asarray(io.load_clip(name, cid, mmap=True))  # one clip (~12.6 MB) at a time
        tm["load"] = time.perf_counter() - t0
        if "pixels" in groups:
            t0 = time.perf_counter()
            g = gray32(clip)                                    # (64,32,32)
            out["pixels/static"] = g[0].ravel()
            out["pixels/last"] = g[-1].ravel()
            out["pixels/clip_seq"] = g.reshape(8, 8, 32, 32).mean(axis=1).ravel()
            tm["pixels"] = time.perf_counter() - t0
        if "track" in groups:
            t0 = time.perf_counter()
            xy, n_miss = track_centroids(clip)
            out["track/clip"] = xy.ravel()
            out["_track_miss"] = n_miss
            tm["track"] = time.perf_counter() - t0
        del clip

    if "state" in groups:
        t0 = time.perf_counter()
        st = io.load_state(name, cid)
        cam = cfg["camera"]
        xpx = oracle_x_px(st, cam)
        out["state/oracle_x"] = xpx
        xc = oracle_xc_px(st, cam)
        out["state/oracle_xc"] = xc
        mpp, u0 = com_scale(cam)
        # unquantised x(t) in metres (CONTRACT v1.2: [log v0, a] fitted to x(t)); 1 um quantum
        # only sets the static/fd fallback thresholds
        x_m = (xc.astype(np.float64) - u0) * mpp
        v0, a, method = fit_v0_a(x_m, fps=cfg["fps"], quantum_m=1e-6)
        out["state/oracle_fit"] = np.array([np.log(v0), a], np.float32)
        out["_fit_method"] = method
        tm["state"] = time.perf_counter() - t0

    if "contact" in groups:
        t0 = time.perf_counter()
        c = io.load_contact(name, cid)
        out["contact/raw"] = np.asarray(c, np.float32).ravel()
        out["contact/realistic"] = realistic_contact(
            c, REALISTIC_NOISE, np.random.default_rng(seed)).ravel()
        if cfg["noise_curve"]:
            for frac in NOISE_CURVE:
                # same rng seed for every level -> same draw, scaled: monotone curve
                out[f"contact/noise_{int(round(frac * 100)):03d}"] = noisy_contact(
                    c, frac, np.random.default_rng(seed)).ravel()
        tm["contact"] = time.perf_counter() - t0
    return cid, out, tm


# ----------------------------------------------------------------------------- main

def block_info(rep, mode, dsj, extra=None):
    T_ = int(dsj.get("T", T))
    H = int(dsj.get("H", 256))
    base = dict(rep=rep, mode=mode, hf_id=None, revision=None, layer=None,
                n_frames=T_, input_res=None, pooling=None, oracle=False)
    spec = {
        ("pixels", "static"): dict(n_frames=1, input_res=32, pooling="gray_block8x8_mean"),
        ("pixels", "last"): dict(n_frames=1, input_res=32, pooling="gray_block8x8_mean"),
        ("pixels", "clip_seq"): dict(input_res=32, n_bins=8,
                                     pooling="gray_block8x8_mean+temporal_bin8_mean"),
        ("state", "oracle_x"): dict(pooling="x_px_round1", oracle=True),
        ("state", "oracle_xc"): dict(pooling="x_px_float", oracle=True,
                                     note="unquantised; the linear x(t) read-out reference: "
                                          "a ceiling for mu, not for log m"),
        ("state", "oracle_fit"): dict(pooling="lsq_stop_model_x0+v0*tau-a*tau^2/2_tau=min(t,v0/a)",
                                      columns=["log_v0", "a"], oracle=True,
                                      note="fitted on unquantised x(t) (state/oracle_xc)"),
        ("contact", "raw"): dict(pooling="flatten_64x3", columns=["fn_N", "ft_N", "slip_mps"],
                                 oracle=True),
        ("contact", "realistic"): dict(pooling="floor+per_clip_max_norm+additive_noise+slip_flag",
                                       noise=REALISTIC_NOISE, noise_kind="additive",
                                       floor_rel=CONTACT_FLOOR_REL, slip_thr_mps=SLIP_THR,
                                       oracle=False),
        ("track", "clip"): dict(input_res=H, pooling="centroid_sat>0.4&chroma>60",
                                columns=["x_px", "y_px"]),
    }
    if rep == "contact" and mode.startswith("noise_"):
        base.update(pooling="realistic_base+additive_gauss_all_channels", floor_rel=CONTACT_FLOOR_REL,
                    slip_thr_mps=SLIP_THR, oracle=False)
    base.update(spec.get((rep, mode), {}))
    if extra:
        base.update(extra)
    return base


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--only", default=",".join(ALL_GROUPS),
                    help="comma list of pixels,state,contact,track")
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--noise-curve", action="store_true",
                    help="also write contact/noise_{000,010,030,100} (realistic + N(0, p))")
    ap.add_argument("--chunksize", type=int, default=4)
    args = ap.parse_args(argv)

    groups = [g.strip() for g in args.only.split(",") if g.strip()]
    bad = set(groups) - set(ALL_GROUPS)
    if bad:
        ap.error(f"unknown groups {sorted(bad)}")
    if args.noise_curve and "contact" not in groups:
        groups.append("contact")

    name = args.dataset
    meta = io.load_meta(name)
    dsj = load_dataset_json(name)
    ids = [str(r["id"]) for r in meta]
    n = len(ids)
    commit = git_commit()
    cfg = dict(dataset=name, groups=groups, noise_seed=args.noise_seed,
               noise_curve=args.noise_curve, camera=dsj.get("camera", {}),
               fps=float(dsj.get("fps", 64)))
    if "state" in groups and not {"m_per_px_floor", "origin_px"} <= set(cfg["camera"]):
        sys.exit("dataset.json camera lacks m_per_px_floor/origin_px (needed for state/*)")
    print(f"[build] {name}: n={n} groups={groups} workers={args.workers} "
          f"noise_curve={args.noise_curve} commit={commit}", flush=True)

    res = {}            # block -> list of arrays in meta order
    tsum = {}
    fit_methods = {"lsq": 0, "fd": 0, "static": 0}
    track_miss_clips = track_miss_frames = 0
    t_start = time.perf_counter()

    def consume(i, cid, out, tm):
        nonlocal track_miss_clips, track_miss_frames
        assert cid == ids[i]
        for k, v in out.items():
            if k == "_fit_method":
                fit_methods[v] += 1
            elif k == "_track_miss":
                track_miss_frames += v
                track_miss_clips += v > 0
            else:
                res.setdefault(k, []).append(v)
        for k, v in tm.items():
            tsum[k] = tsum.get(k, 0.0) + v
        if (i + 1) % max(1, n // 20) == 0 or i + 1 == n:
            el = time.perf_counter() - t_start
            print(f"[build] {i + 1}/{n}  {el:.1f}s  eta {el / (i + 1) * (n - i - 1):.0f}s",
                  flush=True)

    if args.workers <= 1:
        _init(cfg)
        for i, cid in enumerate(ids):
            consume(i, *_process(cid))
    else:
        # spawn, not fork: a child forked after BLAS init can hang (quick-look attempt 1).
        # Workers start with 1 BLAS/OpenMP thread each (env is read when numpy loads in the child).
        ctx = mp.get_context("spawn")
        saved = {k: os.environ.get(k) for k in THREAD_VARS}
        os.environ.update({k: "1" for k in THREAD_VARS})
        try:
            pool = ctx.Pool(args.workers, initializer=_init, initargs=(cfg,))
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        with pool:
            for i, r in enumerate(pool.imap(_process, ids, chunksize=args.chunksize)):
                consume(i, *r)
    wall = time.perf_counter() - t_start

    load_pc = tsum.get("load", 0.0) / max(n, 1)
    print(f"[build] wall {wall:.1f}s ({wall / max(n, 1) * 1e3:.1f} ms/clip wall); "
          "per-clip worker time: " +
          ", ".join(f"{k} {v / max(n, 1) * 1e3:.1f} ms" for k, v in tsum.items()), flush=True)

    for key in sorted(res):
        rep, mode = key.split("/")
        X = np.stack(res[key]).astype(np.float16 if key in FLOAT16_BLOCKS else np.float32)
        assert len(X) == n and np.isfinite(X).all(), key
        grp_t = tsum.get(rep, 0.0) / max(n, 1)
        if rep in ("pixels", "track"):
            grp_t += load_pc
        extra = dict(sec_per_clip=round(grp_t, 6), git_commit=commit, dim=int(X.shape[1]))
        if key == "state/oracle_fit":
            extra["fit_method_counts"] = dict(fit_methods)
        if key == "track/clip":
            extra.update(missing_frames=int(track_miss_frames),
                         clips_with_missing=int(track_miss_clips))
        if key == "contact/realistic":
            extra["noise_seed"] = args.noise_seed
        if mode.startswith("noise_"):
            frac = int(mode.split("_")[1]) / 100
            extra.update(noise_sigma_rel=frac, noise_seed=args.noise_seed,
                         note="realistic_base (floored per-clip normalised fn, ft; slip flag) + "
                              "N(0, p) on every channel and row; p on the normalised scale")
        io.save_features(name, rep, mode, ids, X, block_info(rep, mode, dsj, extra))
        print(f"[build] wrote {rep}/{mode}  X{X.shape}", flush=True)
    if "state" in groups:
        print(f"[build] oracle_fit methods: {fit_methods}", flush=True)
    if "track" in groups:
        print(f"[build] track: {track_miss_frames} missing frames in "
              f"{track_miss_clips} clips (carried forward)", flush=True)


if __name__ == "__main__":
    main()
