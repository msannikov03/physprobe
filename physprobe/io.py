"""Data and feature I/O per CONTRACT v1. Imports: numpy + stdlib only.

CLI: python -m physprobe.io validate <dataset>
"""
import datetime
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

from physprobe import paths

T, H, W = 64, 256, 256
FPS = 64
CLIP_SHAPE = (T, H, W, 3)
CONTACT_SHAPE = (T, 3)
STATE_SHAPE = (T, 6)
CONTACT_CHANNELS = ["fn_N", "ft_N", "slip_mps"]
STATE_CHANNELS = ["x", "y", "z", "vx", "vy", "vz"]
SPLITS = ("train", "val", "test", "ood", "smoke")
FRAME0_SCENARIOS = ("push", "slide")

root = paths.root
dataset_dir = paths.dataset_dir


def git_commit():
    """Short HEAD of the code clone, with '-dirty' appended when tracked files outside results/ differ
    from HEAD (as `git describe --dirty`, untracked files ignored). results/ is excluded because the
    canonical outputs are tracked: otherwise every canonical run after the first would stamp itself
    '-dirty'. 'unknown' outside git."""
    def _git(*args):
        return subprocess.check_output(["git", *args], cwd=paths.code_root(), stderr=subprocess.DEVNULL,
                                       text=True).strip()
    try:
        head = _git("rev-parse", "--short", "HEAD")
    except Exception:
        return "unknown"
    try:
        dirty = bool(_git("status", "--porcelain", "--untracked-files=no", "--", ".", ":!results"))
    except Exception:
        dirty = True
    return head + ("-dirty" if dirty else "")


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def canonical_split(seed):
    r = int(seed) % 20
    return "train" if r <= 13 else ("val" if r <= 16 else "test")


def _chmod_umask(path):
    m = os.umask(0)
    os.umask(m)
    os.chmod(path, 0o666 & ~m)


def _atomic_save_npy(path, arr):
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=".npy")
    os.close(fd)
    try:
        np.save(tmp, arr)
        _chmod_umask(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _atomic_write_text(path, text):
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=".txt")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    _chmod_umask(tmp)
    os.replace(tmp, path)


def _check(arr, shape, dtype, name):
    arr = np.asarray(arr)
    if arr.shape != shape:
        raise ValueError(f"{name}: shape {arr.shape} != {shape}")
    if arr.dtype != dtype:
        raise ValueError(f"{name}: dtype {arr.dtype} != {np.dtype(dtype)}")
    return arr


# ---------------------------------------------------------------- data: write

def write_clip(ds_dir, id, clip, contact, state):
    """Atomically write clip_<id>.npy, contact_<id>.npy, state_<id>.npy."""
    clip = _check(clip, CLIP_SHAPE, np.uint8, "clip")
    contact = _check(contact, CONTACT_SHAPE, np.float32, "contact")
    state = _check(state, STATE_SHAPE, np.float32, "state")
    for kind, arr in (("clip", clip), ("contact", contact), ("state", state)):
        _atomic_save_npy(os.path.join(ds_dir, f"{kind}_{id}.npy"), arr)


def write_meta(ds_dir, rows):
    """meta.jsonl: one JSON object per clip, sorted by id."""
    rows = sorted(rows, key=lambda r: r["id"])
    text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    _atomic_write_text(os.path.join(ds_dir, "meta.jsonl"), text)


def write_dataset_json(ds_dir, d):
    d = dict(d)
    d.setdefault("T", T)
    d.setdefault("H", H)
    d.setdefault("W", W)
    d.setdefault("fps", FPS)
    d.setdefault("contact_channels", CONTACT_CHANNELS)
    d.setdefault("state_channels", STATE_CHANNELS)
    d.setdefault("git_commit", git_commit())
    d.setdefault("created", now_iso())
    _atomic_write_text(os.path.join(ds_dir, "dataset.json"), json.dumps(d, indent=1, sort_keys=True) + "\n")


# ----------------------------------------------------------------- data: read

def load_meta(name):
    p = os.path.join(dataset_dir(name), "meta.jsonl")
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_dataset_json(name):
    with open(os.path.join(dataset_dir(name), "dataset.json")) as f:
        return json.load(f)


def _load(name, kind, id, mmap):
    p = os.path.join(dataset_dir(name), f"{kind}_{id}.npy")
    return np.load(p, mmap_mode="r" if mmap else None)


def load_clip(name, id, mmap=True):
    return _load(name, "clip", id, mmap)


def load_contact(name, id, mmap=False):
    return _load(name, "contact", id, mmap)


def load_state(name, id, mmap=False):
    return _load(name, "state", id, mmap)


# -------------------------------------------------------------------- features

def save_features(name, rep, mode, ids, X, info):
    """features/<name>/<rep>/<mode>.npz with keys ids (<U16), X, info (JSON string)."""
    ids = np.asarray(list(ids), dtype="<U16")
    X = np.asarray(X)
    if X.dtype not in (np.float16, np.float32):
        X = X.astype(np.float32)
    if X.shape[0] != ids.shape[0]:
        raise ValueError(f"X rows {X.shape[0]} != ids {ids.shape[0]}")
    info = dict(info)
    info.setdefault("rep", rep)
    info.setdefault("mode", mode)
    info.setdefault("git_commit", git_commit())
    info.setdefault("created", now_iso())
    path = paths.features_path(name, rep, mode)
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=".npz")
    os.close(fd)
    try:
        np.savez(tmp, ids=ids, X=X, info=np.array(json.dumps(info, sort_keys=True)))
        _chmod_umask(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def load_features(name, rep, mode):
    """-> (ids: list[str], X: ndarray, info: dict)"""
    with np.load(paths.features_path(name, rep, mode)) as z:
        ids = [str(i) for i in z["ids"]]
        X = np.asarray(z["X"])
        info = json.loads(str(z["info"]))
    return ids, X, info


# ---------------------------------------------------------------------- splits

def split_ids(meta, seed=0):
    """seed 0: canonical split from meta (seed % 20). seeds >= 1: reshuffle train/val/test 70/15/15.
    ood and smoke ids are never reshuffled. -> {split: sorted ids}"""
    out = {s: [] for s in SPLITS}
    main = []
    for r in meta:
        sp = r.get("split") or canonical_split(r["seed"])
        if sp in ("ood", "smoke"):
            out[sp].append(r["id"])
        else:
            main.append((r["id"], sp))
    if int(seed) == 0:
        for i, sp in main:
            out[sp].append(i)
    else:
        ids = sorted(i for i, _ in main)
        n = len(ids)
        perm = np.random.default_rng(int(seed)).permutation(n)
        n_tr = int(round(0.70 * n))
        n_va = int(round(0.15 * n))
        for k, j in enumerate(perm):
            out["train" if k < n_tr else ("val" if k < n_tr + n_va else "test")].append(ids[j])
    return {s: sorted(v) for s, v in out.items() if v}


# -------------------------------------------------------------------- validate

SIM_SCENARIOS = ("push", "slide", "poscheck")
SPLIT_TARGET = {"train": 0.70, "val": 0.15, "test": 0.15}
OK_FRAC_MIN = 0.95
X_BACK_TOL_M = 1e-6          # allowed backward step of x(t) on push/slide (float32 noise is ~1e-8)
FN_REL_TOL = 1e-4            # |fn / (m g) - 1| (sim-v1 measures <= 6e-8)
STATE0_TOL = 1e-5            # state row 0 vs [x0, 0, rest_z, v0, 0, 0]
PARAM_REL_TOL = 1e-12        # m, mu, x0 reproduced from default_rng(seed) (allows last-ulp exp differences)
MAX_LISTED = 100
CAMERA_KEYS = ("pos", "fovy", "m_per_px_floor", "origin_px")
CAMERA_KEYS_V11 = ("m_per_px_com", "origin_px_com", "m_per_px_top", "origin_px_top")


def _expected_params(seed, ranges, scenario, path_m):
    """m log-uniform, then mu uniform, then (poscheck) x0 uniform on [0, path_m], as physprobe.sim.generate."""
    rng = np.random.default_rng(int(seed))
    lo, hi = float(ranges["m"][0]), float(ranges["m"][1])
    out = {"m": float(np.exp(rng.uniform(np.log(lo), np.log(hi))))}
    out["mu"] = float(rng.uniform(float(ranges["mu"][0]), float(ranges["mu"][1])))
    if scenario == "poscheck" and path_m is not None:
        out["x0_m"] = float(rng.uniform(0.0, float(path_m)))
    return out


def _close(a, b, rel):
    return abs(float(a) - float(b)) <= rel * max(abs(float(a)), abs(float(b)), 1e-300)


def validate(name, verbose=True):
    """Check files, dtypes, shapes, NaN, ranges, splits, ok fractions, frame-0 identity and clip content.
    -> report dict with a 'problems' list (empty = valid).

    Dataset level: ok_frac and in_frame_frac >= 0.95; required camera keys; for push/slide with canonical
    splits (n >= 20) train/val/test all present and within 0.02 + 10/n of 70/15/15 (the 10/n term covers the
    deterministic seed % 20 rounding of a partial cycle), and a single frame0_hash.
    Per row: split == canonical_split(seed) under canonical splitting; values inside dataset.json ranges;
    for sim datasets (dataset.json has base_seed) seed == base_seed + id and m, mu (x0) reproduce
    default_rng(seed); push v0 == J/m; contact fn > 0 in every row (push/slide/poscheck) and, when
    dataset.json has g, |fn/(m g) - 1| <= 1e-4; state x non-decreasing on push/slide; state row 0 ==
    [x0, 0, rest_z, v0, 0, 0] when dataset.json has rest_z_m.
    Sampled clips (10 random + min m + max m): frame 0 not constant, identical across clips on push/slide,
    and the last frame differs from frame 0 when moved_px >= 3."""
    ds_dir = dataset_dir(name)
    problems = []
    try:
        meta = load_meta(name)
    except FileNotFoundError:
        meta = []
        problems.append("meta.jsonl missing")
    try:
        dsj = load_dataset_json(name)
    except FileNotFoundError:
        dsj = {}
        problems.append("dataset.json missing")
    ids = [r["id"] for r in meta]
    if not meta:
        problems.append("empty meta")
    if ids != sorted(ids):
        problems.append("meta not sorted by id")
    if len(set(ids)) != len(ids):
        problems.append("duplicate ids")
    if dsj and dsj.get("n") != len(meta):
        problems.append(f"dataset.json n={dsj.get('n')} != {len(meta)} rows in meta")
    if dsj:
        cam = dsj.get("camera") or {}
        need = CAMERA_KEYS + (CAMERA_KEYS_V11 if "sim_version" in dsj else ())
        miss = [k for k in need if k not in cam]
        if miss:
            problems.append(f"dataset.json camera lacks {miss}")
    ranges = dsj.get("ranges", {})
    split_mode = dsj.get("split_mode", "canonical")
    base_seed = dsj.get("base_seed")
    g = dsj.get("g")
    rest_z = dsj.get("rest_z_m")
    check_idx = set()
    if meta:
        check_idx = set(np.random.default_rng(0).choice(len(meta), size=min(10, len(meta)), replace=False).tolist())
        ms = [r.get("m") for r in meta]
        if all(m is not None for m in ms):
            check_idx |= {int(np.argmin(ms)), int(np.argmax(ms))}
    frame0 = None
    n_ok = n_in = 0
    splits = {}
    hashes = set()
    for k, r in enumerate(meta):
        i = r["id"]
        sc = r.get("scenario")
        arrs = {}
        for kind, shape, dtype in (("clip", CLIP_SHAPE, np.uint8), ("contact", CONTACT_SHAPE, np.float32),
                                   ("state", STATE_SHAPE, np.float32)):
            p = os.path.join(ds_dir, f"{kind}_{i}.npy")
            if not os.path.exists(p):
                problems.append(f"{i}: missing {kind}")
                continue
            a = np.load(p, mmap_mode="r")
            if a.shape != shape or a.dtype != dtype:
                problems.append(f"{i}: {kind} shape/dtype {a.shape} {a.dtype} != {shape} {np.dtype(dtype)}")
                continue
            if kind != "clip":
                a = np.asarray(a)
                if not np.isfinite(a).all():
                    problems.append(f"{i}: {kind} has NaN/inf")
                    continue
                arrs[kind] = a
            elif k in check_idx:
                f0 = np.asarray(a[0]).astype(np.int16)
                if int(f0.max()) == int(f0.min()):
                    problems.append(f"{i}: frame 0 is constant ({int(f0.max())})")
                if sc in FRAME0_SCENARIOS:
                    if frame0 is None:
                        frame0 = f0
                    elif int(np.abs(f0 - frame0).max()) != 0:
                        d = int(np.abs(f0 - frame0).max())
                        problems.append(f"{i}: frame 0 differs from reference (max|d|={d})")
                    if (r.get("moved_px") or 0) >= 3 and np.array_equal(np.asarray(a[-1]), np.asarray(a[0])):
                        problems.append(f"{i}: last frame equals frame 0 although moved_px={r.get('moved_px')}")
        for key, rng_ in ranges.items():
            if r.get(key) is not None and isinstance(rng_, (list, tuple)) and len(rng_) >= 2:
                lo, hi = float(rng_[0]), float(rng_[1])
                if not (lo <= float(r[key]) <= hi):
                    problems.append(f"{i}: {key}={r[key]} outside [{lo}, {hi}]")
        sp = r.get("split")
        if sp not in SPLITS:
            problems.append(f"{i}: unknown split {sp!r}")
        elif split_mode == "canonical" and sp in SPLIT_TARGET and "seed" in r and sp != canonical_split(r["seed"]):
            problems.append(f"{i}: split {sp} != canonical {canonical_split(r['seed'])} (seed {r['seed']})")
        if sc in SIM_SCENARIOS and base_seed is not None and "seed" in r:
            if int(r["seed"]) != int(base_seed) + int(i):
                problems.append(f"{i}: seed {r['seed']} != base_seed {base_seed} + id")
            if "m" in ranges and "mu" in ranges:
                exp = _expected_params(r["seed"], ranges, sc, dsj.get("path_m"))
                bad = [f"{key}={r.get(key)} (expected {v})" for key, v in exp.items()
                       if r.get(key) is None or not _close(r[key], v, PARAM_REL_TOL)]
                if bad:
                    problems.append(f"{i}: parameters do not reproduce default_rng(seed): {', '.join(bad)}")
        if sc == "push" and r.get("J") is not None and r.get("m"):
            if not _close(r.get("v0", float("nan")), float(r["J"]) / float(r["m"]), 1e-9):
                problems.append(f"{i}: v0={r.get('v0')} != J/m={float(r['J']) / float(r['m'])}")
        if sc in SIM_SCENARIOS and "contact" in arrs:
            fn = arrs["contact"][:, 0].astype(np.float64)
            if not (fn > 0).all():
                problems.append(f"{i}: contact fn <= 0 in {int((fn <= 0).sum())} rows")
            elif g is not None and r.get("m"):
                err = float(np.abs(fn / (float(r["m"]) * float(g)) - 1).max())
                if err > FN_REL_TOL:
                    problems.append(f"{i}: max |fn/(m g) - 1| = {err:.2e} > {FN_REL_TOL}")
        if sc in SIM_SCENARIOS and "state" in arrs:
            st = arrs["state"].astype(np.float64)
            if sc in FRAME0_SCENARIOS:
                back = float(np.diff(st[:, 0]).min())
                if back < -X_BACK_TOL_M:
                    problems.append(f"{i}: state x decreases by {-back:.2e} m")
            if rest_z is not None and r.get("v0") is not None:
                want = np.array([r.get("x0_m") or 0.0, 0.0, float(rest_z), float(r["v0"]), 0.0, 0.0])
                d0 = float(np.abs(st[0] - want).max())
                if d0 > STATE0_TOL:
                    problems.append(f"{i}: state row 0 {st[0].round(6).tolist()} != {want.round(6).tolist()}")
        n_ok += bool(r.get("ok"))
        n_in += bool(r.get("in_frame"))
        splits[sp] = splits.get(sp, 0) + 1
        if r.get("frame0_hash") is not None and sc in FRAME0_SCENARIOS:
            hashes.add(r["frame0_hash"])
    n = max(len(meta), 1)
    ok_frac, in_frac = n_ok / n, n_in / n
    if meta and ok_frac < OK_FRAC_MIN:
        problems.append(f"ok_frac {ok_frac:.4f} < {OK_FRAC_MIN}")
    if meta and in_frac < OK_FRAC_MIN:
        problems.append(f"in_frame_frac {in_frac:.4f} < {OK_FRAC_MIN}")
    scen = {r.get("scenario") for r in meta}
    if meta and split_mode == "canonical" and scen <= set(FRAME0_SCENARIOS) and len(meta) >= 20:
        tol = 0.02 + 10.0 / len(meta)
        for sp, target in SPLIT_TARGET.items():
            frac = splits.get(sp, 0) / len(meta)
            if sp not in splits:
                problems.append(f"split {sp} missing")
            elif abs(frac - target) > tol:
                problems.append(f"split {sp} fraction {frac:.4f} off target {target} by more than {tol:.4f}")
    if len(hashes) > 1:
        problems.append(f"{len(hashes)} distinct frame0_hash values in meta (push/slide expect 1)")
    n_problems = len(problems)
    if n_problems > MAX_LISTED:
        problems = problems[:MAX_LISTED] + [f"... and {n_problems - MAX_LISTED} more"]
    report = {"dataset": name, "n": len(meta), "ok_frac": round(ok_frac, 4), "in_frame_frac": round(in_frac, 4),
              "splits": splits, "frame0_checked": len(check_idx), "n_problems": n_problems, "problems": problems}
    if verbose:
        print(json.dumps(report, indent=1))
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 2 and argv[0] == "validate":
        rep = validate(argv[1])
        return 0 if not rep["problems"] else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
