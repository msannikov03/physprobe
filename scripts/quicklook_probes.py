#!/usr/bin/env python
"""Quick-look linear probes: frozen video features + data baselines -> (log m, mu) / x0.

Reduced version of plan section 4.3 (B2-B4): ridge only, no MLP, no preds/.
Inputs per dataset: every features/<ds>/<rep>/<mode>.npz, baselines built from data/
(cached in <out>/cache/), and two video + contact/realistic combinations.

Run on compute:
  source ~/data2/physprobe/env.sh && cd $PHYSPROBE_ROOT
  OMP_NUM_THREADS=8 nice -n 10 $FEAT_PY scripts/quicklook_probes.py \
      --datasets push slide poscheck --seeds 0 1 2 --out results/quicklook
"""
import argparse
import datetime
import json
import math
import os
import sys
import time
import warnings
import multiprocessing as mp

import numpy as np

try:
    from physprobe import io, paths
except ImportError:  # PYTHONPATH not set: fall back to the clone root
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from physprobe import io, paths
from physprobe.baselines.build import realistic_contact  # same block as features/<ds>/contact/realistic

ALPHAS = np.logspace(-3, 5, 17)
PCA_ABOVE = 4096
PCA_DIM = 512
TARGETS = {"push": ["log_m", "mu"], "slide": ["log_m", "mu"], "poscheck": ["x0_m"]}
SORT_BY = {"push": "mu", "slide": "mu", "poscheck": "x0_m"}
COMBOS = [("vjepa2_l", "clip_seq"), ("dinov2_b", "clip_seq")]
PIXEL_NAMES = ["pixels_clip_seq", "pixels_static", "pixels_last", "track_clip"]
RED_R, RED_G, RED_B = 150, 100, 100      # red mask: R > 150, G < 100, B < 100
NOISE_ADD = 0.05          # additive, on the per-clip normalised scale (build.REALISTIC_NOISE)
SMALL_CACHE = "small_v2"  # bump when state/contact cache semantics change (v2: floored realistic contact)

_T0 = time.time()
_LOGF = None
TIMING = {}


def log(msg):
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')} +{time.time() - _T0:7.1f}s] {msg}"
    print(line, flush=True)
    if _LOGF is not None:
        _LOGF.write(line + "\n")
        _LOGF.flush()


class Stage:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()
        log(f"start {self.name}")
        return self

    def __exit__(self, *exc):
        dt = time.time() - self.t
        TIMING[self.name] = round(TIMING.get(self.name, 0.0) + dt, 1)
        log(f"done  {self.name} in {dt:.1f}s")


# ------------------------------------------------------------------ targets

def target_values(meta, name):
    if name == "log_m":
        return np.log(np.array([r["m"] for r in meta], dtype=np.float64))
    return np.array([r[name] for r in meta], dtype=np.float64)


# ------------------------------------------------------------- baselines

def _pixel_track_one(args):
    """One clip -> grey 32x32 x 8 time bins, grey frame 0, grey frame 63, red centroid (64,2), n empty frames."""
    ds, cid = args
    c = np.asarray(io.load_clip(ds, cid, mmap=True))            # (64,256,256,3) uint8
    T = c.shape[0]
    g = c.sum(axis=-1, dtype=np.uint16)                          # R+G+B, max 765
    b = g.reshape(T, 32, 8, 32, 8).sum(axis=(2, 4), dtype=np.uint32).astype(np.float32) / (3.0 * 64.0)
    seq = b.reshape(8, T // 8, 32, 32).mean(axis=1).reshape(-1)  # 8 bins of 8 frames -> 8192
    mask = (c[..., 0] > RED_R) & (c[..., 1] < RED_G) & (c[..., 2] < RED_B)
    cnt = mask.sum(axis=(1, 2)).astype(np.float64)
    cols = np.arange(mask.shape[2], dtype=np.float64)
    rows = np.arange(mask.shape[1], dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = (mask.sum(axis=1) * cols).sum(axis=1) / cnt          # column centroid
        v = (mask.sum(axis=2) * rows).sum(axis=1) / cnt          # row centroid
    tr = np.stack([u, v], axis=1)
    return seq.astype(np.float32), b[0].reshape(-1).copy(), b[-1].reshape(-1).copy(), tr, int((cnt == 0).sum())


def _fill_track(tr):
    """Forward then backward fill frames where the mask was empty."""
    tr = tr.copy()
    for k in range(1, len(tr)):
        if np.isnan(tr[k]).any():
            tr[k] = tr[k - 1]
    for k in range(len(tr) - 2, -1, -1):
        if np.isnan(tr[k]).any():
            tr[k] = tr[k + 1]
    return np.nan_to_num(tr, nan=0.0)


def cache_path(cache_dir, ds, name):
    return os.path.join(cache_dir, f"{ds}_{name}.npy")


def _cached(cache_dir, ds, names, ids):
    idp = cache_path(cache_dir, ds, "ids")
    if not os.path.exists(idp) or list(np.load(idp)) != ids:
        return False
    return all(os.path.exists(cache_path(cache_dir, ds, n)) for n in names)


def build_pixels(ds, ids, cache_dir, workers):
    if _cached(cache_dir, ds, PIXEL_NAMES, ids):
        log(f"{ds}: pixel/track cache hit")
        return
    n = len(ids)
    seq = np.zeros((n, 8192), np.float32)
    st = np.zeros((n, 1024), np.float32)
    la = np.zeros((n, 1024), np.float32)
    tr = np.zeros((n, 128), np.float32)
    n_empty_frames = n_empty_clips = 0
    t = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:  # spawn: no fork after BLAS threads exist
        for k, (s, f0, fl, trk, ne) in enumerate(pool.imap(_pixel_track_one, [(ds, i) for i in ids], chunksize=4)):
            seq[k], st[k], la[k] = s, f0, fl
            tr[k] = _fill_track(trk).reshape(-1)
            n_empty_frames += ne
            n_empty_clips += ne > 0
            if (k + 1) % 250 == 0 or k + 1 == n:
                el = time.time() - t
                log(f"{ds}: pixel pass {k + 1}/{n} ({el:.0f}s, eta {el / (k + 1) * (n - k - 1):.0f}s)")
    log(f"{ds}: red mask empty in {n_empty_frames} frames over {n_empty_clips} clips (filled from neighbours)")
    for name, arr in zip(PIXEL_NAMES, (seq, st, la, tr)):
        np.save(cache_path(cache_dir, ds, name), arr)
    np.save(cache_path(cache_dir, ds, "ids"), np.array(ids))


def build_small(ds, ids, cache_dir):
    names = ["state_oracle_x", "contact_raw", "contact_realistic", SMALL_CACHE]
    if _cached(cache_dir, ds, names, ids):
        log(f"{ds}: state/contact cache hit")
        return
    cam = io.load_dataset_json(ds)["camera"]
    mpp, u0 = float(cam["m_per_px_top"]), float(cam["origin_px_top"][0])
    n = len(ids)
    ox = np.zeros((n, 64), np.float32)
    raw = np.zeros((n, 192), np.float32)
    real = np.zeros((n, 192), np.float32)
    for k, i in enumerate(ids):
        s = io.load_state(ds, i)
        ox[k] = np.round(u0 + s[:, 0] / mpp)                     # x(t) in pixels, 1 px quantisation
        c = np.asarray(io.load_contact(ds, i), dtype=np.float64)  # (64,3) fn, ft, slip
        raw[k] = c.reshape(-1)
        # build.realistic_contact: |fn|,|ft| < 1e-3*max -> 0 (post-stop solver residual, ~1e-8 N
        # proportional to m, would otherwise survive normalisation as a 1/mu code), per-clip max
        # normalisation, ADDITIVE N(0, 0.05), slip > 1 mm/s flag. Per-clip rng seed = int(id), as in
        # build.py with --noise-seed 0, so this cache equals features/<ds>/contact/realistic.
        real[k] = realistic_contact(c, NOISE_ADD, np.random.default_rng(int(i))).reshape(-1)
    for name, arr in zip(names[:3], (ox, raw, real)):
        np.save(cache_path(cache_dir, ds, name), arr)
    np.save(cache_path(cache_dir, ds, SMALL_CACHE), np.array(2))
    if not os.path.exists(cache_path(cache_dir, ds, "ids")):
        np.save(cache_path(cache_dir, ds, "ids"), np.array(ids))


# ----------------------------------------------------------------- inputs

def discover_features(ds):
    d = os.path.join(paths.features_dir(), ds)
    out = []
    for rep in sorted(os.listdir(d)):
        rd = os.path.join(d, rep)
        if os.path.isdir(rd):
            out += [(rep, f[:-4]) for f in sorted(os.listdir(rd)) if f.endswith(".npz") and not f.startswith(".")]
    return out


def load_feature_aligned(ds, rep, mode, ids):
    fids, X, _ = io.load_features(ds, rep, mode)
    if fids != ids:
        pos = {i: k for k, i in enumerate(fids)}
        missing = [i for i in ids if i not in pos]
        if missing:
            raise ValueError(f"{ds}/{rep}/{mode}: {len(missing)} ids missing")
        X = X[[pos[i] for i in ids]]
    X = np.asarray(X, dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError(f"{ds}/{rep}/{mode}: non-finite values")
    return X


def input_specs(ds, ids, cache_dir, with_pixels):
    """-> list of (rep, mode, inputs_name, loader) where loader() -> list of blocks (n, d_i)."""
    def cached(name):
        return lambda: [np.load(cache_path(cache_dir, ds, name)).astype(np.float32)]

    def feat(rep, mode):
        return lambda: [load_feature_aligned(ds, rep, mode, ids)]

    specs = [(rep, mode, f"{rep}/{mode}", feat(rep, mode)) for rep, mode in discover_features(ds)]
    base = [("state", "oracle_x"), ("contact", "raw"), ("contact", "realistic")]
    if with_pixels:
        base = [("pixels", "clip_seq"), ("pixels", "static"), ("pixels", "last"), ("track", "clip")] + base
    specs += [(rep, mode, f"{rep}/{mode}", cached(f"{rep}_{mode}")) for rep, mode in base]
    have = {(r, m) for r, m, _, _ in specs}
    for rep, mode in COMBOS:
        if (rep, mode) in have:
            def loader(rep=rep, mode=mode):
                return [load_feature_aligned(ds, rep, mode, ids),
                        np.load(cache_path(cache_dir, ds, "contact_realistic")).astype(np.float32)]
            specs.append((f"{rep}+contact", f"{mode}+realistic", f"{rep}/{mode} + contact/realistic", loader))
    return specs


# ------------------------------------------------------------------ probe

def standardise(blocks, tr, te, block_scale):
    """Per-block z-score on train (constant train columns -> 0); combos: each block / sqrt(n_dims)."""
    outs_tr, outs_te = [], []
    for X in blocks:
        a = X[tr].astype(np.float64)
        b = X[te].astype(np.float64)
        varies = a.max(axis=0) != a.min(axis=0)
        mu = a.mean(axis=0)
        sd = a.std(axis=0)
        sd[~varies] = 1.0
        a = (a - mu) / sd
        b = (b - mu) / sd
        a[:, ~varies] = 0.0
        b[:, ~varies] = 0.0
        if block_scale:
            a /= math.sqrt(X.shape[1])
            b /= math.sqrt(X.shape[1])
        outs_tr.append(a.astype(np.float32))
        outs_te.append(b.astype(np.float32))
    return np.hstack(outs_tr), np.hstack(outs_te)


def r2_rmse(y, p):
    from sklearn.metrics import r2_score
    return float(r2_score(y, p)), float(np.sqrt(np.mean((y - p) ** 2)))


def probe_input(blocks, Y, tnames, split_idx):
    """Fit ridge on [targets, shuffled targets] in one RidgeCV (alpha per target column; GCV per column
    equals separate fits). -> list of dict rows (without dataset/rep/mode/inputs/seed)."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV

    tr, va, te = split_idx
    feat_dim = int(sum(b.shape[1] for b in blocks))
    Ztr, Zte = standardise(blocks, tr, te, block_scale=len(blocks) > 1)
    pca_dim = 0
    if feat_dim > PCA_ABOVE:
        k = min(PCA_DIM, len(tr) - 1)
        pca = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(Ztr)
        Ztr, Zte = pca.transform(Ztr), pca.transform(Zte)
        pca_dim = k
    Ytr, Yte = Y[tr], Y[te]
    perm = np.random.default_rng(0).permutation(len(tr))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(Ztr, np.hstack([Ytr, Ytr[perm]]))
    P = r.predict(Zte)
    al = np.atleast_1d(r.alpha_)
    nt = len(tnames)
    common = dict(n_train=len(tr), n_val=len(va), n_test=len(te), feat_dim=feat_dim, pca_dim=pca_dim)
    rows = []
    for j, t in enumerate(tnames):
        y = Yte[:, j]
        for probe, pred, alpha in (("ridge", P[:, j], al[j]), ("shuffle", P[:, nt + j], al[nt + j]),
                                   ("mean", np.full(len(te), Ytr[:, j].mean()), np.nan)):
            r2, rmse = r2_rmse(y, pred)
            rows.append(dict(target=t, probe=probe, alpha=float(alpha), r2=r2, rmse=rmse, **common))
    return rows


def load_ok_meta(ds):
    meta = io.load_meta(ds)
    n_all = len(meta)
    meta = [r for r in meta if r.get("ok", True)]
    if len(meta) != n_all:
        log(f"{ds}: excluded {n_all - len(meta)} clips with ok=false")
    return meta


def build_baselines(ds, cache_dir, with_pixels, workers):
    ids = [r["id"] for r in load_ok_meta(ds)]
    with Stage(f"{ds}: baselines"):
        if with_pixels:
            build_pixels(ds, ids, cache_dir, workers)
        build_small(ds, ids, cache_dir)


def run_dataset(ds, seeds, cache_dir, with_pixels):
    meta = load_ok_meta(ds)
    ids = [r["id"] for r in meta]
    tnames = TARGETS[ds]
    Y = np.stack([target_values(meta, t) for t in tnames], axis=1)
    pos = {i: k for k, i in enumerate(ids)}
    splits = {}
    for s in seeds:
        sp = io.split_ids(meta, s)
        splits[s] = tuple(np.array([pos[i] for i in sp.get(name, [])], dtype=int) for name in ("train", "val", "test"))
    rows = []
    with Stage(f"{ds}: probes"):
        specs = input_specs(ds, ids, cache_dir, with_pixels)
        log(f"{ds}: {len(specs)} inputs x {len(seeds)} seeds, targets {tnames}")
        for rep, mode, name, loader in specs:
            t = time.time()
            blocks = loader()
            for s in seeds:
                for row in probe_input(blocks, Y, tnames, splits[s]):
                    rows.append(dict(dataset=ds, rep=rep, mode=mode, inputs=name, seed=s, **row))
            del blocks
            msg = ", ".join(f"{t_}={np.mean([r['r2'] for r in rows[-3 * len(tnames) * len(seeds):] if r['target'] == t_ and r['probe'] == 'ridge']):.3f}"
                            for t_ in tnames)
            log(f"{ds}: {name:<40s} R2 {msg}  ({time.time() - t:.1f}s)")
    return rows


# ------------------------------------------------------------------ report

def agg(df):
    g = df[df.probe == "ridge"].groupby(["dataset", "inputs", "target"])["r2"]
    return g.agg(["mean", "std", "min", "max"]).reset_index()


def write_table(df, path):
    a = agg(df)
    lines = ["# Quick-look linear probes (ridge), test R², mean ± std over split seeds", "",
             "Std is the sample std (ddof 1) over the seeds. Ridge = RidgeCV(alphas=logspace(-3,5,17)) on train after "
             "train-only z-scoring; inputs above 4096 dims go through PCA to 512 (train-only, randomised, seed 0).", ""]
    for ds in [d for d in TARGETS if d in set(df.dataset)]:
        tnames = TARGETS[ds]
        sub = a[a.dataset == ds]
        piv_m = sub.pivot(index="inputs", columns="target", values="mean")
        piv_s = sub.pivot(index="inputs", columns="target", values="std")
        piv_m = piv_m.sort_values(SORT_BY[ds], ascending=False)
        dims = df[(df.dataset == ds) & (df.probe == "ridge")].groupby("inputs")[["feat_dim", "pca_dim"]].first()
        seeds = sorted(set(df[df.dataset == ds].seed))
        n = df[(df.dataset == ds)].iloc[0]
        lines += [f"## {ds} (n_train {n.n_train}, n_test {n.n_test}, seeds {seeds}; sorted by R²({SORT_BY[ds]}))", ""]
        lines.append("| input | dim | " + " | ".join(f"R²({t})" for t in tnames) + " |")
        lines.append("|---|---|" + "---|" * len(tnames))
        for inp in piv_m.index:
            fd, pd_ = dims.loc[inp, "feat_dim"], dims.loc[inp, "pca_dim"]
            dim = f"{fd}" + (f"→{pd_}" if pd_ else "")
            cells = [f"{piv_m.loc[inp, t]:.3f} ± {piv_s.loc[inp, t]:.3f}" for t in tnames]
            lines.append(f"| {inp} | {dim} | " + " | ".join(cells) + " |")
        ctl = df[df.dataset == ds]
        sh, mn = ctl[ctl.probe == "shuffle"].r2, ctl[ctl.probe == "mean"].r2
        lines += ["", f"Controls ({ds}): shuffle R² in [{sh.min():.3f}, {sh.max():.3f}] over {len(sh)} fits; "
                      f"mean R² in [{mn.min():.3f}, {mn.max():.3f}].", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _family(inp):
    if inp.startswith("state/") or inp == "contact/raw":
        return "oracle"
    if "contact/realistic" in inp:
        return "contact"
    rep = inp.split("/")[0]
    if rep.endswith("_random"):
        return "random"
    if rep in ("pixels", "track"):
        return "baseline"
    return "pretrained"


FAMILIES = {  # colour, hatch, legend label (blue / orange / aqua: validated all-pairs; greys carry texture)
    "pretrained": ("#2a78d6", None, "pretrained encoder"),
    "random": ("#eb6834", None, "random-init encoder"),
    "baseline": ("#1baf7a", None, "pixel / track baseline"),
    "contact": ("#9a9992", "...", "realistic contact (alone or + video)"),
    "oracle": ("#c3c2b7", "///", "oracle (state x(t), raw contact)"),
}


def write_headline(df, path, ds="push"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    a = agg(df[df.dataset == ds])
    tnames = TARGETS[ds]
    m = a.pivot(index="inputs", columns="target", values="mean").sort_values(SORT_BY[ds], ascending=True)
    s = a.pivot(index="inputs", columns="target", values="std").loc[m.index]
    n = len(m)
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": ink2, "axes.labelcolor": ink2,
                         "xtick.color": ink2, "ytick.color": ink, "hatch.color": "#52514e", "hatch.linewidth": 0.6})
    fig, axes = plt.subplots(1, len(tnames), figsize=(8.3, 0.24 * n + 1.9), sharey=True)
    labels = {"log_m": "ln m", "mu": "μ", "x0_m": "x0"}
    y = np.arange(n)
    lo = min(-0.1, float(np.nanmin(m.values - s.values)) - 0.02)
    for ax, t in zip(np.atleast_1d(axes), tnames):
        for k, inp in enumerate(m.index):
            col, hat, _ = FAMILIES[_family(inp)]
            ax.barh(k, m.loc[inp, t], height=0.72, color=col, hatch=hat, edgecolor="#fcfcfb" if hat is None else ink2,
                    linewidth=0.4 if hat else 0.8)
        ax.errorbar(m[t].values, y, xerr=s[t].values, fmt="none", ecolor=ink, elinewidth=0.8, capsize=1.8)
        ax.axvline(0, color=ink2, lw=0.8)
        ax.set_xlim(lo, 1.02)
        ax.set_title(f"R²({labels[t]})", color=ink, fontsize=9.5)
        ax.set_xlabel("test R², mean ± sd over 3 splits")
        ax.grid(axis="x", color=grid, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    np.atleast_1d(axes)[0].set_yticks(y)
    np.atleast_1d(axes)[0].set_yticklabels(m.index)
    handles = [Patch(facecolor=c, hatch=h, edgecolor=ink2 if h else c, label=l) for c, h, l in FAMILIES.values()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(f"{ds}: linear probes on frozen features vs baselines (sorted by R²({labels[SORT_BY[ds]]}))",
                 color=ink, fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(path, dpi=200, facecolor="#fcfcfb")
    plt.close(fig)


def write_gates(df, path):
    r = df[df.probe == "ridge"]
    out = ["# Sanity gates (quick-look)", "",
           "A gate passes only if every seed passes. Numbers are mean [worst seed] over the split seeds. "
           "These are the plan-v1 rules: G3 (quantised state/oracle_x ≥ 0.9) and G5 (single shuffle ≤ 0.05) "
           "are superseded by CONTRACT v1.2 (unquantised oracle_xc; label-permutation null with an "
           "n-aware bound), so a FAIL there is expected; the canonical gates are in results/<ds>/gates.md.",
           ""]
    results = []

    def stat(sub):
        return sub.groupby(["dataset", "inputs", "target"])["r2"].agg(["mean", "min", "max"]).reset_index()

    def gate(gid, text, rows, ok_fn, worst_col):
        if rows.empty:
            results.append((gid, "SKIP"))
            out.extend([f"## {gid} SKIP: {text}", "", "no matching rows", ""])
            return
        st = stat(rows)
        st["pass"] = st.apply(ok_fn, axis=1)
        verdict = "PASS" if st["pass"].all() else "FAIL"
        results.append((gid, verdict))
        out.extend([f"## {gid} {verdict}: {text}", "", "| dataset | input | target | mean | worst seed | ok |",
                    "|---|---|---|---|---|---|"])
        for _, x in st.iterrows():
            out.append(f"| {x.dataset} | {x.inputs} | {x.target} | {x['mean']:.3f} | {x[worst_col]:.3f} | "
                       f"{'PASS' if x['pass'] else 'FAIL'} |")
        out.append("")

    video = r.rep.str.match(r"^(pixels|track|dinov2_b|dinov2_b_random|vjepa2_l|vjepa2_l_random)$")
    gate("G1", "static-mode inputs on push and slide, R² ≤ 0.05 for log_m and mu",
         r[r.dataset.isin(["push", "slide"]) & (r["mode"] == "static")], lambda x: x["max"] <= 0.05, "max")
    gate("G2", "slide: R²(log_m) ≤ 0.05 for every video input (pixels, track, dinov2_*, vjepa2_*)",
         r[(r.dataset == "slide") & (r.target == "log_m") & video], lambda x: x["max"] <= 0.05, "max")
    g3 = r[(r.inputs == "state/oracle_x") & (((r.dataset == "push") & r.target.isin(["log_m", "mu"])) |
                                             ((r.dataset == "slide") & (r.target == "mu")))]
    gate("G3", "state/oracle_x: push R² ≥ 0.9 for log_m and mu; slide R²(mu) ≥ 0.9", g3,
         lambda x: x["min"] >= 0.9, "min")
    gate("G4", "poscheck: static modes of dinov2_b, vjepa2_l and pixels/static give R²(x0_m) ≥ 0.9",
         r[(r.dataset == "poscheck") & r.inputs.isin(["dinov2_b/static", "vjepa2_l/static", "pixels/static"])],
         lambda x: x["min"] >= 0.9, "min")
    sh = df[df.probe == "shuffle"]
    st = sh.groupby(["dataset", "target"])["r2"].agg(["min", "max", "count"]).reset_index()
    ok5 = bool(((sh.r2 <= 0.05) & (sh.r2 >= -0.5)).all())
    results.append(("G5", "PASS" if ok5 else "FAIL"))
    out.extend([f"## G5 {'PASS' if ok5 else 'FAIL'}: every shuffle control has -0.5 ≤ R² ≤ 0.05", "",
                "| dataset | target | fits | min R² | max R² |", "|---|---|---|---|---|"])
    out += [f"| {x.dataset} | {x.target} | {x['count']} | {x['min']:.3f} | {x['max']:.3f} |" for _, x in st.iterrows()]
    bad = sh[(sh.r2 > 0.05) | (sh.r2 < -0.5)]
    if len(bad):
        out += ["", "Failing shuffle fits:", ""] + [f"- {x.dataset} {x.inputs} {x.target} seed {x.seed}: {x.r2:.3f}"
                                                    for _, x in bad.iterrows()]
    diag = []
    for ds in [d for d in TARGETS if d in set(df.dataset)]:
        meta = [m_ for m_ in io.load_meta(ds) if m_.get("ok", True)]
        pos = {m_["id"]: k for k, m_ in enumerate(meta)}
        for t in TARGETS[ds]:
            yv = target_values(meta, t)
            for s in sorted(set(df[df.dataset == ds].seed)):
                tr = np.array([pos[i] for i in io.split_ids(meta, s)["train"]])
                ytr = yv[tr]
                c = float(np.corrcoef(ytr, ytr[np.random.default_rng(0).permutation(len(tr))])[0, 1])
                diag.append(f"| {ds} | {t} | {s} | {len(tr)} | {c:+.3f} |")
    out += ["", "Diagnostic: correlation between the permuted and the true train labels. When an input encodes the "
                "target almost perfectly, the shuffled ridge inherits this chance correlation and its test R² is "
                "roughly 2·corr, so a positive corr of about 0.1 explains shuffle R² near 0.2 without any leak.", "",
            "| dataset | target | seed | n_train | corr(y_perm, y) |", "|---|---|---|---|---|"] + diag
    out.append("")
    gate("G6", "contact/raw on push (oracle) R² ≥ 0.95 for log_m and mu",
         r[(r.dataset == "push") & (r.inputs == "contact/raw")], lambda x: x["min"] >= 0.95, "min")
    summary = ["## Summary", ""] + [f"- {g}: {v}" for g, v in results] + [""]
    with open(path, "w") as f:
        f.write("\n".join(out[:4] + summary + out[4:]) + "\n")
    return results


# -------------------------------------------------------------------- main

def main(argv=None):
    global _LOGF
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["push", "slide", "poscheck"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default="results/quicklook")
    ap.add_argument("--workers", type=int, default=4, help="processes for the pixel pass")
    ap.add_argument("--pixel-datasets", nargs="*", default=None,
                    help="datasets that get pixels/track baselines (default: all)")
    ap.add_argument("--report-only", action="store_true", help="rebuild table/figure/gates from probes.csv")
    args = ap.parse_args(argv)
    import pandas as pd

    out = args.out if os.path.isabs(args.out) else os.path.join(paths.root(), args.out)
    cache_dir = os.path.join(out, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    _LOGF = open(os.path.join(out, "run.log"), "a")
    log(f"quicklook_probes args={vars(args)} OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    csv = os.path.join(out, "probes.csv")
    cols = ["dataset", "rep", "mode", "inputs", "target", "probe", "seed", "n_train", "n_val", "n_test",
            "feat_dim", "pca_dim", "alpha", "r2", "rmse", "timestamp"]
    if args.report_only:
        df = pd.read_csv(csv)
    else:
        pix = set(args.datasets if args.pixel_datasets is None else args.pixel_datasets)
        for ds in args.datasets:  # all pixel passes first, before any BLAS/OpenMP threads start
            build_baselines(ds, cache_dir, ds in pix, args.workers)
        rows = []
        for ds in args.datasets:
            rows += run_dataset(ds, args.seeds, cache_dir, ds in pix)
        ts = io.now_iso()
        df = pd.DataFrame([dict(r, timestamp=ts) for r in rows])[cols]
        df.to_csv(csv, index=False)
        log(f"wrote {os.path.relpath(csv, paths.root())} ({len(df)} rows)")
    with Stage("report"):
        write_table(df, os.path.join(out, "table.md"))
        if "push" in set(df.dataset):
            write_headline(df, os.path.join(out, "headline.png"))
        res = write_gates(df, os.path.join(out, "gates.md"))
    log("gates: " + ", ".join(f"{g} {v}" for g, v in res))
    TIMING["total"] = round(time.time() - _T0, 1)
    with open(os.path.join(out, "timing.json"), "w") as f:
        json.dump(TIMING, f, indent=1)
    log(f"timing {json.dumps(TIMING)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
