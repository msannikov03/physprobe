"""Linear (and optional MLP) probes on cached features -> results/<D>/probes.csv.

Usage:
  python -m physprobe.probes.run --dataset push [--targets log_m mu] [--reps r1 r2]
      [--modes m1 m2] [--inputs "vjepa2_l/clip_seq+contact/realistic" ...]
      [--seeds 0 1 2] [--probes ridge mean shuffle perm [mlp]] [--n-perm 100] [--no-preds]
      [--no-singles] [--fresh]

Output: rows are merged into an existing results/<D>/probes.csv by key (rep, mode, inputs, target,
probe, seed): new rows replace old ones, rows of inputs not rerun are KEPT. --fresh deletes
results/<D>/probes.csv and results/<D>/preds/*.npz first, so every row comes from this run
(use it for canonical reruns; report.py warns when probes.csv mixes git commits).

Protocol (CONTRACT.md, v1.2 "Probe protocol" with the v1.3 amendments):
  * rows filtered by meta `ok`; splits via io.split_ids(meta, seed), seeds {0,1,2};
  * StandardScaler fit on train -> PCA(256) on train if D > 2048 -> RidgeCV(logspace(-3,6,28), cv=5)
    on train only. Val is never used for fitting/selection here, only reported (r2_val) so that
    report.py can choose e.g. the V-JEPA layer on val. Test is never used for selection.
  * combo inputs "A+B" (A,B = "rep/mode"): each block standardised on train and divided by
    sqrt(D_block) (total variance 1 per block), then concatenated; no second per-column scaler
    (it would undo the block weighting).
  * perm: label-permutation null. alpha fixed to the real ridge fit's alpha; for k < n_perm the
    train labels are permuted with default_rng(seed*100003 + k) and ridge is refit in closed form
    from one SVD of the preprocessed train features. One row: r2 = 95th percentile of the null test
    R^2, rmse = NaN, p_value = (1 + #{null >= real ridge test R^2}) / (1 + n_perm), null_mean.
    No preds file.
Only numpy/scipy/sklearn/pandas (no torch).
"""
import argparse
import datetime as _dt
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from physprobe import io, paths

CONTRACT_COLS = ["dataset", "rep", "mode", "inputs", "target", "probe", "seed", "n_train", "n_val",
                 "n_test", "n_excluded", "feat_dim", "pca_dim", "alpha", "r2", "rmse", "git_commit",
                 "timestamp"]
EXTRA_COLS = ["r2_val", "r2_train", "oracle", "p_value", "null_mean"]
KEY_COLS = ["rep", "mode", "inputs", "target", "probe", "seed"]
ALPHAS = np.logspace(-3, 6, 28)
PCA_THRESHOLD = 2048
PCA_DIM = 256
MIN_TRAIN = 10
SPLITS = ("train", "val", "test")


def warn(msg):
    print(f"[warn] {msg}", file=sys.stderr, flush=True)


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------------- targets / splits
def target_values(meta, target):
    """{id: float} for rows where the target is defined; None if absent everywhere."""
    out = {}
    src = "m" if target == "log_m" else target
    for r in meta:
        v = r.get(src)
        if v is None:
            continue
        if isinstance(v, (list, tuple)):  # e.g. position vector -> x component
            if not v:
                continue
            v = v[0]
        v = float(v)
        if target == "log_m":
            if v <= 0:
                continue
            v = np.log(v)
        if np.isfinite(v):
            out[r["id"]] = v
    return out or None


def make_splits(meta, seed):
    """Split via io.split_ids, then drop ok=False rows. Returns (splits, n_excluded)."""
    sp = io.split_ids(meta, seed)
    ok = {r["id"]: bool(r.get("ok", True)) for r in meta}
    pool = [i for s in SPLITS for i in sp.get(s, [])]
    n_excl = sum(not ok.get(i, False) for i in pool)
    return {s: [i for i in sp.get(s, []) if ok.get(i, False)] for s in SPLITS}, n_excl


# ----------------------------------------------------------------------------- features
def discover(name):
    base = Path(paths.root()) / "features" / name
    if not base.exists():
        return []
    return sorted((p.parent.name, p.stem) for p in base.glob("*/*.npz")
                  if not p.name.startswith("."))   # io.save_features temp files: .tmp_*.npz


class Block:
    def __init__(self, name, rep, mode):
        ids, X, info = io.load_features(name, rep, mode)
        ids = [str(i) for i in np.asarray(ids)]
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]
        X = X.reshape(X.shape[0], -1)
        if len(ids) != X.shape[0]:
            raise ValueError(f"{rep}/{mode}: {len(ids)} ids vs X rows {X.shape[0]}")
        self.rep, self.mode, self.info, self.X = rep, mode, info or {}, X
        self.row = {i: k for k, i in enumerate(ids)}
        if len(self.row) != len(ids):
            warn(f"{rep}/{mode}: duplicate ids in features; last occurrence wins")

    @property
    def dim(self):
        return self.X.shape[1]

    def take(self, ids):
        return self.X[[self.row[i] for i in ids]].astype(np.float64)


_WARNED = set()


def impute_nans(Xs, label):
    """Replace NaN/inf with the train column mean (Xs = [train, val, test], in place)."""
    bad = [~np.isfinite(X) for X in Xs]
    if not any(b.any() for b in bad):
        return Xs
    if label not in _WARNED:
        _WARNED.add(label)
        warn(f"{label}: {sum(int(b.sum()) for b in bad)} non-finite values replaced with train "
             "column means (reported once)")
    tr = np.where(bad[0], np.nan, Xs[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(tr, axis=0) if tr.shape[0] else np.zeros(tr.shape[1])
    mu = np.where(np.isfinite(mu), mu, 0.0)
    for X, b in zip(Xs, bad):
        if b.any():
            X[b] = np.broadcast_to(mu, X.shape)[b]
    return Xs


def scale_blocks(parts):
    """parts: list of [train, val, test] per block -> concatenated [train, val, test]."""
    out = [[], [], []]
    for P in parts:
        sc = StandardScaler().fit(P[0])
        w = 1.0 / np.sqrt(P[0].shape[1])
        for k in range(3):
            out[k].append(sc.transform(P[k]) * w)
    return [np.hstack(o) for o in out]


# ----------------------------------------------------------------------------- probes
def metrics(y, p):
    if len(y) < 2:
        return np.nan, np.nan
    return float(r2_score(y, p)), float(np.sqrt(mean_squared_error(y, p)))


def preprocess(Xtr, Xva, Xte, seed, prescaled, pca_dim=None):
    """Scaler (unless prescaled) -> PCA if D > 2048 (or pca_dim forced). Returns X's, pca_dim."""
    if not prescaled:
        sc = StandardScaler().fit(Xtr)
        Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    D = Xtr.shape[1]
    if pca_dim is None and D > PCA_THRESHOLD:
        pca_dim = PCA_DIM
    used = 0
    if pca_dim:
        k = int(min(pca_dim, D, Xtr.shape[0]))
        pca = PCA(n_components=k, random_state=seed).fit(Xtr)
        Xtr, Xva, Xte = pca.transform(Xtr), pca.transform(Xva), pca.transform(Xte)
        used = k
    return Xtr, Xva, Xte, used


class _SVDRidge:
    """Equivalent of RidgeCV(alphas=ALPHAS, cv=5) (KFold(5) w/o shuffle, mean fold R^2, refit on
    all train), but with one SVD per fold for all alphas: ~10-100x faster than sklearn's
    GridSearchCV path and immune to BLAS oversubscription."""

    def __init__(self, alphas=ALPHAS, cv=5):
        self.alphas, self.cv = np.asarray(alphas, float), cv

    @staticmethod
    def _svd(X, y):
        xm, ym = X.mean(0), y.mean()
        U, s, Vt = np.linalg.svd(X - xm, full_matrices=False)
        return xm, ym, U, s, Vt, U.T @ (y - ym)

    def fit(self, X, y):
        n = X.shape[0]
        scores = np.zeros(len(self.alphas))
        for te in np.array_split(np.arange(n), self.cv):
            tr = np.setdiff1d(np.arange(n), te)
            xm, ym, U, s, Vt, Uty = self._svd(X[tr], y[tr])
            yte = y[te]
            ss = ((yte - yte.mean()) ** 2).sum()
            Z = (X[te] - xm) @ Vt.T                      # (n_te, k)
            for a_i, a in enumerate(self.alphas):
                p = Z @ (s / (s ** 2 + a) * Uty) + ym
                r = ((yte - p) ** 2).sum()
                scores[a_i] += (1 - r / ss) if ss > 0 else (1.0 if r == 0 else 0.0)
        self.alpha_ = float(self.alphas[int(np.argmax(scores))])
        self.best_score_ = float(scores.max() / self.cv)
        xm, ym, U, s, Vt, Uty = self._svd(X, y)
        self.coef_ = Vt.T @ (s / (s ** 2 + self.alpha_) * Uty)
        self.intercept_ = float(ym - xm @ self.coef_)
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_


def fit_ridge(Xtr, ytr):
    return _SVDRidge(ALPHAS, cv=5).fit(Xtr, ytr)


def perm_null(Ptr, ytr, Pte, yte, alpha, seed, n_perm):
    """Test R^2 of ridge(alpha) refit on n_perm permutations of ytr -> (n_perm,) array.

    One SVD of the centred train matrix serves all permutations (only y changes);
    the permuted mean equals ytr.mean(), so centring is shared too."""
    xm, ym, U, s, Vt, _ = _SVDRidge._svd(Ptr, ytr)
    Y = np.stack([np.random.default_rng(seed * 100003 + k).permutation(ytr)
                  for k in range(n_perm)], axis=1)                     # (n_train, n_perm)
    coef = Vt.T @ ((s / (s ** 2 + alpha))[:, None] * (U.T @ (Y - ym)))  # (D, n_perm)
    P = (Pte - xm) @ coef + ym                                         # (n_test, n_perm)
    ss = ((yte - yte.mean()) ** 2).sum()
    if len(yte) < 2 or ss == 0:
        return np.full(n_perm, np.nan)
    return 1 - ((yte[:, None] - P) ** 2).sum(0) / ss


def fit_mlp(Xtr, ytr, seed):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return MLPRegressor(hidden_layer_sizes=(256,), alpha=1e-3, early_stopping=True, max_iter=500,
                            random_state=seed).fit(Xtr, ytr)


def preds_name(rep, mode, probe, target, seed):
    s = f"{rep}__{mode}__{probe}__{target}__s{seed}"
    return s.replace("/", "-").replace("+", "_plus_") + ".npz"


# ----------------------------------------------------------------------------- main loop
def parse_inputs(specs):
    out = []
    for s in specs or []:
        s = s.replace(" ", "")
        parts = [p for p in s.split("+") if p]
        bad = [p for p in parts if p.count("/") != 1]
        if not parts or bad:
            raise SystemExit(f"bad --inputs '{s}': expected rep/mode[+rep/mode...]")
        out.append(("+".join(parts), [tuple(p.split("/")) for p in parts]))
    return out


def run(args):
    name = args.dataset
    meta = io.load_meta(name)
    log(f"dataset {name}: {len(meta)} meta rows, ok={sum(bool(r.get('ok', True)) for r in meta)}")
    commit, now = io.git_commit(), _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")

    targets = args.targets
    if targets is None:
        scen = {r.get("scenario") for r in meta}
        targets = ["x0_m"] if (name.startswith("poscheck") or scen == {"poscheck"}) else ["log_m", "mu"]
    tvals = {}
    for t in targets:
        if t not in ("log_m", "mu", "x0_m"):
            warn(f"unknown target {t}; trying meta key '{t}'")
        v = target_values(meta, t)
        if v is None:
            warn(f"target {t} absent in meta of {name}; skipped")
            continue
        if np.ptp(np.fromiter(v.values(), float)) == 0:
            warn(f"target {t} is constant in {name}; skipped")
            continue
        tvals[t] = v
    if not tvals:
        raise SystemExit("no usable targets")

    # inputs
    avail = discover(name)
    singles = []
    if not args.no_singles:
        singles = [(r, m) for r, m in avail
                   if (args.reps is None or r in args.reps) and (args.modes is None or m in args.modes)]
    inputs = [(f"{r}/{m}", [(r, m)]) for r, m in singles]
    for spec, blocks in parse_inputs(args.inputs):
        missing = [f"{r}/{m}" for r, m in blocks if (r, m) not in avail]
        if missing:
            warn(f"input {spec}: missing features {missing}; skipped")
            continue
        if spec not in [i[0] for i in inputs]:
            inputs.append((spec, blocks))
    log(f"inputs ({len(inputs)}): " + ", ".join(i[0] for i in inputs))
    log(f"targets: {list(tvals)}  seeds: {args.seeds}  probes: {args.probes}")

    splits = {s: make_splits(meta, s) for s in args.seeds}
    res_dir = Path(paths.results_dir(name))
    pred_dir = res_dir / "preds"
    if args.fresh:
        clear_previous(res_dir)
    if not args.no_preds:
        pred_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    def add_row(rep, mode, inp, target, probe, seed, n, n_excl, feat_dim, pca_dim, alpha, r2, rmse,
                r2_val, r2_train, oracle, p_value=np.nan, null_mean=np.nan):
        rows.append(dict(dataset=name, rep=rep, mode=mode, inputs=inp, target=target, probe=probe,
                         seed=seed, n_train=n[0], n_val=n[1], n_test=n[2], n_excluded=n_excl,
                         feat_dim=feat_dim, pca_dim=pca_dim, alpha=alpha, r2=r2, rmse=rmse,
                         git_commit=commit, timestamp=now, r2_val=r2_val, r2_train=r2_train,
                         oracle=bool(oracle), p_value=p_value, null_mean=null_mean))

    def emit(rep, mode, inp, target, probe, seed, n, n_excl, feat_dim, pca_dim, alpha,
             yte, pte, r2_val, r2_train, oracle, ids_te):
        r2, rmse = metrics(yte, pte)
        add_row(rep, mode, inp, target, probe, seed, n, n_excl, feat_dim, pca_dim, alpha, r2, rmse,
                r2_val, r2_train, oracle)
        if not args.no_preds:
            np.savez(pred_dir / preds_name(rep, mode, probe, target, seed),
                     ids=np.asarray(ids_te, dtype="<U16"), y_true=np.asarray(yte, np.float64),
                     y_pred=np.asarray(pte, np.float64))
        return r2

    def split_ids_for(seed, target, have=None):
        sp, n_excl = splits[seed]
        tv = tvals[target]
        out = {s: [i for i in sp[s] if i in tv and (have is None or i in have)] for s in SPLITS}
        return out, n_excl

    # mean probe: once per target/seed
    if "mean" in args.probes:
        for t in tvals:
            for seed in args.seeds:
                sp, n_excl = split_ids_for(seed, t)
                y = {s: np.array([tvals[t][i] for i in sp[s]]) for s in SPLITS}
                if len(y["train"]) < MIN_TRAIN:
                    warn(f"mean/{t}/s{seed}: only {len(y['train'])} train rows; skipped")
                    continue
                mu = y["train"].mean()
                r2v = metrics(y["val"], np.full(len(y["val"]), mu))[0]
                r2 = emit("mean", "-", "mean", t, "mean", seed, [len(y[s]) for s in SPLITS], n_excl,
                          0, 0, np.nan, y["test"], np.full(len(y["test"]), mu), r2v, 0.0, False,
                          sp["test"])
                log(f"  mean      {t:6s} s{seed}  r2={r2:+.3f}")

    feat_probes = [p for p in args.probes if p in ("ridge", "shuffle", "perm", "mlp")]
    for k_in, (spec, blocks_spec) in enumerate(inputs):
        if not feat_probes:
            break
        log(f"[{k_in + 1}/{len(inputs)}] {spec}")
        try:
            blocks = [Block(name, r, m) for r, m in blocks_spec]
        except Exception as e:  # noqa: BLE001
            warn(f"{spec}: failed to load ({e}); skipped")
            continue
        combo = len(blocks) > 1
        rep, mode = ("combo", spec) if combo else (blocks[0].rep, blocks[0].mode)
        oracle = any(bool(b.info.get("oracle")) for b in blocks)
        feat_dim = sum(b.dim for b in blocks)
        have = set.intersection(*(set(b.row) for b in blocks))
        for t in tvals:
            for seed in args.seeds:
                sp, n_excl = split_ids_for(seed, t, have)
                n_all = sum(len(splits[seed][0][s]) for s in SPLITS)
                n_here = sum(len(sp[s]) for s in SPLITS)
                if n_here < n_all and t == next(iter(tvals)):
                    warn(f"{spec} s{seed}: {n_all - n_here} ok ids without features/target dropped")
                n = [len(sp[s]) for s in SPLITS]
                if n[0] < MIN_TRAIN or n[2] == 0:
                    warn(f"{spec}/{t}/s{seed}: n_train={n[0]} n_test={n[2]}; skipped")
                    continue
                y = {s: np.array([tvals[t][i] for i in sp[s]]) for s in SPLITS}
                parts = [impute_nans([b.take(sp[s]) for s in SPLITS], f"{b.rep}/{b.mode}")
                         for b in blocks]
                if combo:
                    Xtr, Xva, Xte = scale_blocks(parts)
                else:
                    Xtr, Xva, Xte = parts[0]
                Ptr, Pva, Pte, pdim = preprocess(Xtr, Xva, Xte, seed, prescaled=combo)
                msg = []
                ridge = None
                if "ridge" in feat_probes:
                    ridge = m = fit_ridge(Ptr, y["train"])
                    r2 = emit(rep, mode, spec, t, "ridge", seed, n, n_excl, feat_dim, pdim,
                              float(m.alpha_), y["test"], m.predict(Pte),
                              metrics(y["val"], m.predict(Pva))[0],
                              metrics(y["train"], m.predict(Ptr))[0], oracle, sp["test"])
                    msg.append(f"ridge={r2:+.3f}(a={m.alpha_:.3g})")
                if "perm" in feat_probes and args.n_perm > 0:
                    if ridge is None:
                        ridge = fit_ridge(Ptr, y["train"])
                    real = metrics(y["test"], ridge.predict(Pte))[0]
                    null = perm_null(Ptr, y["train"], Pte, y["test"], ridge.alpha_, seed, args.n_perm)
                    p95 = float(np.percentile(null, 95))
                    pval = float((1 + np.sum(null >= real)) / (1 + args.n_perm))
                    add_row(rep, mode, spec, t, "perm", seed, n, n_excl, feat_dim, pdim,
                            float(ridge.alpha_), p95, np.nan, np.nan, np.nan, oracle,
                            p_value=pval, null_mean=float(np.mean(null)))
                    msg.append(f"perm_p95={p95:+.3f}(p={pval:.3f})")
                if "shuffle" in feat_probes:
                    ysh = np.random.default_rng(seed).permutation(y["train"])
                    m = fit_ridge(Ptr, ysh)
                    r2 = emit(rep, mode, spec, t, "shuffle", seed, n, n_excl, feat_dim, pdim,
                              float(m.alpha_), y["test"], m.predict(Pte),
                              metrics(y["val"], m.predict(Pva))[0],
                              metrics(ysh, m.predict(Ptr))[0], oracle, sp["test"])
                    msg.append(f"shuffle={r2:+.3f}")
                if "mlp" in feat_probes:
                    Mtr, Mva, Mte, mdim = preprocess(Xtr, Xva, Xte, seed, prescaled=combo,
                                                     pca_dim=PCA_DIM)
                    if mdim:  # one global rescale so mean PCA-score variance = 1 (keeps spectrum
                        # shape; per-component whitening would inflate noise directions)
                        g = np.sqrt(Mtr.var(0).mean()) or 1.0
                        Mtr, Mva, Mte = Mtr / g, Mva / g, Mte / g
                    m = fit_mlp(Mtr, y["train"], seed)
                    r2 = emit(rep, mode, spec, t, "mlp", seed, n, n_excl, feat_dim, mdim, 1e-3,
                              y["test"], m.predict(Mte), metrics(y["val"], m.predict(Mva))[0],
                              metrics(y["train"], m.predict(Mtr))[0], oracle, sp["test"])
                    msg.append(f"mlp={r2:+.3f}")
                log(f"    {t:6s} s{seed} n={n} D={feat_dim} pca={pdim}  " + "  ".join(msg))
        del blocks

    if not rows:
        warn("no rows produced")
        return None
    write_csv(res_dir / "probes.csv", pd.DataFrame(rows))
    log(f"wrote {len(rows)} rows -> {res_dir / 'probes.csv'}")
    return res_dir / "probes.csv"


def clear_previous(res_dir):
    """--fresh: delete results/<D>/probes.csv and results/<D>/preds/*.npz (nothing else)."""
    csv = Path(res_dir) / "probes.csv"
    if csv.exists():
        csv.unlink()
        log(f"--fresh: removed {csv}")
    pdir = Path(res_dir) / "preds"
    if pdir.is_dir():
        old = list(pdir.glob("*.npz"))
        for p in old:
            p.unlink()
        log(f"--fresh: removed {len(old)} files from {pdir}")


def write_csv(path, new):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = CONTRACT_COLS + EXTRA_COLS
    new = new[cols]
    if path.exists():
        old = pd.read_csv(path, dtype={"mode": str, "rep": str, "inputs": str})
        key = lambda d: d[KEY_COLS].astype(str).agg("|".join, axis=1)  # noqa: E731
        old = old[~key(old).isin(set(key(new)))]
        for c in cols:  # older probes.csv may lack newer extra columns (e.g. p_value, null_mean)
            if c not in old.columns:
                old[c] = np.nan
        extra = [c for c in old.columns if c not in cols]
        df = pd.concat([old, new], ignore_index=True)[cols + extra]
    else:
        df = new
    df = df.sort_values(["target", "rep", "mode", "inputs", "probe", "seed"], kind="stable")
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--targets", nargs="+", default=None,
                    help="default: log_m mu (x0_m for poscheck)")
    ap.add_argument("--reps", nargs="+", default=None)
    ap.add_argument("--modes", nargs="+", default=None)
    ap.add_argument("--inputs", nargs="+", default=None,
                    help='extra inputs, e.g. "vjepa2_l/clip_seq+contact/realistic"')
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--probes", nargs="+", default=["ridge", "mean", "shuffle", "perm"],
                    choices=["ridge", "mean", "shuffle", "perm", "mlp"])
    ap.add_argument("--n-perm", type=int, default=100,
                    help="permutations for the perm (label-permutation null) probe")
    ap.add_argument("--no-preds", action="store_true", help="do not write preds/*.npz")
    ap.add_argument("--no-singles", action="store_true",
                    help="run only --inputs, not every discovered rep/mode")
    ap.add_argument("--fresh", action="store_true",
                    help="delete results/<D>/probes.csv and preds/*.npz before running (no merge with "
                         "older rows); without it, rows of inputs not rerun are kept")
    ap.add_argument("--threads", type=int, default=None,
                    help="cap BLAS threads (helps a lot on a busy machine; e.g. 1-4)")
    args = ap.parse_args(argv)
    if args.threads:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=args.threads):
            return run(args)
    return run(args)


if __name__ == "__main__":
    main()
