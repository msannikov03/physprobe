"""End-to-end test of the B pipeline on synthetic dummy data (CPU only).

make_dummy (push + slide) -> baselines.build -> probes.run --fresh -> probes.report, then checks
(PLAN.md §4.3 B5 expectations, CONTRACT v1.2 gates):
  static R² ≈ 0; oracle_x R²(log m) > 0.9; contact/raw R²(log m) ≈ 1;
  slide: oracle R²(log m) < 0.05; unquantised oracle_xc >= quantised oracle_x on μ;
  label-permutation null p95 <= report.null_p95_bound(n_train) (n-aware) on pixels/clip_seq;
  mean predictor inside report.mean_r2_bounds(n_test, n_train);
  report.run_gates on push (with slide) and on slide: no FAIL in the deterministic gates (static,
  oracle, slide, contact_label). The statistical gates (mean, null) are printed as INFO there: at
  n = 100 (n_test = 15) each cell is a few-% tail event and all inputs share the split/permutation
  draws, so a no-FAIL assertion flips with the dummy draw (seen: n=100 slide pixels/last null p95
  0.408 vs 0.394; n=200 canonical-split mean R²(log m) −0.56 on seed 0, z = 2.7). Their n-aware
  bounds are asserted on the explicit push cells above instead;
  contact/realistic carries stop-time information only: R²(log m) < 0.7, and no column of the
  block is near-constant (0 < std < 1e-3) — such a column is what train z-scoring blows up;
  train/val/test ids disjoint for split seeds 0/1/2; scaler + PCA fitted on train only.
The plan-v1 single-shuffle rule (≤ 0.05) is printed as INFO only (retired by CONTRACT v1.2).
Unit tests (no pipeline): build.realistic_contact on a synthetic post-stop residual ∝ m
(the sim-v1 leak), build.noisy_contact, probes.run.clear_previous (--fresh).

Run:  python tests/test_pipeline.py [--n 100] [--regen]   (uses $PHYSPROBE_ROOT, else the clone root)
      pytest tests/test_pipeline.py   (n from $PHYSPROBE_TEST_N, default 100; runs in a temporary
      PHYSPROBE_ROOT unless $PHYSPROBE_TEST_ROOT is set)
"""
import argparse
import os
import subprocess
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from physprobe import io, paths  # noqa: E402
from physprobe.baselines import build  # noqa: E402
from physprobe.probes import report, run  # noqa: E402

PUSH, SLIDE = "dummy", "dummy_slide"
G = 9.81


def sh(*args):
    print("$", " ".join(args), flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    subprocess.run([sys.executable, "-m", *args], check=True, env=env)


def r2(df, rep, mode, target, probe="ridge", col="r2"):
    sel = df[(df.rep == rep) & (df["mode"] == mode) & (df.target == target) & (df.probe == probe)]
    assert len(sel), f"no {probe} rows for {rep}/{mode}/{target}"
    return sel[col].mean()


STAT_GATES = {"mean", "null"}   # tail-probability gates: INFO in the dummy-scale run_gates check


def _gate_checks(ds, slide=None):
    df = report.load_probes(ds)
    agg = report.aggregate(df)
    sl = None
    if slide:
        sdf = report.load_probes(slide)
        sl = (slide, sdf, report.aggregate(sdf))
    gates = report.run_gates(ds, df, agg, sl, None, report.dataset_kind(ds))
    out = []
    for key, name, status, _, info in gates:
        if info:
            out.append((f"{ds} gate {key} (plan v1, info): {status}", None, 0.0))
        elif key in STAT_GATES:
            out.append((f"{ds} gate {key} (statistical, info at dummy n): {status}", None, 0.0))
        else:
            out.append((f"{ds} gate {key}: {status}", status != "FAIL", 0.0))
    return out


def run_checks(n=100, regen=False):
    """Run the pipeline on dummy push + slide; -> list of (name, ok | None for info, value)."""
    for ds, extra in ((PUSH, []), (SLIDE, ["--scenario", "slide"])):
        if regen or not (Path(paths.dataset_dir(ds)) / "meta.jsonl").exists():
            sh("physprobe.tools.make_dummy", "--n", str(n), "--dataset", ds, *extra)
        sh("physprobe.io", "validate", ds)
        sh("physprobe.baselines.build", "--dataset", ds, "--noise-curve")
        sh("physprobe.probes.run", "--dataset", ds, "--targets", "log_m", "mu", "--fresh")
    sh("physprobe.probes.report", "--dataset", PUSH, "--slide-dataset", SLIDE)

    push = pd.read_csv(Path(paths.results_dir(PUSH)) / "probes.csv")
    slide = pd.read_csv(Path(paths.results_dir(SLIDE)) / "probes.csv")
    checks = []
    for rep in sorted(set(push[push["mode"] == "static"].rep)):
        v = r2(push, rep, "static", "log_m")
        checks.append((f"push {rep}/static R²(log m) < 0.05", v < 0.05, v))
    checks += [
        ("push state/oracle_x R²(log m) > 0.9", r2(push, "state", "oracle_x", "log_m") > 0.9,
         r2(push, "state", "oracle_x", "log_m")),
        ("push contact/raw R²(log m) > 0.98", r2(push, "contact", "raw", "log_m") > 0.98,
         r2(push, "contact", "raw", "log_m")),
        ("slide state/oracle_x R²(log m) < 0.05", r2(slide, "state", "oracle_x", "log_m") < 0.05,
         r2(slide, "state", "oracle_x", "log_m")),
    ]
    xc, xq = r2(push, "state", "oracle_xc", "mu"), r2(push, "state", "oracle_x", "mu")
    checks.append(("push state/oracle_xc R²(μ) ≥ state/oracle_x R²(μ) − 0.01", xc >= xq - 0.01, xc - xq))

    # chance control, CONTRACT v1.2: null p95 <= max(0.10, 3.3/sqrt(n_train)), null mean <= 0.02
    pm = push[(push.probe == "perm") & (push.rep == "pixels") & (push["mode"] == "clip_seq")]
    assert len(pm), "no perm rows for pixels/clip_seq"
    for t, g in pm.groupby("target"):
        bound = report.null_p95_bound(g.n_train.mean())
        checks.append((f"push perm null p95 ≤ {bound:.3f} (n_train={g.n_train.mean():.0f}) pixels/clip_seq {t}",
                       g.r2.mean() <= bound, g.r2.mean()))
        checks.append((f"push perm null mean ≤ 0.02 pixels/clip_seq {t}", g.null_mean.mean() <= 0.02,
                       g.null_mean.mean()))
    # mean predictor, CONTRACT v1.2 (n-aware): [3·E − 0.01, 0.01], E = −(1 + n_te/n_tr)/(n_te − 1)
    for t, g in push[push.probe == "mean"].groupby("target"):
        lo, hi = report.mean_r2_bounds(g.n_test.mean(), g.n_train.mean())
        checks.append((f"push mean R²({t}) ∈ [{lo:.3f}, {hi}]", lo <= g.r2.mean() <= hi, g.r2.mean()))
    sh_ = push[push.probe == "shuffle"]
    if len(sh_):
        worst = sh_.groupby(["rep", "mode", "target"]).r2.mean().max()
        checks.append(("push shuffle R² ≤ 0.05 (plan v1 rule, info only)", None, worst))
    checks += _gate_checks(PUSH, SLIDE) + _gate_checks(SLIDE)

    # contact/realistic: only stop-time information may survive (no solver-residual / scale leak)
    v = r2(push, "contact", "realistic", "log_m")
    checks.append(("push contact/realistic R²(log m) < 0.7 (stop time only)", v < 0.7, v))
    for mode in ["realistic"] + [f"noise_{p:03d}" for p in (0, 10, 30, 100)]:
        _, X, info = io.load_features(PUSH, "contact", mode)
        sd = X.astype(np.float64).std(0)
        tiny = int(((sd > 0) & (sd < 1e-3)).sum())
        checks.append((f"push contact/{mode}: no near-constant column (0 < std < 1e-3)", tiny == 0, tiny))
        checks.append((f"push contact/{mode}: not labelled oracle", not info.get("oracle"), 0.0))
    nz = [r2(push, "contact", f"noise_{p:03d}", "mu") for p in (0, 100)]
    checks.append(("push contact/noise_100 R²(μ) ≤ noise_000 R²(μ) + 0.05 (noise curve can degrade)",
                   nz[1] <= nz[0] + 0.05, nz[1] - nz[0]))

    # splits: train/val/test disjoint for seeds 0/1/2
    meta = io.load_meta(PUSH)
    for s in (0, 1, 2):
        sp, _ = run.make_splits(meta, s)
        sets = [set(sp[k]) for k in run.SPLITS]
        disjoint = all(not (sets[a] & sets[b]) for a in range(3) for b in range(a + 1, 3))
        checks.append((f"split seed {s}: train/val/test disjoint and non-empty",
                       disjoint and all(sets), float(sum(map(len, sets)))))
    return checks


def print_checks(checks):
    fails = 0
    gated = [c for c in checks if c[1] is not None]
    for name, ok, val in checks:
        tag = "INFO" if ok is None else ("PASS" if ok else "FAIL")
        print(f"{tag}  {name}  ({val:+.3f})")
        fails += ok is False
    print(f"{len(gated) - fails}/{len(gated)} checks passed")
    return fails


def test_pipeline(tmp_path_factory, monkeypatch):
    root = os.environ.get("PHYSPROBE_TEST_ROOT") or str(tmp_path_factory.mktemp("physprobe_root"))
    monkeypatch.setenv("PHYSPROBE_ROOT", root)
    monkeypatch.setenv("MPLCONFIGDIR", os.path.join(root, ".cache", "mpl"))
    checks = run_checks(int(os.environ.get("PHYSPROBE_TEST_N", "100")))
    print_checks(checks)
    bad = [f"{name} ({val:+.3f})" for name, ok, val in checks if ok is False]
    assert not bad, "failed: " + "; ".join(bad)


# ----------------------------------------------------------------------------- unit tests

def _sim_like_contact(m, mu, J=1.0, resid=1.1e-9):
    """sim-v1-like (64,3) contact: ft = mu m g while sliding, then a solver residual ∝ m."""
    t = np.arange(64) / 64
    v0 = J / m
    moving = t < v0 / (mu * G)
    c = np.zeros((64, 3))
    c[:, 0] = m * G
    c[:, 1] = np.where(moving, mu * m * G, resid * m * G)
    c[:, 2] = np.clip(v0 - mu * G * t, 0, None)
    return c


def test_realistic_contact_floor():
    rng = np.random.default_rng(0)
    ms = np.exp(rng.uniform(np.log(0.5), np.log(2.0), 300))
    mus = rng.uniform(0.2, 0.8, 300)
    clips = [_sim_like_contact(m, u) for m, u in zip(ms, mus)]
    # noise-free: post-stop residual must be exactly zero, fn/max == 1, slip is a 0/1 flag
    base = np.stack([build.realistic_contact(c, 0.0, np.random.default_rng(k)) for k, c in enumerate(clips)])
    stopped = np.stack([c[:, 2] == 0 for c in clips])
    assert stopped.any(axis=1).all()
    assert np.all(base[:, :, 1][stopped] == 0.0), "post-stop ft residual survived the floor"
    assert np.allclose(base[:, :, 0], 1.0)
    assert set(np.unique(base[:, :, 2])) <= {0.0, 1.0}
    # with noise: the last frame (stopped in every clip) carries no μ information
    X = np.stack([build.realistic_contact(c, build.REALISTIC_NOISE, np.random.default_rng(k))
                  for k, c in enumerate(clips)])
    assert abs(np.corrcoef(X[:, -1, 1], mus)[0, 1]) < 0.2
    sd = X.reshape(len(X), -1).std(0)
    assert not ((sd > 0) & (sd < 1e-3)).any(), "near-constant column (z-scoring would amplify it)"
    # the multiplicative-noise version leaks: sanity-check that the test would catch it
    mx = np.array([c[:, 1].max() for c in clips])
    leak = np.array([c[-1, 1] for c in clips]) / mx
    assert abs(np.corrcoef(np.log(leak), np.log(mus))[0, 1]) > 0.9


def test_noisy_contact():
    c = _sim_like_contact(1.0, 0.5)
    n0 = build.noisy_contact(c, 0.0, np.random.default_rng(3))
    assert np.array_equal(n0, build.realistic_contact(c, 0.0, np.random.default_rng(3)))
    assert set(np.unique(n0[:, 2])) <= {0.0, 1.0}, "raw slip speed must not be included"
    n1 = build.noisy_contact(c, 1.0, np.random.default_rng(3))
    assert 0.6 < (n1 - n0).std() < 1.4


def test_preprocess_train_only():
    rng = np.random.default_rng(0)
    Xtr, Xva, Xte = rng.normal(size=(60, 2100)), rng.normal(size=(20, 2100)), rng.normal(size=(20, 2100))
    a = run.preprocess(Xtr, Xva, Xte, 0, prescaled=False)
    b = run.preprocess(Xtr, Xva * 50 + 7, Xte * 50 + 7, 0, prescaled=False)
    assert a[3] == b[3] > 0                     # PCA path taken
    assert np.allclose(a[0], b[0]), "train features depend on val/test (scaler or PCA not train-only)"


def test_fresh_clears_previous(tmp_path):
    (tmp_path / "preds").mkdir()
    (tmp_path / "probes.csv").write_text("x\n")
    (tmp_path / "preds" / "a__b__ridge__mu__s0.npz").write_bytes(b"")
    (tmp_path / "table.md").write_text("keep\n")
    run.clear_previous(tmp_path)
    assert not (tmp_path / "probes.csv").exists()
    assert not list((tmp_path / "preds").glob("*.npz"))
    assert (tmp_path / "table.md").exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--regen", action="store_true")
    a = ap.parse_args()
    sys.exit(1 if print_checks(run_checks(a.n, a.regen)) else 0)


if __name__ == "__main__":
    main()
