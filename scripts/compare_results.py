#!/usr/bin/env python
"""Compare a fresh probes.csv against a reference copy, row by row.

  python scripts/compare_results.py REF.csv NEW.csv            # two files
  python scripts/compare_results.py --rev HEAD                 # results/<ds>/probes.csv vs `git show HEAD:...`
  python scripts/compare_results.py --rev v1.1 --datasets poscheck

Rows are matched on (inputs, target, probe, seed). The check fails when a key is missing on either
side, when n_train / n_val / n_test / feat_dim / pca_dim differ, or when |Δ r2| (and |Δ null_mean| for
perm rows) exceeds the tolerance. Every canonical probe is seeded (splits by seed, PCA random_state =
seed, shuffle default_rng(seed), perm default_rng(seed * 100003 + k)), so ridge, mean, shuffle and perm
rows are deterministic up to floating-point (BLAS) differences between machines: --tol 1e-6 applies
to all of them. The optional mlp probe depends on the optimiser's floating-point path and gets the
looser --loose-tol 0.02. git_commit and timestamp are ignored; perm p-values are reported, not gated.
Exit code 0 when every compared file passes, 1 otherwise, 2 on usage errors.
"""
import argparse
import io as _io
import os
import subprocess
import sys

import numpy as np
import pandas as pd

KEY = ["inputs", "target", "probe", "seed"]
EXACT = ["n_train", "n_val", "n_test", "feat_dim", "pca_dim"]
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def load(src, label):
    df = pd.read_csv(src)
    missing = [c for c in KEY + ["r2"] if c not in df.columns]
    if missing:
        sys.exit(f"{label}: no column(s) {missing}")
    dup = df.duplicated(KEY, keep=False)
    if dup.any():
        sys.exit(f"{label}: {int(dup.sum())} rows share a (inputs, target, probe, seed) key; "
                 "this is not a single canonical run (use probes.run --fresh)")
    return df.set_index(KEY)


def compare(ref, new, tol, loose_tol, loose_probes, label=""):
    """-> (ok, lines)."""
    lines, ok = [], True
    only_ref = ref.index.difference(new.index)
    only_new = new.index.difference(ref.index)
    if len(only_ref) or len(only_new):
        ok = False
        lines.append(f"FAIL keys: {len(only_ref)} rows only in the reference, {len(only_new)} only in the new file")
        for k in list(only_ref[:5]) + list(only_new[:5]):
            lines.append(f"     {'ref' if k in only_ref else 'new'} only: {k}")
    both = ref.index.intersection(new.index)
    r, n = ref.loc[both], new.loc[both]
    for c in EXACT:
        if c in r.columns and c in n.columns:
            a, b = r[c].to_numpy(float), n[c].to_numpy(float)
            bad = ~((a == b) | (np.isnan(a) & np.isnan(b)))
            if bad.any():
                ok = False
                lines.append(f"FAIL {c}: {int(bad.sum())} rows differ, e.g. {both[np.argmax(bad)]}")
    d = pd.DataFrame({"probe": both.get_level_values("probe"),
                      "dr2": (n["r2"].to_numpy(float) - r["r2"].to_numpy(float)),
                      "key": list(both)})
    nan_mismatch = r["r2"].isna().to_numpy() != n["r2"].isna().to_numpy()
    if nan_mismatch.any():
        ok = False
        lines.append(f"FAIL r2: {int(nan_mismatch.sum())} rows are NaN on one side only")
    lines.append(f"{'probe':8s} {'rows':>5s} {'max |Δr2|':>11s} {'tol':>8s}  worst row")
    for probe, g in d.groupby("probe", sort=True):
        t = loose_tol if probe in loose_probes else tol
        a = g["dr2"].abs()
        worst = float(a.max()) if a.notna().any() else 0.0
        verdict = worst <= t
        ok &= verdict
        where = g["key"].loc[a.idxmax()] if a.notna().any() else "-"
        lines.append(f"{probe:8s} {len(g):5d} {worst:11.3g} {t:8.0e}  {where if worst > 0 else '-'}"
                     f"{'' if verdict else '   <-- FAIL'}")
    perm = [k for k in both if k[2] == "perm"]
    if perm and {"null_mean", "p_value"} <= set(r.columns) & set(n.columns):
        dm = (n.loc[perm, "null_mean"] - r.loc[perm, "null_mean"]).abs().max()
        dp = (n.loc[perm, "p_value"] - r.loc[perm, "p_value"]).abs().max()
        if dm > tol:
            ok = False
        lines.append(f"perm null_mean max |Δ| {dm:.3g} (tol {tol:.0e}){'' if dm <= tol else '   <-- FAIL'}; "
                     f"p_value max |Δ| {dp:.3g} (reported only)")
    if "alpha" in r.columns and "alpha" in n.columns:
        a, b = r["alpha"].to_numpy(float), n["alpha"].to_numpy(float)
        fin = np.isfinite(a) & np.isfinite(b)
        rel = np.abs(a[fin] - b[fin]) / np.maximum(np.abs(a[fin]), 1e-300)
        lines.append(f"alpha: {int((rel > 1e-9).sum())} of {int(fin.sum())} rows chose another value (reported only)")
    lines.insert(0, f"{'PASS' if ok else 'FAIL'} {label} ({len(both)} common rows)")
    return ok, lines


def git_show(rev, path):
    out = subprocess.run(["git", "-C", ROOT, "show", f"{rev}:{path}"], capture_output=True, text=True)
    if out.returncode != 0:
        return None
    return _io.StringIO(out.stdout)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", metavar="CSV", help="REF.csv NEW.csv")
    ap.add_argument("--rev", default=None,
                    help="git revision of the reference: compares results/<ds>/probes.csv with `git show REV:...`")
    ap.add_argument("--datasets", nargs="+", default=["slide", "poscheck", "push"])
    ap.add_argument("--tol", type=float, default=1e-6, help="max |Δ r2| for ridge, mean, shuffle, perm (default 1e-6)")
    ap.add_argument("--loose-tol", type=float, default=0.02, help="max |Δ r2| for --loose-probes (default 0.02)")
    ap.add_argument("--loose-probes", nargs="*", default=["mlp"])
    a = ap.parse_args(argv)
    loose = set(a.loose_probes)
    pairs = []
    if a.files:
        if len(a.files) != 2 or a.rev:
            ap.error("give either REF.csv NEW.csv or --rev REV")
        pairs.append((a.files[0], a.files[1], f"{a.files[1]} vs {a.files[0]}"))
    else:
        if not a.rev:
            ap.error("give REF.csv NEW.csv, or --rev REV to compare results/<ds>/probes.csv with git")
        for ds in a.datasets:
            rel = f"results/{ds}/probes.csv"
            ref = git_show(a.rev, rel)
            new = os.path.join(os.environ.get("PHYSPROBE_ROOT") or ROOT, rel)
            if ref is None or not os.path.exists(new):
                print(f"SKIP {ds}: {'no ' + rel + ' at ' + a.rev if ref is None else 'no ' + new}")
                continue
            pairs.append((ref, new, f"{ds}: {new} vs {a.rev}:{rel}"))
    if not pairs:
        print("nothing compared")
        return 1
    all_ok = True
    for ref, new, label in pairs:
        ok, lines = compare(load(ref, "reference"), load(new, "new"), a.tol, a.loose_tol, loose, label)
        all_ok &= ok
        print("\n".join(lines))
        print()
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
