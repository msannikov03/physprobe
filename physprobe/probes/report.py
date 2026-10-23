"""Report over results/<D>/probes.csv -> table.md, gates.md, figs/*.png.

    python -m physprobe.probes.report --dataset push [--slide-dataset slide] [--poscheck-dataset poscheck]
    python -m physprobe.probes.report --all     # slide, poscheck, then push (each with a probes.csv)

Everything is optional: any subset of representations / modes / datasets may be
present; missing pieces are reported as "n/a" (tables) or skipped (figures).

Conventions:
  * Aggregation: mean +- std of test R^2 over seeds (rows deduplicated by
    (rep, mode, probe, target, seed), latest timestamp wins).
  * Mode selection (best V-JEPA layer, best clip pooling per representation; candidates
    clip_mean*, clip_seq*, clip_grid<k>*) uses mean `r2_val` averaged over the available
    targets — never test R^2. If `r2_val` is missing, a fixed preference order is used
    instead and this is stated in the output.
  * Gates use the seed-mean test R^2.
  * State oracle: `state/oracle_xc` (unquantised x(t)) is the linear x(t) read-out
    reference: a ceiling for μ, not for log m (video inputs beat it on log m). It is used
    for the "frac" column (frac > 1 + 1e-3 printed as "n/a (>1)"; ties print 1.00), the oracle gate and the
    headline bar when present; `state/oracle_x` (1-px quantised) is informational only.
    Falls back to `oracle_x` for older feature sets.
  * probes.csv rows from more than one git commit trigger a loud warning at the top of
    table.md and gates.md (rerun with `probes.run --fresh`).
  * Chance control: the label-permutation null (`perm` rows: r2 = null p95, null_mean,
    p_value) when present, else the single shuffled-label fit (`shuffle` rows).
  * Gate scope: the static gate needs log_m/mu targets; the state-oracle gate runs on
    push-type datasets only (dataset.json scenario / dummy_kind, else the name). The plan v1
    rules (the original gate rules, recorded in CONTRACT.md v1.2: state/oracle_x R²(log m) > 0.95
    and R²(μ) > 0.9; shuffle R² ≤ 0.05) are printed
    next to the new ones as "plan v1 rule, informational" and never change the summary.
  * Cross-dataset gates read results/slide and results/poscheck by default (dummy:
    results/dummy_slide); missing -> N/A. `--all` re-renders push after slide/poscheck.
"""
import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from physprobe.paths import results_dir as _results_dir


def results_dir(ds):
    return Path(_results_dir(ds))


TARGETS = ["log_m", "mu", "x0_m"]
TARGET_LABEL = {"log_m": "log m", "mu": "μ", "x0_m": "x0"}
TARGET_LABEL_EN = {"log_m": "log m", "mu": "μ (friction)", "x0_m": "x0 (m)"}

REP_ORDER = ["mean", "pixels", "track", "dinov2_b_random", "dinov2_b", "dinov2_l", "siglip2_b",
             "vjepa2_l_random", "vjepa2_l", "cosmos_cv8"]
TAIL_REPS = ["combo", "state", "contact"]  # after "other" reps
REP_LABEL = {
    "mean": "Mean pred.", "pixels": "Pixels", "track": "Track (centroid)",
    "dinov2_b_random": "DINOv2-B rand", "dinov2_b": "DINOv2-B", "dinov2_l": "DINOv2-L",
    "siglip2_b": "SigLIP2-B", "vjepa2_l_random": "V-JEPA2-L rand", "vjepa2_l": "V-JEPA2-L",
    "cosmos_cv8": "Cosmos CV8", "state": "State oracle", "contact": "Contact oracle",
    "fake_enc": "fake_enc",
}
ORACLE_MODES = {("contact", "raw")}
NON_VIDEO_REPS = {"mean", "state", "contact"}
CLIP_MODE_RE = r"^clip_(mean|seq|grid\d+)(_L\d+)?$"   # headline / ★ candidates (chosen on val)

# colorblind-safe categorical slots (validated reference palette), text inks
C_TARGET = {"log_m": "#2a78d6", "mu": "#eb6834", "x0_m": "#1baf7a"}
C_SLIDE = "#6250d6"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e4e3df"


# ----------------------------------------------------------------------------- data

def mode_key(m):
    """Sort key for modes: static, last, clip_mean_L08.., clip_mean, clip_seq.., others."""
    m = str(m)
    base = ["-", "static", "last", "clip", "clip_mean", "clip_seq", "clip_grid", "oracle_xc",
            "oracle_x", "oracle_fit",
            "raw", "realistic"]
    mm = re.match(r"^(clip_mean|clip_seq|clip_grid)(\d*)(?:_L(\d+))?$", m)
    if mm:
        return (base.index(mm.group(1)), int(mm.group(3)) if mm.group(3) else 99, m)
    if m.startswith("noise"):
        return (len(base), 0, m)
    return (base.index(m) if m in base else len(base) + 1, 0, m)


def rep_key(r):
    if r in REP_ORDER:
        return (0, REP_ORDER.index(r))
    if r in TAIL_REPS:
        return (2, TAIL_REPS.index(r))
    return (1, r)


def rep_label(rep):
    return REP_LABEL.get(rep, rep)


def _bool(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in ("true", "1", "yes")


def load_probes(ds):
    if not ds:
        return None
    p = results_dir(ds) / "probes.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, dtype={"mode": str, "rep": str, "inputs": str})
    if df.empty:
        return None
    for c in ("r2_val", "r2_train", "oracle", "timestamp", "git_commit", "n_train", "n_val",
              "n_test", "n_excluded", "p_value", "null_mean"):
        if c not in df.columns:
            df[c] = np.nan
    df["mode"] = df["mode"].fillna("-").astype(str)
    df["oracle"] = df["oracle"].map(_bool)
    df.loc[[(r, m) in ORACLE_MODES for r, m in zip(df["rep"], df["mode"])], "oracle"] = True
    df = df.sort_values("timestamp", kind="stable")
    df = df.drop_duplicates(["rep", "mode", "probe", "target", "seed"], keep="last")
    return df


def mean_r2_expected(n_test, n_train=None):
    """E[test R²] of the train-mean predictor = −(1 + n_test/n_train)/(n_test − 3) for Gaussian y.

    R² = −n_te·(ȳ_te − ȳ_tr)²/SS_te. For Gaussian y, ȳ_te is independent of SS_te ~ σ²χ²(n_te − 1),
    Var(ȳ_te − ȳ_tr) = σ²(1/n_te + 1/n_tr) and E[1/χ²_k] = 1/(k − 2), hence the n_te − 3.
    Without n_train: −1/(n_test − 3)."""
    def ok(v):
        return v is not None and not (isinstance(v, float) and math.isnan(v)) and v > 0
    if not ok(n_test) or n_test <= 3:
        return float("nan")
    ratio = n_test / n_train if ok(n_train) else 0.0
    return -(1.0 + ratio) / (n_test - 3.0)


def mean_r2_bounds(n_test, n_train=None):
    """Mean-gate interval [3·E − 0.01, 0.01], E = mean_r2_expected (−0.03 if E is unknown).
    R²/E is approximately χ²₁, so a single seed leaves it with p ≈ P(χ²₁ > 3) ≈ 0.08. For the 3-seed
    mean, p ≈ 0.03 is approximate and optimistic: seeds 1 and 2 reshuffle the same clips as seed 0,
    so the three splits are correlated, not independent."""
    e = mean_r2_expected(n_test, n_train)
    return (3.0 * e - 0.01 if not math.isnan(e) else -0.03), 0.01


def commit_warning(df):
    """Loud markdown warning when the (deduplicated) probes.csv rows come from > 1 git commit."""
    if "git_commit" not in df.columns:
        return None
    vc = df["git_commit"].dropna().astype(str).value_counts()
    if len(vc) <= 1:
        return None
    parts = ", ".join(f"`{c}` ({k} rows)" for c, k in vc.items())
    return (f"> ⚠️ **WARNING: probes.csv mixes {len(vc)} git commits: {parts}.** Rows of inputs that "
            "were not rerun keep older numbers, so this report is NOT a reproduction at one commit. "
            "Rerun with `python -m physprobe.probes.run --dataset <ds> --fresh` "
            "(scripts/probe_all.sh does this).")


def null_p95_bound(n_train):
    """Upper bound for the permutation-null p95 R² (see the null gate in run_gates)."""
    if n_train is None or math.isnan(n_train) or n_train <= 0:
        return 0.10
    return max(0.10, 3.3 / math.sqrt(n_train))


def aggregate(df):
    """-> DataFrame indexed by (probe, rep, mode, target) with r2/rmse/r2_val stats."""
    g = df.groupby(["probe", "rep", "mode", "target"])
    a = g.agg(r2=("r2", "mean"), r2_std=("r2", "std"), rmse=("rmse", "mean"),
              r2_val=("r2_val", "mean"), r2_train=("r2_train", "mean"),
              n_seeds=("seed", "nunique"), oracle=("oracle", "any"),
              n_test=("n_test", "mean"), n_train=("n_train", "mean"), p_value=("p_value", "mean"),
              p_value_max=("p_value", "max"), null_mean=("null_mean", "mean"))
    return a


def get(agg, probe, rep, mode, target, col="r2"):
    try:
        v = agg.loc[(probe, rep, mode, target), col]
    except KeyError:
        return float("nan")
    return float(v)


def state_oracle_mode(agg):
    """'oracle_xc' (unquantised, true upper bound) if present, else 'oracle_x'."""
    return "oracle_xc" if "oracle_xc" in ridge_modes(agg, "state") else "oracle_x"


QUANT_ORACLE_NOTE = "pixel-quantised oracle (not an upper bound)"


def present_targets(df):
    return [t for t in TARGETS if t in set(df["target"])]


def ridge_modes(agg, rep):
    try:
        sub = agg.loc["ridge"].loc[rep]
    except KeyError:
        return []
    return sorted(set(sub.index.get_level_values(0)), key=mode_key)


def val_score(agg, rep, mode, targets):
    vals = [get(agg, "ridge", rep, mode, t, "r2_val") for t in targets]
    vals = [v for v in vals if not math.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def pick_by_val(agg, rep, candidates, targets, prefer=()):
    """Best of `candidates` by mean r2_val. Returns (mode, how) or (None, None)."""
    cands = [m for m in candidates if m in ridge_modes(agg, rep)]
    if not cands:
        return None, None
    scored = [(val_score(agg, rep, m, targets), m) for m in cands]
    scored_ok = [s for s in scored if not math.isnan(s[0])]
    if scored_ok:
        best = max(scored_ok)[1]
        return best, "val"
    for p in prefer:
        if p in cands:
            return p, "default (no r2_val)"
    return cands[-1], "default (no r2_val)"


def clip_mode_for(agg, rep, targets):
    """Headline 'clip' mode of a representation."""
    modes = ridge_modes(agg, rep)
    if rep == "pixels":
        return ("clip_seq", "fixed") if "clip_seq" in modes else (None, None)
    if rep == "state":
        om = state_oracle_mode(agg)
        return (om, "fixed") if om in modes else (None, None)
    if rep == "contact":
        return ("raw", "fixed") if "raw" in modes else (None, None)
    cands = [m for m in modes if re.match(CLIP_MODE_RE, m)]
    if not cands:
        cands = [m for m in modes if m.startswith("clip")]
    if not cands:
        cands = [m for m in modes if m not in ("static", "last")]
    return pick_by_val(agg, rep, cands, targets, prefer=("clip_seq", "clip_mean", "clip"))


def fmt(v, nd=3):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def fmt_pm(m, s):
    if math.isnan(m):
        return "n/a"
    return f"{m:.3f} ± {s:.3f}" if not math.isnan(s) else f"{m:.3f}"


def header_info(df):
    lines = []
    sub = df[df["probe"] == "ridge"]
    if sub.empty:
        sub = df
    parts = []
    for s, g in sub.groupby("seed"):
        def u(c):
            vals = sorted({int(x) for x in g[c].dropna()})
            return "/".join(map(str, vals)) if vals else "n/a"
        parts.append(f"s{s}: n_train={u('n_train')}, n_val={u('n_val')}, n_test={u('n_test')}, "
                     f"n_excluded={u('n_excluded')}")
    lines.append("Splits — " + "; ".join(parts))
    commits = sorted({str(c) for c in df["git_commit"].dropna()})
    lines.append("git commit: " + (", ".join(f"`{c}`" for c in commits) if commits else "n/a"))
    ts = df["timestamp"].dropna().astype(str)
    if len(ts):
        lines.append(f"runs: {ts.min()} … {ts.max()}")
    return lines


# ----------------------------------------------------------------------------- table.md

def build_table(ds, df, agg, slide, gate_summary):
    targets = present_targets(df)
    omode = state_oracle_mode(agg)
    out = [f"# PhysProbe results: `{ds}`", ""]
    warn_ = commit_warning(df)
    if warn_:
        out += [warn_, ""]
    out += [f"- {l}" for l in header_info(df)]
    out += ["- R² = test R², mean ± std over seeds (ridge probe unless stated). "
            "RMSE for log m is log-RMSE.",
            f"- Δ = R² − R²(pixels/clip_seq); frac = R² / R²(state/{omode}) (per target); "
            "a reference that is beaten (frac > 1.001) is printed as `n/a (>1)`, a tie as 1.00"
            + ("; `state/oracle_xc` = unquantised x(t), the linear x(t) read-out reference: a "
               "ceiling for μ, not for log m (a linear probe on x(t) does not invert "
               "d = J²/(2m²μg), so frac for log m is not a fraction of a ceiling); "
               "`state/oracle_x` (1-px quantised) is shown for information only."
               if omode == "oracle_xc" else "."),
            "- ★ = mode chosen by validation R² (mean r2_val over targets; candidates clip_mean*, "
            "clip_seq*, clip_grid*), used in headline.png.",
            ""]
    if gate_summary:
        out += [f"**Gates:** {gate_summary} (details in `gates.md`)", ""]

    # selections
    sel = {}
    sel_notes = []
    for rep in sorted({r for r in df.loc[df.probe == "ridge", "rep"]}, key=rep_key):
        m, how = clip_mode_for(agg, rep, targets)
        if m:
            sel[rep] = m
            if how not in ("fixed",):
                sel_notes.append(f"`{rep}` → `{m}` ({how})")
    for rep in ("vjepa2_l", "vjepa2_l_random"):
        for pool in ("clip_mean", "clip_seq"):
            cands = [x for x in ridge_modes(agg, rep) if re.match(rf"^{pool}(_L\d+)?$", x)]
            if len(cands) > 1:
                b, how = pick_by_val(agg, rep, cands, targets, prefer=(pool,))
                lay = re.search(r"_L(\d+)$", b)
                sel_notes.append(f"best `{rep}` layer for {pool}: "
                                 f"{'L' + lay.group(1) if lay else 'L24 (last)'} (`{b}`, by {how})")
    if sel_notes:
        out += ["**Mode / layer selection (by val, not test):** " + "; ".join(sel_notes), ""]

    ref = {t: get(agg, "ridge", "pixels", "clip_seq", t) for t in targets}
    orc = {t: get(agg, "ridge", "state", omode, t) for t in targets}

    cols = ["Representation", "Mode"]
    for t in targets:
        cols += [f"{TARGET_LABEL[t]} R²", f"{TARGET_LABEL[t]} RMSE"]
    cols += [f"Δ vs pixels/clip_seq ({TARGET_LABEL[t]})" for t in targets]
    cols += [f"frac of state/{omode} ({TARGET_LABEL[t]})" for t in targets]
    cols += ["seeds", "note"]
    out += ["## Main table (ridge)", "", "| " + " | ".join(cols) + " |",
            "|" + "|".join(["---"] * len(cols)) + "|"]

    def row(probe, rep, mode, label_rep, note_extra=""):
        cells = [label_rep, f"`{mode}`" if mode != "-" else "-"]
        for t in targets:
            cells += [fmt_pm(get(agg, probe, rep, mode, t), get(agg, probe, rep, mode, t, "r2_std")),
                      fmt(get(agg, probe, rep, mode, t, "rmse"))]
        for t in targets:
            r = get(agg, probe, rep, mode, t)
            d = r - ref[t]
            cells.append("n/a" if math.isnan(d) else f"{d:+.3f}")
        for t in targets:
            r = get(agg, probe, rep, mode, t)
            f = r / orc[t] if (not math.isnan(orc[t]) and orc[t] > 0) else float("nan")
            cells.append("n/a (>1)" if f > 1.0 + 1e-3 else fmt(min(f, 1.0), 2))
        ns = [get(agg, probe, rep, mode, t, "n_seeds") for t in targets]
        ns = [int(n) for n in ns if not math.isnan(n)]
        cells.append(str(max(ns)) if ns else "0")
        notes = []
        orc_flag = any(_bool(get(agg, probe, rep, mode, t, "oracle") == 1.0) for t in targets)
        if (rep, mode) in ORACLE_MODES or orc_flag:
            notes.append("**ORACLE**")
        elif rep == "state":
            notes.append("oracle (state)")
        if rep == "state" and mode == "oracle_x" and omode == "oracle_xc":
            notes.append(QUANT_ORACLE_NOTE)
        if sel.get(rep) == mode and probe == "ridge":
            notes.append("★")
        if note_extra:
            notes.append(note_extra)
        cells.append(" ".join(notes))
        return "| " + " | ".join(cells) + " |"

    if ("mean", "mean", "-") in {(p, r, m) for p, r, m, _ in agg.index}:
        out.append(row("mean", "mean", "-", "mean (predict train mean)"))
    else:
        mean_rows = [(r, m) for p, r, m, _ in agg.index if p == "mean"]
        for r, m in sorted(set(mean_rows)):
            out.append(row("mean", r, m, "mean (predict train mean)"))
    ridge_rm = sorted({(r, m) for p, r, m, _ in agg.index if p == "ridge"},
                      key=lambda x: (rep_key(x[0]), mode_key(x[1])))
    for r, m in ridge_rm:
        out.append(row("ridge", r, m, r))
    shuf_rm = sorted({(r, m) for p, r, m, _ in agg.index if p == "shuffle"},
                     key=lambda x: (rep_key(x[0]), mode_key(x[1])))
    for r, m in shuf_rm:
        out.append(row("shuffle", r, m, f"{r} [shuffled labels]", "control"))
    out.append("")

    # MLP (optional)
    mlp_rm = sorted({(r, m) for p, r, m, _ in agg.index if p == "mlp"},
                    key=lambda x: (rep_key(x[0]), mode_key(x[1])))
    if mlp_rm:
        cols = ["Representation", "Mode"] + sum(
            [[f"{TARGET_LABEL[t]} R² test", f"{TARGET_LABEL[t]} R² train"] for t in targets], [])
        out += ["## MLP probe (upper bound for a small nonlinear probe)", "",
                "| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
        for r, m in mlp_rm:
            cells = [r, f"`{m}`"]
            for t in targets:
                cells += [fmt_pm(get(agg, "mlp", r, m, t), get(agg, "mlp", r, m, t, "r2_std")),
                          fmt(get(agg, "mlp", r, m, t, "r2_train"))]
            out.append("| " + " | ".join(cells) + " |")
        out.append("")

    # slide
    out += ["## Identifiability check: slide (log m should be ≈ 0)", ""]
    if slide is None:
        out += ["n/a (no slide probes.csv)", ""]
    else:
        sname, sdf, sagg = slide
        out += [f"Dataset `{sname}`. " + " ".join(header_info(sdf)[:2]), "",
                "| Representation | Mode | probe | log m R² | log m RMSE | note |",
                "|---|---|---|---|---|---|"]
        rows = sorted({(p, r, m) for p, r, m, t in sagg.index
                       if t == "log_m" and p not in ("mlp", "perm")},
                      key=lambda x: (["mean", "ridge", "shuffle"].index(x[0])
                                     if x[0] in ("mean", "ridge", "shuffle") else 9,
                                     rep_key(x[1]), mode_key(x[2])))
        for p, r, m in rows:
            note = ""
            if (r == "contact" and m == "raw") or "contact/raw" in m:
                note = "leaks m via N=mg (by construction)"
            elif r == "state":
                note = "state oracle (reported separately)"
            if (r, m) in ORACLE_MODES:
                note = "**ORACLE**; " + note
            out.append(f"| {r} | `{m}` | {p} | "
                       f"{fmt_pm(get(sagg, p, r, m, 'log_m'), get(sagg, p, r, m, 'log_m', 'r2_std'))} | "
                       f"{fmt(get(sagg, p, r, m, 'log_m', 'rmse'))} | {note} |")
        out.append("")
    return "\n".join(out) + "\n", sel


# ----------------------------------------------------------------------------- gates

def dataset_kind(ds):
    """'push' | 'slide' | 'poscheck' from dataset.json (dummy_kind, then scenario), else the name."""
    try:
        from physprobe.io import load_dataset_json
        dj = load_dataset_json(ds)
        for k in (dj.get("dummy_kind"), dj.get("scenario")):
            if k in ("push", "slide", "poscheck"):
                return k
    except Exception:  # noqa: BLE001  (no data/ on this machine: fall back to the name)
        pass
    n = str(ds).lower()
    if "poscheck" in n:
        return "poscheck"
    if "slide" in n:
        return "slide"
    return "push"


def companions(ds):
    """Default (slide, poscheck) datasets for the cross-dataset gates."""
    if str(ds).startswith("dummy"):
        return "dummy_slide", None
    return "slide", "poscheck"


def run_gates(ds, df, agg, slide, poscheck, kind="push"):
    """-> list of (key, name, status, lines, info); info gates are plan v1 rules, not summarised."""
    gates = []
    targets = [t for t in ("log_m", "mu") if t in set(df["target"])] or present_targets(df)
    lm_mu = [t for t in ("log_m", "mu") if t in set(df["target"])]

    # mean: E[test R² of the train-mean predictor] = −(1 + n_test/n_train)/(n_test − 3) (Gaussian y)
    lines, st = [], []
    mean_keys = sorted({(r, m) for p, r, m, _ in agg.index if p == "mean"})
    for r, m in mean_keys:
        for t in targets:
            v = get(agg, "mean", r, m, t)
            if math.isnan(v):
                continue
            nt = get(agg, "mean", r, m, t, "n_test")
            ntr = get(agg, "mean", r, m, t, "n_train")
            e = mean_r2_expected(nt, ntr)
            lo, hi = mean_r2_bounds(nt, ntr)
            ok = lo <= v <= hi
            st.append(ok)
            lines.append(f"- {t}: R² = {v:.4f}, n_test = {fmt(nt, 0)}, n_train = {fmt(ntr, 0)}, "
                         f"bound [{lo:.4f}, {hi}] (E ≈ {fmt(e, 4)}) → {'ok' if ok else 'out of bound'}")
    if lines:
        lines.append("- bound: R² ∈ [3·E − 0.01, 0.01] with E = −(1 + n_test/n_train)/(n_test − 3), the "
                     "expected test R² of the train-mean predictor for Gaussian y (R² = −n_test·(ȳ_test − "
                     "ȳ_train)²/SS_test, so R²/E ≈ χ²₁: one seed leaves the bound with approximate p ≈ 0.08; "
                     "the 3-seed mean with approximate p ≈ 0.03, optimistic because the splits reshuffle "
                     "the same clips)")
    gates.append(("mean", "mean R² ∈ [3·E − 0.01, 0.01], E = −(1 + n_test/n_train)/(n_test − 3)",
                  _status(st), lines, False))

    # static (frame 0): log_m / mu only; on poscheck frame 0 shows x0 by design
    lines, st = [], []
    if lm_mu:
        for r, m in sorted({(r, m) for p, r, m, _ in agg.index if p == "ridge" and m == "static"},
                           key=lambda x: rep_key(x[0])):
            for t in lm_mu:
                v = get(agg, "ridge", r, m, t)
                if math.isnan(v):
                    continue
                ok = v < 0.05
                st.append(ok)
                lines.append(f"- {r}/static {t}: R² = {v:.4f}{'' if ok else '  ← FAIL'}")
    else:
        lines.append(f"- not applicable: no log_m/mu targets in `{ds}` (frame 0 encodes x0 by design)")
    gates.append(("static", "static (frame 0) R² < 0.05 for all reps (log m, μ)", _status(st), lines,
                  False))

    # chance control: label-permutation null if available, else single shuffle
    lines, st = [], []
    perm_rm = sorted({(r, m) for p, r, m, _ in agg.index if p == "perm"},
                     key=lambda x: (rep_key(x[0]), mode_key(x[1])))
    if perm_rm:
        bad = []
        for r, m in perm_rm:
            for t in targets:
                p95 = get(agg, "perm", r, m, t)
                nm = get(agg, "perm", r, m, t, "null_mean")
                if math.isnan(p95):
                    continue
                ntr = get(agg, "perm", r, m, t, "n_train")
                thr = null_p95_bound(ntr)
                ok = p95 <= thr and nm <= 0.02
                st.append(ok)
                real = get(agg, "ridge", r, m, t)
                pv = get(agg, "perm", r, m, t, "p_value_max")
                line = (f"- {r}/{m} {t}: null p95 = {p95:.4f} (bound {thr:.3f}), null mean = {nm:.4f}; "
                        f"real ridge R² = {fmt(real, 4)}, p ≤ {fmt(pv, 3)} (max over seeds)")
                lines.append(line + ("" if ok else "  ← FAIL"))
                if not ok:
                    bad.append(f"{r}/{m} {t}")
        if bad:
            lines.insert(0, "- offending: " + ", ".join(bad))
        lines.append("- gate: seed-mean null p95 ≤ max(0.10, 3.3/√n_train) AND seed-mean null mean ≤ 0.02 "
                     "per (input, target); null = ridge with the real fit's alpha on permuted train "
                     "labels. The √n term: for an input that encodes y almost exactly, a permuted fit "
                     "returns ≈ ȳ + r·(y − ȳ) with chance correlation r ~ N(0, 1/n_train), so null "
                     "R² ≈ 2r − r² and its p95 ≈ 3.3/√n_train without any leak")
        sh_v = [get(agg, "shuffle", r, m, t) for p, r, m, t in agg.index if p == "shuffle"]
        sh_v = [v for v in sh_v if not math.isnan(v)]
        if sh_v:
            lines.append(f"- (info, not gated) single shuffled-label fits: max seed-mean R² = "
                         f"{max(sh_v):.4f} over {len(sh_v)} (input, target) cells")
        gates.append(("null", "label-permutation null: p95 ≤ max(0.10, 3.3/√n_train), mean ≤ 0.02",
                      _status(st), lines, False))
        # plan v1 rule, informational: single shuffled-label fit, seed-mean R² ≤ 0.05
        lines, st = [], []
        for r, m in sorted({(r, m) for p, r, m, _ in agg.index if p == "shuffle"},
                           key=lambda x: (rep_key(x[0]), mode_key(x[1]))):
            for t in targets:
                v = get(agg, "shuffle", r, m, t)
                if math.isnan(v):
                    continue
                ok = v <= 0.05
                st.append(ok)
                if not ok:
                    lines.append(f"- {r}/{m} {t}: R² = {v:.4f}  ← FAIL")
        if st:
            lines.insert(0, f"- {sum(st)}/{len(st)} (input, target) cells pass (seed-mean R²); "
                            "failing cells listed")
        gates.append(("shuffle_v1", "plan v1 rule, informational: shuffled-label control R² ≤ 0.05",
                      _status(st), lines, True))
    else:
        for r, m in sorted({(r, m) for p, r, m, _ in agg.index if p == "shuffle"},
                           key=lambda x: (rep_key(x[0]), mode_key(x[1]))):
            for t in targets:
                v = get(agg, "shuffle", r, m, t)
                if math.isnan(v):
                    continue
                ok = v <= 0.05
                st.append(ok)
                lines.append(f"- {r}/{m} {t}: R² = {v:.4f}{'' if ok else '  ← FAIL'}")
        gates.append(("null", "shuffled-label control R² ≤ 0.05 (no perm rows)", _status(st), lines,
                      False))

    # state oracle (unquantised oracle_xc if present): log m and μ on push, μ only on slide
    om = state_oracle_mode(agg)
    lines, st = [], []
    if kind == "slide":
        mu = get(agg, "ridge", "state", om, "mu")
        if not math.isnan(mu):
            st.append(mu > 0.9)
            lines.append(f"- μ: R² = {mu:.4f} (need > 0.9){'' if mu > 0.9 else '  ← FAIL'}")
        lm = get(agg, "ridge", "state", om, "log_m")
        lines.append(f"- (info, not gated) log m: R² = {fmt(lm, 4)} (m is not identifiable on slide; "
                     "expected ≈ 0)")
        if om == "oracle_xc" and "oracle_x" in ridge_modes(agg, "state"):
            lines.append(f"- (info, not gated) state/oracle_x, {QUANT_ORACLE_NOTE}: R²(μ) = "
                         f"{fmt(get(agg, 'ridge', 'state', 'oracle_x', 'mu'), 4)}")
    elif kind == "push":
        lm, mu = get(agg, "ridge", "state", om, "log_m"), get(agg, "ridge", "state", om, "mu")
        if not math.isnan(lm):
            st.append(lm > 0.95)
            lines.append(f"- log m: R² = {lm:.4f} (need > 0.95)")
        if not math.isnan(mu):
            st.append(mu > 0.9)
            lines.append(f"- μ: R² = {mu:.4f} (need > 0.9)")
        if om == "oracle_xc":
            q = {t: get(agg, "ridge", "state", "oracle_x", t) for t in ("log_m", "mu")}
            if any(not math.isnan(v) for v in q.values()):
                lines.append(f"- (info, not gated) state/oracle_x, {QUANT_ORACLE_NOTE}: "
                             f"R²(log m) = {fmt(q['log_m'], 4)}, R²(μ) = {fmt(q['mu'], 4)}")
    else:
        lines.append(f"- not applicable: defined for push- and slide-type datasets only (`{ds}` is {kind})")
        for o in [x for x in ("oracle_xc", "oracle_x") if x in ridge_modes(agg, "state")]:
            vals = ", ".join(f"R²({TARGET_LABEL[t]}) = {fmt(get(agg, 'ridge', 'state', o, t), 4)}"
                             for t in targets)
            lines.append(f"- (info) state/{o}: {vals}")
    oname = ("R²(μ) > 0.9 (μ only on slide)" if kind == "slide"
             else "R²(log m) > 0.95, R²(μ) > 0.9")
    gates.append(("oracle", f"state/{om} on {ds}: {oname}", _status(st), lines, False))
    if kind == "push" and om == "oracle_xc":
        # plan v1 rule, informational: the 1-px quantised oracle_x
        lines, st = [], []
        for t, thr in (("log_m", 0.95), ("mu", 0.9)):
            v = get(agg, "ridge", "state", "oracle_x", t)
            if not math.isnan(v):
                st.append(v > thr)
                lines.append(f"- {TARGET_LABEL[t]}: R² = {v:.4f} (need > {thr}){'' if v > thr else '  ← FAIL'}")
        gates.append(("oracle_v1", f"plan v1 rule, informational: state/oracle_x on {ds}: "
                                   "R²(log m) > 0.95, R²(μ) > 0.9", _status(st), lines, True))

    # slide
    lines, st = [], []
    if slide is not None:
        sname, sdf, sagg = slide
        lines.append(f"dataset `{sname}`; video/pixel reps only (contact, state excluded)")
        for p, r, m, t in sorted(sagg.index, key=lambda x: (rep_key(x[1]), mode_key(x[2]))):
            if p != "ridge" or t != "log_m":
                continue
            v = get(sagg, p, r, m, t)
            if r in NON_VIDEO_REPS or "contact/" in m or "state/" in m:
                continue
            ok = v < 0.05
            st.append(ok)
            lines.append(f"- {r}/{m}: R²(log m) = {v:.4f}{'' if ok else '  ← LEAK?'}")
        s_orc = [x for x in ("oracle_xc", "oracle_x") if x in ridge_modes(sagg, "state")]
        for om_ in s_orc or ["oracle_x"]:
            sx = get(sagg, "ridge", "state", om_, "log_m")
            lines.append(f"- (info, not gated) state/{om_} on slide: R²(log m) = {fmt(sx, 4)} "
                         "(expected ≈ 0)")
        cr = get(sagg, "ridge", "contact", "raw", "log_m")
        if not math.isnan(cr):
            lines.append(f"- (info, not gated) contact/raw ORACLE on slide: R²(log m) = {cr:.4f} "
                         "(high by construction, N = mg)")
    gates.append(("slide", "slide: R²(log m) < 0.05 for all video/pixel reps", _status(st), lines,
                  False))

    # poscheck
    lines, st = [], []
    if poscheck is not None:
        pname, pdf, pagg = poscheck
        lines.append(f"dataset `{pname}`, target x0_m")
        reps = sorted({r for p, r, m, t in pagg.index if p == "ridge" and t == "x0_m"}, key=rep_key)
        for r in reps:
            modes = [m for m in ridge_modes(pagg, r) if re.match(r"^clip_mean(_L\d+)?$", m)]
            if not modes:
                continue
            main = "clip_mean" if "clip_mean" in modes else modes[-1]
            v = get(pagg, "ridge", r, main, "x0_m")
            allm = [m for m in ridge_modes(pagg, r) if m.startswith(("clip_mean", "clip_seq", "clip_grid"))]

            def val_key(m, r=r):   # choose the pooling on val; test R² is only reported
                vv = get(pagg, "ridge", r, m, "x0_m", "r2_val")
                return -math.inf if math.isnan(vv) else vv
            best_mode = max(allm, key=val_key)
            best = get(pagg, "ridge", r, best_mode, "x0_m")
            ok = best >= 0.5
            st.append(ok)
            others = ", ".join(f"{m}={get(pagg, 'ridge', r, m, 'x0_m'):.3f}" for m in modes if m != main)
            seqs = [m for m in allm if not m.startswith("clip_mean")]
            seqtxt = ", ".join(f"{m}={get(pagg, 'ridge', r, m, 'x0_m'):.3f}" for m in seqs)
            msg = f"- {r}/{main}: R²(x0) = {v:.4f}"
            if others:
                msg += f" (other layers: {others})"
            if seqtxt:
                msg += f"; other poolings: {seqtxt}"
            if v < 0.5:
                msg += (f"  ← global mean loses position for {r}; position-preserving pooling "
                        f"(chosen on val): {best_mode} = {best:.3f} (test)"
                        + ("" if ok else " (< 0.5)"))
            elif best_mode != main:
                msg += f"; val-chosen pooling {best_mode} = {best:.3f} (test)"
            lines.append(msg)
    gates.append(("poscheck", "poscheck: R²(x0) ≥ 0.5 per encoder for its val-chosen pooling "
                              "(clip_mean shown)", _status(st), lines, False))

    # contact oracle labelling
    has_raw = any(p == "ridge" and r == "contact" and m == "raw" for p, r, m, _ in agg.index)
    if has_raw:
        flag = get(agg, "ridge", "contact", "raw", targets[0], "oracle")
        lines = ["- contact/raw rows are labelled **ORACLE** in table.md and drawn hatched "
                 "('Contact oracle') in headline.png",
                 f"- probes.csv `oracle` column for contact/raw: {bool(flag == 1.0)}"
                 + ("" if flag == 1.0 else " (report forces the label anyway)")]
        gates.append(("contact_label", "contact/raw labelled ORACLE", "PASS", lines, False))
    else:
        gates.append(("contact_label", "contact/raw labelled ORACLE", "N/A", ["- no contact/raw rows"],
                      False))
    return gates


def _status(oks):
    if not oks:
        return "N/A"
    return "PASS" if all(oks) else "FAIL"


def write_gates(ds, gates, path, df):
    warn_ = commit_warning(df)
    out = [f"# Sanity gates: `{ds}`", ""] + ([warn_, ""] if warn_ else []) + [
        f"- {l}" for l in header_info(df)] + [
        "- All numbers: seed-mean test R² (perm: seed-mean of the per-seed null p95 / null mean).",
        "", "| Gate | Status |", "|---|---|"]
    out += [f"| {n} | **{s}**{' (info)' if info else ''} |" for _, n, s, _, info in gates]
    out.append("")
    for _, n, s, lines, info in gates:
        out += [f"## {s}{' (info)' if info else ''} — {n}", ""] + (lines or ["- n/a"]) + [""]
    path.write_text("\n".join(out) + "\n")


def gate_summary(gates):
    main = ", ".join(f"{k} {s}" for k, _, s, _, info in gates if not info)
    info = ", ".join(f"{k} {s}" for k, _, s, _, i in gates if i)
    return main + (f"; plan v1 rules (informational): {info}" if info else "")


# ----------------------------------------------------------------------------- figures

def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "axes.titlecolor": INK, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "legend.frameon": False, "savefig.dpi": 150, "figure.dpi": 100,
        "hatch.color": "white", "hatch.linewidth": 1.2,
    })
    return plt


def headline_entries(df, agg):
    """Ordered list of (rep, probe, mode, label, is_oracle)."""
    targets = present_targets(df)
    reps = sorted({r for p, r, m, _ in agg.index if p == "ridge" and r != "combo"}, key=rep_key)
    ent = []
    if any(p == "mean" for p, *_ in agg.index):
        r, m = next((r, m) for p, r, m, _ in agg.index if p == "mean")
        ent.append((r, "mean", m, "Mean pred.", False))
    tail = []
    for rep in reps:
        if rep == "mean":
            continue
        m, how = clip_mode_for(agg, rep, targets)
        if rep == "contact":
            if "realistic" in ridge_modes(agg, rep):
                tail.append((rep, "ridge", "realistic", "Contact realistic", False))
            if m:
                tail.append((rep, "ridge", m, "Contact raw (ORACLE)", True))
            continue
        if not m:
            continue
        lab = rep_label(rep)
        if rep == "state":
            lab = "State x(t) [oracle_xc]" if m == "oracle_xc" else "State x(t) [oracle_x]"
            tail.insert(0, (rep, "ridge", m, lab, True))
            continue
        lab += f" [{m}]"
        ent.append((rep, "ridge", m, lab, False))
    return ent + tail


def last_frame_mode(agg, rep, probe, mode):
    """'last' if this video/pixel rep has a last-frame ridge row (drawn as a hollow marker)."""
    if probe != "ridge" or rep in NON_VIDEO_REPS or rep == "combo" or mode == "last":
        return None
    return "last" if "last" in ridge_modes(agg, rep) else None


def fig_headline(ds, df, agg, slide, path):
    """Horizontal bars (one row per input, readable labels). Grey row band = oracle input (by
    each entry's flag). Hollow marker = the same encoder's last-frame mode. Dashed line per
    target = final-frame-only reference (pixels/last)."""
    plt = _style()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    targets = [t for t in ("log_m", "mu") if t in set(df["target"])] or present_targets(df)
    ent = headline_entries(df, agg)
    if not ent:
        return False
    n = len(ent)
    has_slide = slide is not None
    fig_h = 0.27 * n * max(len(targets), 1) + 1.9
    fig = plt.figure(figsize=(11.5 if has_slide else 8.5, fig_h))
    if has_slide:
        gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1.0], wspace=0.05)
        ax = fig.add_subplot(gs[0])
        axs = fig.add_subplot(gs[1], sharey=ax)
    else:
        ax, axs = fig.add_subplot(111), None
    y = np.arange(n)
    h = 0.8 / len(targets)
    BAND = "#efeee9"
    for i, e in enumerate(ent):            # oracle band by each entry's flag, not by position
        if e[4]:
            for a_ in (ax, axs):
                if a_ is not None:
                    a_.axhspan(i - 0.5, i + 0.5, color=BAND, zorder=0, lw=0)
    vals_all = []
    for k, t in enumerate(targets):
        ys = y - 0.4 + h * (k + 0.5)
        for i, (r, p, m, lab, orc) in enumerate(ent):
            v = get(agg, p, r, m, t)
            e = get(agg, p, r, m, t, "r2_std")
            e = 0.0 if math.isnan(e) else e
            if math.isnan(v):
                ax.text(0.01, ys[i], "n/a", va="center", fontsize=7, color=MUTED)
                continue
            vals_all.append(v - e)
            ax.barh(ys[i], v, height=h * 0.9, color=C_TARGET[t], edgecolor="white", linewidth=0.8,
                    hatch="///" if (r, m) in ORACLE_MODES else None, zorder=2)
            ax.errorbar(v, ys[i], xerr=e, fmt="none", ecolor=INK2, elinewidth=1, capsize=2, zorder=3)
            ax.text(max(v, 0) + e + 0.012, ys[i], f"{v + 0.0:.2f}".replace("-0.00", "0.00"),
                    va="center", fontsize=7, color=INK2)
            lm = last_frame_mode(agg, r, p, m)
            lv = get(agg, "ridge", r, lm, t) if lm else float("nan")
            if not math.isnan(lv):
                vals_all.append(lv)
                ax.plot([lv], [ys[i]], marker="o", ms=5.5, mfc="white", mec=INK, mew=1.1, ls="none",
                        zorder=4)
    refs = {t: get(agg, "ridge", "pixels", "last", t) for t in targets}
    for t, rv in refs.items():
        if not math.isnan(rv):
            ax.axvline(rv, color=C_TARGET[t], ls="--", lw=1.1, zorder=1)
    ax.axvline(0, color=INK, lw=0.9, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([e[3] for e in ent], fontsize=8.5)
    ax.set_ylim(n - 0.5, -0.5)                    # first entry on top
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    lo = min([-0.1] + [v - 0.05 for v in vals_all])
    ax.set_xlim(max(lo, -1.0), 1.12)
    ax.set_xlabel("test R² (mean ± std over 3 splits)")
    handles = [Patch(facecolor=C_TARGET[t], label=f"R² {TARGET_LABEL_EN[t]}") for t in targets]
    if any(e[4] for e in ent):
        handles.append(Patch(facecolor=BAND, edgecolor=MUTED, linewidth=0.5, label="oracle input (grey row)"))
    if any((e[0], e[2]) in ORACLE_MODES for e in ent):
        handles.append(Patch(facecolor=MUTED, hatch="///", edgecolor="white", label="exact contact (oracle)"))
    if any(last_frame_mode(agg, e[0], e[1], e[2]) for e in ent):
        handles.append(Line2D([], [], marker="o", ms=5.5, mfc="white", mec=INK, mew=1.1, ls="none",
                              label="same encoder, last frame only"))
    for t, rv in refs.items():
        if not math.isnan(rv):
            handles.append(Line2D([], [], color=C_TARGET[t], ls="--", lw=1.1,
                                  label=f"final frame only (pixels/last), {TARGET_LABEL[t]}: {rv:.2f}"))
    ax.set_title(f"Linear probes on {ds}: clip-level input (mode chosen on val)", loc="left",
                 fontsize=11)

    if axs is not None:
        sname, sdf, sagg = slide
        drawn = []
        for i, (r, p, m, lab, orc) in enumerate(ent):
            v = get(sagg, p, r, m, "log_m")
            e = get(sagg, p, r, m, "log_m", "r2_std")
            if math.isnan(v) and p == "ridge" and r not in ("pixels", "state", "contact"):
                m2, _ = clip_mode_for(sagg, r, ["log_m"])
                if m2:
                    v, e = get(sagg, p, r, m2, "log_m"), get(sagg, p, r, m2, "log_m", "r2_std")
            e = 0.0 if math.isnan(e) else e
            if math.isnan(v):
                axs.text(0.0, i, " n/a", va="center", fontsize=7, color=MUTED)
                continue
            drawn.append((v, e))
            hatch = (r, m) in ORACLE_MODES
            axs.barh(i, v, height=0.6, color=C_SLIDE if not hatch else MUTED,
                     hatch="///" if hatch else None, edgecolor="white", xerr=e, zorder=2,
                     error_kw=dict(ecolor=INK2, elinewidth=1, capsize=2))
            axs.text(max(v, 0) + e + 0.03, i, f"{v:.2f}".replace("-0.00", "0.00"), va="center",
                     fontsize=7, color=INK2)
        axs.axvline(0, color=INK, lw=0.9)
        axs.axvspan(-0.05, 0.05, color="#f2f1ee", zorder=0)
        axs.tick_params(axis="y", left=False, labelleft=False)
        axs.grid(axis="x", color=GRID)
        axs.grid(axis="y", visible=False)
        # limits from what was drawn (fallback modes, the mean row and the error bars included)
        axs.set_xlim(min([-0.2] + [v - e - 0.05 for v, e in drawn]),
                     max([0.3] + [v + e + 0.2 for v, e in drawn]))
        axs.set_xlabel("test R²(log m)")
        axs.set_title(f"{sname}: mass not identifiable\n(video reps should be ≈ 0)", loc="left",
                      fontsize=9.5)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.0), ncol=3, fontsize=8,
               borderaxespad=0.2)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def preds_path(ds, rep, mode, probe, target, seed):
    m = mode.replace("/", "-").replace("+", "_plus_")
    return results_dir(ds) / "preds" / f"{rep}__{m}__{probe}__{target}__s{seed}.npz"


def load_preds(ds, rep, mode, probe, target, seed):
    p = preds_path(ds, rep, mode, probe, target, seed)
    if not p.exists():
        return None
    try:
        z = np.load(p, allow_pickle=False)
        return np.asarray(z["ids"]).astype(str), np.asarray(z["y_true"], float), np.asarray(z["y_pred"], float)
    except Exception as e:  # noqa: BLE001
        print(f"[report] cannot read {p}: {e}", file=sys.stderr)
        return None


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) < 2:
        return float("nan")
    ss = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - p) ** 2) / ss) if ss > 0 else float("nan")


def scatter_slots(agg, rep, targets):
    """4 column slots: (static, mid, seq, grid); each pick by val among its family."""
    modes = ridge_modes(agg, rep)
    if rep == "pixels":
        return [m if m in modes else None for m in ("static", "last", "clip_seq")] + [None]
    if rep == "state":
        om = state_oracle_mode(agg)
        return [None, "oracle_fit" if "oracle_fit" in modes else None, om if om in modes else None, None]
    if rep == "contact":
        return [None, "realistic" if "realistic" in modes else None, "raw" if "raw" in modes else None,
                None]
    st = "static" if "static" in modes else ("last" if "last" in modes else None)
    cm = [m for m in modes if re.match(r"^clip_mean(_L\d+)?$", m)]
    cs = [m for m in modes if re.match(r"^clip_seq(_L\d+)?$", m)]
    cg = [m for m in modes if re.match(r"^clip_grid\d+(_L\d+)?$", m)]
    mid = pick_by_val(agg, rep, cm, targets, prefer=("clip_mean",))[0] if cm else None
    seq = pick_by_val(agg, rep, cs, targets, prefer=("clip_seq",))[0] if cs else None
    grd = pick_by_val(agg, rep, cg, targets)[0] if cg else None
    if mid is None and seq is None and grd is None:
        others = [m for m in modes if m not in ("static", "last")]
        seq = others[-1] if others else None
    return [st, mid, seq, grd]


def fig_scatter(ds, df, agg, target, path):
    plt = _style()
    targets = present_targets(df)
    reps = sorted({r for p, r, m, t in agg.index if p == "ridge" and t == target and r not in ("combo",)},
                  key=rep_key)
    grid = []  # rows of 4 cells: (rep|None, mode|None, preds|None)
    for rep in reps:
        if rep in ("state", "contact", "mean"):
            continue
        slots = scatter_slots(agg, rep, targets)
        names = (("static", "last", "clip_seq", "clip_grid") if rep == "pixels"
                 else ("static", "clip_mean", "clip_seq", "clip_grid"))
        cells = [(rep, m or nm, load_preds(ds, rep, m, "ridge", target, 0) if m else None)
                 for m, nm in zip(slots, names)]
        if any(c[2] is not None for c in cells):
            grid.append(cells)
    # 4th (grid) column only if some representation has clip_grid preds
    ncol = 4 if any(row[3][2] is not None for row in grid) else 3
    grid = [row[:ncol] for row in grid]
    # oracles share one row: state/oracle_xc (else oracle_x), contact/realistic, contact/raw
    orow = [(r, m, load_preds(ds, r, m, "ridge", target, 0))
            for r, m in (("state", state_oracle_mode(agg)), ("contact", "realistic"), ("contact", "raw"))
            if m in ridge_modes(agg, r)]
    orow = [c for c in orow if c[2] is not None]
    if orow:
        grid.append(orow + [(None, None, None)] * (ncol - len(orow)))
    if not grid:
        return False
    alld = [c[2] for row in grid for c in row if c[2] is not None]
    allv = np.concatenate([np.r_[d[1], d[2]] for d in alld])
    lo, hi = np.percentile(allv, [0.5, 99.5])
    yt = np.concatenate([d[1] for d in alld])
    lo, hi = min(lo, yt.min()), max(hi, yt.max())
    pad = 0.06 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    nr = len(grid)
    fig, axes = plt.subplots(nr, ncol, figsize=(3.0 * ncol, 2.75 * nr + 0.6), squeeze=False)
    for i, row in enumerate(grid):
        for j in range(ncol):
            ax = axes[i, j]
            ax.grid(axis="both", color=GRID)
            rep, mode, d = row[j]
            if d is None:
                ax.set_axis_off()
                if rep is not None:
                    ax.text(0.5, 0.5, f"{rep_label(rep)} · {mode}\nn/a", ha="center",
                            va="center", fontsize=8, color=MUTED, transform=ax.transAxes)
                continue
            _, yt_, yp_ = d
            ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--", zorder=1)
            orc = (rep, mode) in ORACLE_MODES or rep == "state"
            ax.scatter(yt_, np.clip(yp_, lo, hi), s=9, alpha=0.6, lw=0,
                       color=C_TARGET.get(target, "#2a78d6") if not orc else INK2, zorder=2)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            tag = " (ORACLE)" if (rep, mode) in ORACLE_MODES else ""
            ax.set_title(f"{rep_label(rep)} · {mode}{tag}\nR² = {r2(yt_, yp_):.3f} (n={len(yt_)})",
                         fontsize=8.5, loc="left")
            ax.tick_params(labelsize=7.5)
            if all(g[j][2] is None for g in grid[i + 1:]):
                ax.set_xlabel(f"true {TARGET_LABEL_EN[target]}", fontsize=8)
            if all(row[k][2] is None for k in range(j)):
                ax.set_ylabel("predicted", fontsize=8)
    fig.suptitle(f"{ds}: {TARGET_LABEL_EN[target]}: predicted vs true (ridge, test, split seed 0)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / (2.75 * nr + 0.6)))
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def fig_by_moved_px(ds, df, agg, path):
    try:
        from physprobe.io import load_meta
        meta = load_meta(ds)
    except Exception as e:  # noqa: BLE001
        print(f"[report] r2_by_moved_px skipped: no meta ({e})", file=sys.stderr)
        return False
    mv = {str(r["id"]): r.get("moved_px") for r in meta}
    mm = {str(r["id"]): r.get("m") for r in meta}
    if not any(v is not None for v in mv.values()):
        return False
    targets = [t for t in ("log_m", "mu") if t in set(df["target"])]
    if not targets:
        return False
    ptargets = present_targets(df)
    keys = []
    for rep in ("pixels", "dinov2_b", "vjepa2_l", "state"):
        m, _ = clip_mode_for(agg, rep, ptargets)
        if m:
            keys.append((rep, m))
    if not keys:
        return False
    plt = _style()
    okm = [r for r in meta if r.get("ok", True)]

    def edges(key):
        v = np.array([r[key] for r in okm if r.get(key) is not None], float)
        return np.quantile(v, [1 / 3, 2 / 3]) if len(v) >= 3 else None

    bins = {"moved_px": (mv, edges("moved_px")), "m": (mm, edges("m"))}
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
    fig, axes = plt.subplots(len(targets), 2, figsize=(10, 3.3 * len(targets)), squeeze=False)
    any_drawn = False
    for i, t in enumerate(targets):
        for j, (bk, (lut, ed)) in enumerate(bins.items()):
            ax = axes[i, j]
            if ed is None:
                ax.set_axis_off()
                continue
            w = 0.8 / len(keys)
            for k, (rep, mode) in enumerate(keys):
                per_seed = []
                for s in sorted(set(df["seed"])):
                    d = load_preds(ds, rep, mode, "ridge", t, int(s))
                    if d is None:
                        continue
                    ids, yt_, yp_ = d
                    v = np.array([lut.get(str(x)) if lut.get(str(x)) is not None else np.nan
                                  for x in ids], float)
                    b = np.digitize(v, ed)
                    per_seed.append([r2(yt_[(b == q) & ~np.isnan(v)], yp_[(b == q) & ~np.isnan(v)])
                                     for q in range(3)])
                if not per_seed:
                    continue
                arr = np.array(per_seed, float)
                mu_, sd_ = np.nanmean(arr, 0), np.nanstd(arr, 0)
                ax.bar(np.arange(3) - 0.4 + w * (k + 0.5), mu_, w * 0.92, yerr=sd_, color=pal[k % 4],
                       edgecolor="white", error_kw=dict(ecolor=INK2, elinewidth=1, capsize=2),
                       label=f"{rep_label(rep)} [{mode}]")
                any_drawn = True
            ax.axhline(0, color=INK, lw=0.9)
            unit = "px" if bk == "moved_px" else "kg"
            ax.set_xticks(range(3))
            ax.set_xticklabels([f"low\n< {ed[0]:.3g} {unit}", f"mid\n{ed[0]:.3g}–{ed[1]:.3g}",
                                f"high\n> {ed[1]:.3g} {unit}"], fontsize=8)
            ax.set_title(f"R²({TARGET_LABEL_EN[t]}) by tercile of {bk}", loc="left", fontsize=10)
            ylo = ax.get_ylim()[0]
            ax.set_ylim(max(min(-0.2, ylo), -1.0), 1.05)
            if ylo < -1.0:
                ax.set_title(ax.get_title(loc="left") + "  (axis clipped at −1)", loc="left",
                             fontsize=10)
            if j == 0:
                ax.set_ylabel("test R² within bin")
    if not any_drawn:
        plt.close(fig)
        return False
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(keys), fontsize=8.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{ds}: probe quality vs visible motion and mass\n(within-bin test R², mean ± std over splits;"
                 " bins narrow the target range, so R² drops by construction)",
                 x=0.01, ha="left", fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


# ----------------------------------------------------------------------------- main

def _opt(name):
    df = load_probes(name)
    return (name, df, aggregate(df)) if df is not None else None


def render(ds, s_name=None, p_name=None):
    d_s, d_p = companions(ds)
    s_name = s_name or d_s
    p_name = p_name or d_p
    df = load_probes(ds)
    if df is None:
        print(f"[report] no {results_dir(ds) / 'probes.csv'}", file=sys.stderr)
        return 1
    agg = aggregate(df)
    kind = dataset_kind(ds)
    slide, poscheck = _opt(s_name), _opt(p_name)
    fig_slide = slide if kind == "push" else None   # slide section / panel only on push reports
    out = results_dir(ds)
    figs = out / "figs"
    figs.mkdir(parents=True, exist_ok=True)

    gates = run_gates(ds, df, agg, slide, poscheck, kind)
    summ = gate_summary(gates)
    warn_ = commit_warning(df)
    if warn_:
        print(f"[report] {ds}: " + warn_.lstrip("> "), file=sys.stderr)
    table, _ = build_table(ds, df, agg, fig_slide, summ)
    (out / "table.md").write_text(table)
    write_gates(ds, gates, out / "gates.md", df)

    made = ["table.md", "gates.md"]
    if fig_headline(ds, df, agg, fig_slide, figs / "headline.png"):
        made.append("figs/headline.png")
    for t in present_targets(df):
        if fig_scatter(ds, df, agg, t, figs / f"scatter_{t}.png"):
            made.append(f"figs/scatter_{t}.png")
    try:
        if fig_by_moved_px(ds, df, agg, figs / "r2_by_moved_px.png"):
            made.append("figs/r2_by_moved_px.png")
    except Exception as e:  # noqa: BLE001  (optional figure must never break the report)
        print(f"[report] r2_by_moved_px failed: {e}", file=sys.stderr)

    print(f"[report] {ds}: wrote {', '.join(made)} -> {out}", file=sys.stderr)
    print(f"gates[{ds}] ({kind}) slide={s_name if slide else 'n/a'} "
          f"poscheck={p_name if poscheck else 'n/a'}: {summ}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--slide-dataset", default=None)
    ap.add_argument("--poscheck-dataset", default=None)
    ap.add_argument("--all", action="store_true",
                    help="render slide, poscheck, then push (each that has a probes.csv), so push's "
                         "cross-dataset gates see the others; --dataset, if given, is rendered first")
    a = ap.parse_args(argv)
    if not a.dataset and not a.all:
        ap.error("--dataset or --all is required")
    def have(d):
        return (results_dir(d) / "probes.csv").exists()

    names = [a.dataset] if a.dataset else []
    if a.all:  # push always last, so its slide/poscheck gates see the others
        names = [n for n in names if n != "push"]
        names += [d for d in ("slide", "poscheck") if d not in names and have(d)]
        if have("push") or a.dataset == "push":
            names.append("push")
        if not names:
            print("[report] --all: no results/{slide,poscheck,push}/probes.csv", file=sys.stderr)
            return 1
    rc = 0
    for n in names:
        o = (a.slide_dataset, a.poscheck_dataset) if n == a.dataset else (None, None)
        rc = render(n, *o) or rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
