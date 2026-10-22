"""Visibility and stability statistics of a generated dataset, from meta.jsonl.

  $SIM_PY -m physprobe.sim.stats push [--first 300]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

from physprobe import io

QUANTILES = (0.0, 0.05, 0.5, 0.95, 1.0)


def gen_times(name):
    """Per-clip generation seconds (one worker, render + simulate + write) from data/<name>/.parts/."""
    out = {}
    for p in glob.glob(os.path.join(io.dataset_dir(name), ".parts", "[0-9]*.json")):
        with open(p) as f:
            r = json.load(f)
        if "_gen_s" in r:
            out[r["id"]] = r["_gen_s"]
    return out


def _q(vals):
    v = np.asarray([x for x in vals if x is not None], dtype=np.float64)
    if v.size == 0:
        return None
    return {f"q{int(q * 100):02d}": round(float(np.quantile(v, q)), 6) for q in QUANTILES}


def summarize(rows, times=None):
    n = len(rows)
    if n == 0:
        return {"n": 0}

    def frac(pred):
        return round(sum(bool(pred(r)) for r in rows) / n, 4)

    out = dict(
        n=n,
        frac_moved_px_lt3=frac(lambda r: r["moved_px"] < 3),
        frac_frames_moving_lt4=frac(lambda r: r["frames_moving"] < 4),
        frac_out_of_frame=frac(lambda r: not r["in_frame"]),
        frac_ok=frac(lambda r: r["ok"]),
        frac_tilt_ge5=frac(lambda r: r["max_tilt_deg"] is None or r["max_tilt_deg"] >= 5),
        frac_yaw_ge5=frac(lambda r: r["max_yaw_deg"] is None or r["max_yaw_deg"] >= 5),
        frac_moving_at_end=frac(lambda r: r.get("moving_at_end")),
        splits={s: round(c / n, 4) for s, c in sorted(_count(r["split"] for r in rows).items())},
        frame0_hashes=len({r.get("frame0_hash") for r in rows}),
    )
    for key in ("moved_px", "disp_m", "frames_moving", "t_stop_s", "max_tilt_deg", "max_yaw_deg", "z_pen_max_m"):
        out[key] = _q(r.get(key) for r in rows)
    if times:
        t = [times[r["id"]] for r in rows if r["id"] in times]
        if t:
            out["gen_s_per_clip"] = dict(mean=round(float(np.mean(t)), 4), median=round(float(np.median(t)), 4),
                                         max=round(float(np.max(t)), 4), n=len(t))
    return out


def _count(it):
    c = {}
    for x in it:
        c[x] = c.get(x, 0) + 1
    return c


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("dataset")
    p.add_argument("--first", type=int, default=None, help="only the first N clips")
    a = p.parse_args(argv)
    rows = io.load_meta(a.dataset)
    if a.first:
        rows = rows[:a.first]
    print(json.dumps(summarize(rows, gen_times(a.dataset)), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
