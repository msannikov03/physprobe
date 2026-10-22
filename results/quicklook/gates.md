# Sanity gates (quick-look)

A gate passes only if every seed passes. Numbers are mean [worst seed] over the split seeds. These are the plan-v1 rules: G3 (quantised state/oracle_x ≥ 0.9) and G5 (single shuffle ≤ 0.05) are superseded by CONTRACT v1.2 (unquantised oracle_xc; label-permutation null with an n-aware bound), so a FAIL there is expected; the canonical gates are in results/<ds>/gates.md.

## Summary

- G1: PASS
- G2: PASS
- G3: FAIL
- G4: PASS
- G5: FAIL
- G6: PASS

## G1 PASS: static-mode inputs on push and slide, R² ≤ 0.05 for log_m and mu

| dataset | input | target | mean | worst seed | ok |
|---|---|---|---|---|---|
| push | dinov2_b/static | log_m | -0.006 | -0.000 | PASS |
| push | dinov2_b/static | mu | -0.003 | -0.002 | PASS |
| push | dinov2_b_random/static | log_m | -0.006 | -0.000 | PASS |
| push | dinov2_b_random/static | mu | -0.003 | -0.002 | PASS |
| push | pixels/static | log_m | -0.006 | -0.000 | PASS |
| push | pixels/static | mu | -0.003 | -0.002 | PASS |
| push | vjepa2_l/static | log_m | -0.006 | -0.000 | PASS |
| push | vjepa2_l/static | mu | -0.003 | -0.002 | PASS |
| push | vjepa2_l_random/static | log_m | -0.006 | -0.000 | PASS |
| push | vjepa2_l_random/static | mu | -0.003 | -0.002 | PASS |
| slide | dinov2_b/static | log_m | -0.024 | -0.003 | PASS |
| slide | dinov2_b/static | mu | -0.001 | -0.000 | PASS |
| slide | dinov2_b_random/static | log_m | -0.024 | -0.003 | PASS |
| slide | dinov2_b_random/static | mu | -0.001 | -0.000 | PASS |
| slide | pixels/static | log_m | -0.024 | -0.003 | PASS |
| slide | pixels/static | mu | -0.001 | -0.000 | PASS |
| slide | vjepa2_l/static | log_m | -0.024 | -0.003 | PASS |
| slide | vjepa2_l/static | mu | -0.001 | -0.000 | PASS |
| slide | vjepa2_l_random/static | log_m | -0.024 | -0.003 | PASS |
| slide | vjepa2_l_random/static | mu | -0.001 | -0.000 | PASS |

## G2 PASS: slide: R²(log_m) ≤ 0.05 for every video input (pixels, track, dinov2_*, vjepa2_*)

| dataset | input | target | mean | worst seed | ok |
|---|---|---|---|---|---|
| slide | dinov2_b/clip_mean | log_m | -0.030 | 0.002 | PASS |
| slide | dinov2_b/clip_seq | log_m | -0.055 | -0.004 | PASS |
| slide | dinov2_b/last | log_m | -0.029 | 0.002 | PASS |
| slide | dinov2_b/static | log_m | -0.024 | -0.003 | PASS |
| slide | dinov2_b_random/clip_mean | log_m | -0.029 | -0.004 | PASS |
| slide | dinov2_b_random/clip_seq | log_m | -0.042 | -0.018 | PASS |
| slide | dinov2_b_random/last | log_m | -0.028 | -0.004 | PASS |
| slide | dinov2_b_random/static | log_m | -0.024 | -0.003 | PASS |
| slide | pixels/clip_seq | log_m | -0.025 | -0.001 | PASS |
| slide | pixels/last | log_m | -0.024 | -0.002 | PASS |
| slide | pixels/static | log_m | -0.024 | -0.003 | PASS |
| slide | track/clip | log_m | -0.024 | -0.002 | PASS |
| slide | vjepa2_l/clip_mean | log_m | -0.027 | -0.008 | PASS |
| slide | vjepa2_l/clip_mean_L08 | log_m | -0.028 | 0.001 | PASS |
| slide | vjepa2_l/clip_mean_L16 | log_m | -0.028 | -0.002 | PASS |
| slide | vjepa2_l/clip_seq | log_m | -0.053 | -0.030 | PASS |
| slide | vjepa2_l/clip_seq_L08 | log_m | -0.054 | -0.018 | PASS |
| slide | vjepa2_l/clip_seq_L16 | log_m | -0.056 | -0.019 | PASS |
| slide | vjepa2_l/static | log_m | -0.024 | -0.003 | PASS |
| slide | vjepa2_l_random/clip_mean | log_m | -0.027 | -0.004 | PASS |
| slide | vjepa2_l_random/clip_seq | log_m | -0.042 | -0.015 | PASS |
| slide | vjepa2_l_random/static | log_m | -0.024 | -0.003 | PASS |

## G3 FAIL: state/oracle_x: push R² ≥ 0.9 for log_m and mu; slide R²(mu) ≥ 0.9

| dataset | input | target | mean | worst seed | ok |
|---|---|---|---|---|---|
| push | state/oracle_x | log_m | 0.974 | 0.973 | PASS |
| push | state/oracle_x | mu | 0.872 | 0.850 | FAIL |
| slide | state/oracle_x | mu | 0.999 | 0.999 | PASS |

## G4 PASS: poscheck: static modes of dinov2_b, vjepa2_l and pixels/static give R²(x0_m) ≥ 0.9

| dataset | input | target | mean | worst seed | ok |
|---|---|---|---|---|---|
| poscheck | dinov2_b/static | x0_m | 0.999 | 0.999 | PASS |
| poscheck | pixels/static | x0_m | 1.000 | 1.000 | PASS |
| poscheck | vjepa2_l/static | x0_m | 0.999 | 0.999 | PASS |

## G5 FAIL: every shuffle control has -0.5 ≤ R² ≤ 0.05

| dataset | target | fits | min R² | max R² |
|---|---|---|---|---|
| poscheck | x0_m | 117 | -0.234 | 0.216 |
| push | log_m | 129 | -0.084 | 0.010 |
| push | mu | 129 | -0.060 | 0.023 |
| slide | log_m | 129 | -0.104 | 0.083 |
| slide | mu | 129 | -0.101 | 0.049 |

Failing shuffle fits:

- slide contact/raw log_m seed 1: 0.083
- slide contact/raw log_m seed 1: 0.083
- poscheck cosmos_cv8/clip_grid4 x0_m seed 0: 0.167
- poscheck dinov2_b/clip_mean x0_m seed 0: 0.074
- poscheck dinov2_b/clip_seq x0_m seed 0: 0.195
- poscheck dinov2_b/last x0_m seed 0: 0.074
- poscheck dinov2_b/static x0_m seed 0: 0.073
- poscheck state/oracle_x x0_m seed 0: 0.216
- poscheck state/oracle_xc x0_m seed 0: 0.216
- poscheck track/clip x0_m seed 0: 0.154
- poscheck vjepa2_l/clip_mean x0_m seed 0: 0.094
- poscheck vjepa2_l/clip_seq x0_m seed 0: 0.199
- poscheck vjepa2_l/clip_seq_L08 x0_m seed 0: 0.094
- poscheck vjepa2_l/clip_seq_L16 x0_m seed 0: 0.150
- poscheck vjepa2_l/static x0_m seed 0: 0.094
- poscheck vjepa2_l_random/clip_seq x0_m seed 0: 0.082
- poscheck track/clip x0_m seed 0: 0.151
- poscheck state/oracle_x x0_m seed 0: 0.216

Diagnostic: correlation between the permuted and the true train labels. When an input encodes the target almost perfectly, the shuffled ridge inherits this chance correlation and its test R² is roughly 2·corr, so a positive corr of about 0.1 explains shuffle R² near 0.2 without any leak.

| dataset | target | seed | n_train | corr(y_perm, y) |
|---|---|---|---|---|
| push | log_m | 0 | 1400 | -0.008 |
| push | log_m | 1 | 1400 | -0.003 |
| push | log_m | 2 | 1400 | -0.021 |
| push | mu | 0 | 1400 | -0.006 |
| push | mu | 1 | 1400 | -0.003 |
| push | mu | 2 | 1400 | -0.019 |
| slide | log_m | 0 | 700 | -0.041 |
| slide | log_m | 1 | 700 | +0.051 |
| slide | log_m | 2 | 700 | -0.027 |
| slide | mu | 0 | 700 | +0.024 |
| slide | mu | 1 | 700 | -0.036 |
| slide | mu | 2 | 700 | +0.011 |
| poscheck | x0_m | 0 | 210 | +0.142 |
| poscheck | x0_m | 1 | 210 | -0.036 |
| poscheck | x0_m | 2 | 210 | -0.110 |

## G6 PASS: contact/raw on push (oracle) R² ≥ 0.95 for log_m and mu

| dataset | input | target | mean | worst seed | ok |
|---|---|---|---|---|---|
| push | contact/raw | log_m | 1.000 | 1.000 | PASS |
| push | contact/raw | mu | 1.000 | 1.000 | PASS |

