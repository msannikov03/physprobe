# PhysProbe results: `poscheck`

- Splits — s0: n_train=210, n_val=45, n_test=45, n_excluded=0; s1: n_train=210, n_val=45, n_test=45, n_excluded=0; s2: n_train=210, n_val=45, n_test=45, n_excluded=0
- git commit: `c5fe74b`
- runs: 2026-10-23T00:58:59+00:00 … 2026-10-23T00:58:59+00:00
- R² = test R², mean ± std over seeds (ridge probe unless stated). RMSE for log m is log-RMSE.
- Δ = R² − R²(pixels/clip_seq); frac = R² / R²(state/oracle_xc) (per target); a reference that is beaten (frac > 1.001) is printed as `n/a (>1)`, a tie as 1.00; `state/oracle_xc` = unquantised x(t), the linear x(t) read-out reference: a ceiling for μ, not for log m (a linear probe on x(t) does not invert d = J²/(2m²μg), so frac for log m is not a fraction of a ceiling); `state/oracle_x` (1-px quantised) is shown for information only.
- ★ = mode chosen by validation R² (mean r2_val over targets; candidates clip_mean*, clip_seq*, clip_grid*), used in headline.png.

**Gates:** mean PASS, static N/A, null PASS, oracle N/A, slide PASS, poscheck PASS, contact_label PASS; plan v1 rules (informational): shuffle_v1 FAIL (details in `gates.md`)

**Mode / layer selection (by val, not test):** `track` → `clip` (val); `dinov2_b_random` → `clip_mean` (val); `dinov2_b` → `clip_seq` (val); `vjepa2_l_random` → `clip_mean` (val); `vjepa2_l` → `clip_mean` (val); `cosmos_cv8` → `clip_grid4` (val); best `vjepa2_l` layer for clip_mean: L24 (last) (`clip_mean`, by val); best `vjepa2_l` layer for clip_seq: L24 (last) (`clip_seq`, by val)

## Main table (ridge)

| Representation | Mode | x0 R² | x0 RMSE | Δ vs pixels/clip_seq (x0) | frac of state/oracle_xc (x0) | seeds | note |
|---|---|---|---|---|---|---|---|
| mean (predict train mean) | - | -0.017 ± 0.017 | 0.341 | -1.017 | -0.02 | 3 |  |
| pixels | `static` | 1.000 ± 0.000 | 0.004 | +0.000 | 1.00 | 3 |  |
| pixels | `last` | 1.000 ± 0.000 | 0.004 | +0.000 | 1.00 | 3 |  |
| pixels | `clip_seq` | 1.000 ± 0.000 | 0.004 | +0.000 | 1.00 | 3 | ★ |
| track | `clip` | 1.000 ± 0.000 | 0.001 | +0.000 | 1.00 | 3 | ★ |
| dinov2_b_random | `static` | 0.998 ± 0.000 | 0.014 | -0.002 | 1.00 | 3 |  |
| dinov2_b_random | `last` | 0.998 ± 0.000 | 0.014 | -0.002 | 1.00 | 3 |  |
| dinov2_b_random | `clip_mean` | 0.998 ± 0.000 | 0.014 | -0.002 | 1.00 | 3 | ★ |
| dinov2_b_random | `clip_seq` | 0.997 ± 0.001 | 0.019 | -0.003 | 1.00 | 3 |  |
| dinov2_b | `static` | 0.999 ± 0.000 | 0.009 | -0.001 | 1.00 | 3 |  |
| dinov2_b | `last` | 0.999 ± 0.000 | 0.009 | -0.001 | 1.00 | 3 |  |
| dinov2_b | `clip_mean` | 0.999 ± 0.000 | 0.009 | -0.001 | 1.00 | 3 |  |
| dinov2_b | `clip_seq` | 0.999 ± 0.000 | 0.009 | -0.001 | 1.00 | 3 | ★ |
| vjepa2_l_random | `static` | 0.999 ± 0.000 | 0.012 | -0.001 | 1.00 | 3 |  |
| vjepa2_l_random | `clip_mean` | 0.999 ± 0.000 | 0.012 | -0.001 | 1.00 | 3 | ★ |
| vjepa2_l_random | `clip_seq` | 0.998 ± 0.001 | 0.015 | -0.002 | 1.00 | 3 |  |
| vjepa2_l | `static` | 0.999 ± 0.000 | 0.009 | -0.001 | 1.00 | 3 |  |
| vjepa2_l | `clip_mean_L08` | 0.984 ± 0.005 | 0.042 | -0.016 | 0.98 | 3 |  |
| vjepa2_l | `clip_mean_L16` | 0.997 ± 0.001 | 0.017 | -0.003 | 1.00 | 3 |  |
| vjepa2_l | `clip_mean` | 0.999 ± 0.000 | 0.009 | -0.001 | 1.00 | 3 | ★ |
| vjepa2_l | `clip_seq_L08` | 0.985 ± 0.007 | 0.041 | -0.015 | 0.98 | 3 |  |
| vjepa2_l | `clip_seq_L16` | 0.993 ± 0.004 | 0.027 | -0.007 | 0.99 | 3 |  |
| vjepa2_l | `clip_seq` | 0.999 ± 0.001 | 0.011 | -0.001 | 1.00 | 3 |  |
| cosmos_cv8 | `clip_mean` | 0.158 ± 0.070 | 0.310 | -0.842 | 0.16 | 3 |  |
| cosmos_cv8 | `clip_seq` | 0.166 ± 0.025 | 0.309 | -0.833 | 0.17 | 3 |  |
| cosmos_cv8 | `clip_grid4` | 0.986 ± 0.002 | 0.040 | -0.014 | 0.99 | 3 | ★ |
| state | `oracle_xc` | 1.000 ± 0.000 | 0.000 | +0.000 | 1.00 | 3 | **ORACLE** ★ |
| state | `oracle_x` | 1.000 ± 0.000 | 0.002 | +0.000 | 1.00 | 3 | **ORACLE** pixel-quantised oracle (not an upper bound) |
| state | `oracle_fit` | -0.017 ± 0.017 | 0.341 | -1.017 | -0.02 | 3 | **ORACLE** |
| contact | `raw` | -0.017 ± 0.017 | 0.341 | -1.017 | -0.02 | 3 | **ORACLE** ★ |
| contact | `realistic` | -0.017 ± 0.017 | 0.341 | -1.017 | -0.02 | 3 |  |
| pixels [shuffled labels] | `static` | -0.000 ± 0.086 | 0.338 | -1.000 | -0.00 | 3 | control |
| pixels [shuffled labels] | `last` | -0.000 ± 0.086 | 0.338 | -1.000 | -0.00 | 3 | control |
| pixels [shuffled labels] | `clip_seq` | -0.001 ± 0.076 | 0.338 | -1.001 | -0.00 | 3 | control |
| track [shuffled labels] | `clip` | 0.038 ± 0.137 | 0.331 | -0.962 | 0.04 | 3 | control |
| dinov2_b_random [shuffled labels] | `static` | -0.018 ± 0.014 | 0.341 | -1.018 | -0.02 | 3 | control |
| dinov2_b_random [shuffled labels] | `last` | -0.018 ± 0.014 | 0.341 | -1.018 | -0.02 | 3 | control |
| dinov2_b_random [shuffled labels] | `clip_mean` | -0.018 ± 0.014 | 0.341 | -1.018 | -0.02 | 3 | control |
| dinov2_b_random [shuffled labels] | `clip_seq` | -0.024 ± 0.024 | 0.342 | -1.024 | -0.02 | 3 | control |
| dinov2_b [shuffled labels] | `static` | -0.018 ± 0.063 | 0.341 | -1.017 | -0.02 | 3 | control |
| dinov2_b [shuffled labels] | `last` | -0.018 ± 0.063 | 0.341 | -1.018 | -0.02 | 3 | control |
| dinov2_b [shuffled labels] | `clip_mean` | -0.018 ± 0.063 | 0.341 | -1.018 | -0.02 | 3 | control |
| dinov2_b [shuffled labels] | `clip_seq` | -0.002 ± 0.109 | 0.338 | -1.002 | -0.00 | 3 | control |
| vjepa2_l_random [shuffled labels] | `static` | -0.025 ± 0.026 | 0.342 | -1.025 | -0.02 | 3 | control |
| vjepa2_l_random [shuffled labels] | `clip_mean` | -0.025 ± 0.026 | 0.342 | -1.025 | -0.02 | 3 | control |
| vjepa2_l_random [shuffled labels] | `clip_seq` | -0.036 ± 0.033 | 0.344 | -1.036 | -0.04 | 3 | control |
| vjepa2_l [shuffled labels] | `static` | 0.014 ± 0.069 | 0.335 | -0.985 | 0.01 | 3 | control |
| vjepa2_l [shuffled labels] | `clip_mean_L08` | -0.021 ± 0.018 | 0.342 | -1.021 | -0.02 | 3 | control |
| vjepa2_l [shuffled labels] | `clip_mean_L16` | -0.016 ± 0.018 | 0.341 | -1.016 | -0.02 | 3 | control |
| vjepa2_l [shuffled labels] | `clip_mean` | 0.014 ± 0.069 | 0.335 | -0.986 | 0.01 | 3 | control |
| vjepa2_l [shuffled labels] | `clip_seq_L08` | -0.038 ± 0.052 | 0.344 | -1.038 | -0.04 | 3 | control |
| vjepa2_l [shuffled labels] | `clip_seq_L16` | -0.010 ± 0.056 | 0.340 | -1.010 | -0.01 | 3 | control |
| vjepa2_l [shuffled labels] | `clip_seq` | 0.023 ± 0.108 | 0.334 | -0.977 | 0.02 | 3 | control |
| cosmos_cv8 [shuffled labels] | `clip_mean` | -0.007 ± 0.027 | 0.339 | -1.007 | -0.01 | 3 | control |
| cosmos_cv8 [shuffled labels] | `clip_seq` | -0.007 ± 0.027 | 0.339 | -1.007 | -0.01 | 3 | control |
| cosmos_cv8 [shuffled labels] | `clip_grid4` | 0.045 ± 0.119 | 0.330 | -0.955 | 0.04 | 3 | control |
| state [shuffled labels] | `oracle_xc` | 0.065 ± 0.157 | 0.326 | -0.934 | 0.07 | 3 | **ORACLE** control |
| state [shuffled labels] | `oracle_x` | 0.065 ± 0.157 | 0.326 | -0.934 | 0.07 | 3 | **ORACLE** pixel-quantised oracle (not an upper bound) control |
| state [shuffled labels] | `oracle_fit` | -0.017 ± 0.017 | 0.341 | -1.017 | -0.02 | 3 | **ORACLE** control |
| contact [shuffled labels] | `raw` | -0.016 ± 0.017 | 0.341 | -1.016 | -0.02 | 3 | **ORACLE** control |
| contact [shuffled labels] | `realistic` | -0.027 ± 0.036 | 0.342 | -1.027 | -0.03 | 3 | control |

## Identifiability check: slide (log m should be ≈ 0)

n/a (no slide probes.csv)

