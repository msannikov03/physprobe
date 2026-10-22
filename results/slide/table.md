# PhysProbe results: `slide`

- Splits — s0: n_train=700, n_val=150, n_test=150, n_excluded=0; s1: n_train=700, n_val=150, n_test=150, n_excluded=0; s2: n_train=700, n_val=150, n_test=150, n_excluded=0
- git commit: `fce21d1`
- runs: 2026-10-22T22:45:46+00:00 … 2026-10-22T22:45:46+00:00
- R² = test R², mean ± std over seeds (ridge probe unless stated). RMSE for log m is log-RMSE.
- Δ = R² − R²(pixels/clip_seq); frac = R² / R²(state/oracle_xc) (per target); a reference that is beaten (frac > 1) is printed as `n/a (>1)`; `state/oracle_xc` = unquantised x(t), the linear x(t) read-out reference: a ceiling for μ, not for log m (a linear probe on x(t) does not invert d = J²/(2m²μg), so frac for log m is not a fraction of a ceiling); `state/oracle_x` (1-px quantised) is shown for information only.
- ★ = mode chosen by validation R² (mean r2_val over targets; candidates clip_mean*, clip_seq*, clip_grid*), used in headline.png.

**Gates:** mean PASS, static PASS, null PASS, oracle PASS, slide PASS, poscheck PASS, contact_label PASS; plan v1 rules (informational): shuffle_v1 PASS (details in `gates.md`)

**Mode / layer selection (by val, not test):** `track` → `clip` (val); `dinov2_b_random` → `clip_seq` (val); `dinov2_b` → `clip_seq` (val); `vjepa2_l_random` → `clip_seq` (val); `vjepa2_l` → `clip_mean_L08` (val); `cosmos_cv8` → `clip_grid4` (val); `combo` → `vjepa2_l/clip_mean_L16+contact/realistic` (val); best `vjepa2_l` layer for clip_mean: L08 (`clip_mean_L08`, by val); best `vjepa2_l` layer for clip_seq: L08 (`clip_seq_L08`, by val)

## Main table (ridge)

| Representation | Mode | log m R² | log m RMSE | μ R² | μ RMSE | Δ vs pixels/clip_seq (log m) | Δ vs pixels/clip_seq (μ) | frac of state/oracle_xc (log m) | frac of state/oracle_xc (μ) | seeds | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mean (predict train mean) | - | -0.024 ± 0.025 | 0.409 | -0.001 ± 0.001 | 0.177 | -0.000 | -1.001 | n/a | -0.00 | 3 |  |
| pixels | `static` | -0.024 ± 0.025 | 0.409 | -0.001 ± 0.001 | 0.177 | -0.000 | -1.001 | n/a | -0.00 | 3 |  |
| pixels | `last` | -0.024 ± 0.025 | 0.409 | 0.999 ± 0.000 | 0.006 | -0.000 | -0.001 | n/a | 1.00 | 3 |  |
| pixels | `clip_seq` | -0.024 ± 0.025 | 0.409 | 1.000 ± 0.000 | 0.001 | +0.000 | +0.000 | n/a | 1.00 | 3 | ★ |
| track | `clip` | -0.024 ± 0.025 | 0.409 | 1.000 ± 0.000 | 0.003 | +0.000 | -0.000 | n/a | 1.00 | 3 | ★ |
| dinov2_b_random | `static` | -0.024 ± 0.025 | 0.409 | -0.001 ± 0.001 | 0.177 | -0.000 | -1.001 | n/a | -0.00 | 3 |  |
| dinov2_b_random | `last` | -0.025 ± 0.024 | 0.409 | 1.000 ± 0.000 | 0.002 | -0.000 | -0.000 | n/a | 1.00 | 3 |  |
| dinov2_b_random | `clip_mean` | -0.026 ± 0.024 | 0.409 | 1.000 ± 0.000 | 0.002 | -0.002 | -0.000 | n/a | 1.00 | 3 |  |
| dinov2_b_random | `clip_seq` | -0.028 ± 0.023 | 0.410 | 1.000 ± 0.000 | 0.002 | -0.003 | -0.000 | n/a | 1.00 | 3 | ★ |
| dinov2_b | `static` | -0.024 ± 0.025 | 0.409 | -0.001 ± 0.001 | 0.177 | -0.000 | -1.001 | n/a | -0.00 | 3 |  |
| dinov2_b | `last` | -0.025 ± 0.024 | 0.409 | 0.999 ± 0.000 | 0.004 | -0.000 | -0.001 | n/a | 1.00 | 3 |  |
| dinov2_b | `clip_mean` | -0.025 ± 0.024 | 0.409 | 1.000 ± 0.000 | 0.003 | -0.001 | -0.000 | n/a | 1.00 | 3 |  |
| dinov2_b | `clip_seq` | -0.031 ± 0.031 | 0.410 | 1.000 ± 0.000 | 0.003 | -0.007 | -0.000 | n/a | 1.00 | 3 | ★ |
| vjepa2_l_random | `static` | -0.024 ± 0.025 | 0.409 | -0.001 ± 0.001 | 0.177 | -0.000 | -1.001 | n/a | -0.00 | 3 |  |
| vjepa2_l_random | `clip_mean` | -0.026 ± 0.023 | 0.409 | 1.000 ± 0.000 | 0.004 | -0.001 | -0.000 | n/a | 1.00 | 3 |  |
| vjepa2_l_random | `clip_seq` | -0.031 ± 0.021 | 0.410 | 1.000 ± 0.000 | 0.002 | -0.007 | -0.000 | n/a | 1.00 | 3 | ★ |
| vjepa2_l | `static` | -0.024 ± 0.025 | 0.409 | -0.001 ± 0.001 | 0.177 | -0.000 | -1.001 | n/a | -0.00 | 3 |  |
| vjepa2_l | `clip_mean_L08` | -0.025 ± 0.026 | 0.409 | 1.000 ± 0.000 | 0.004 | -0.001 | -0.000 | n/a | 1.00 | 3 | ★ |
| vjepa2_l | `clip_mean_L16` | -0.025 ± 0.025 | 0.409 | 1.000 ± 0.000 | 0.003 | -0.000 | -0.000 | n/a | 1.00 | 3 |  |
| vjepa2_l | `clip_mean` | -0.025 ± 0.024 | 0.409 | 1.000 ± 0.000 | 0.004 | -0.000 | -0.000 | n/a | 1.00 | 3 |  |
| vjepa2_l | `clip_seq_L08` | -0.032 ± 0.037 | 0.410 | 1.000 ± 0.000 | 0.003 | -0.008 | -0.000 | n/a | 1.00 | 3 |  |
| vjepa2_l | `clip_seq_L16` | -0.030 ± 0.033 | 0.410 | 1.000 ± 0.000 | 0.003 | -0.005 | -0.000 | n/a | 1.00 | 3 |  |
| vjepa2_l | `clip_seq` | -0.029 ± 0.026 | 0.410 | 1.000 ± 0.000 | 0.003 | -0.004 | -0.000 | n/a | 1.00 | 3 |  |
| cosmos_cv8 | `clip_mean` | -0.025 ± 0.024 | 0.409 | 0.761 ± 0.012 | 0.087 | -0.000 | -0.239 | n/a | 0.76 | 3 |  |
| cosmos_cv8 | `clip_seq` | -0.024 ± 0.024 | 0.409 | 0.967 ± 0.003 | 0.032 | +0.000 | -0.033 | n/a | 0.97 | 3 |  |
| cosmos_cv8 | `clip_grid4` | -0.024 ± 0.025 | 0.409 | 0.999 ± 0.000 | 0.006 | +0.000 | -0.001 | n/a | 1.00 | 3 | ★ |
| combo | `dinov2_b/clip_seq+contact/realistic` | -0.024 ± 0.025 | 0.409 | 1.000 ± 0.000 | 0.004 | -0.000 | -0.000 | n/a | 1.00 | 3 |  |
| combo | `vjepa2_l/clip_mean_L16+contact/realistic` | -0.024 ± 0.025 | 0.409 | 1.000 ± 0.000 | 0.003 | -0.000 | -0.000 | n/a | 1.00 | 3 | ★ |
| combo | `vjepa2_l/clip_seq+contact/realistic` | -0.024 ± 0.025 | 0.409 | 1.000 ± 0.000 | 0.004 | -0.000 | -0.000 | n/a | 1.00 | 3 |  |
| state | `oracle_xc` | -0.024 ± 0.025 | 0.409 | 1.000 ± 0.000 | 0.000 | +0.000 | +0.000 | n/a | 1.00 | 3 | **ORACLE** ★ |
| state | `oracle_x` | -0.024 ± 0.025 | 0.409 | 0.999 ± 0.000 | 0.006 | +0.000 | -0.001 | n/a | 1.00 | 3 | **ORACLE** pixel-quantised oracle (not an upper bound) |
| state | `oracle_fit` | -0.024 ± 0.025 | 0.409 | 1.000 ± 0.000 | 0.000 | +0.000 | +0.000 | n/a | 1.00 | 3 | **ORACLE** |
| contact | `raw` | 0.971 ± 0.003 | 0.068 | 1.000 ± 0.000 | 0.000 | +0.996 | +0.000 | n/a | n/a (>1) | 3 | **ORACLE** ★ |
| contact | `realistic` | -0.024 ± 0.025 | 0.409 | 0.999 ± 0.000 | 0.007 | -0.000 | -0.001 | n/a | 1.00 | 3 |  |
| contact | `noise_000` | -0.024 ± 0.025 | 0.409 | 1.000 ± 0.000 | 0.000 | -0.000 | +0.000 | n/a | 1.00 | 3 |  |
| contact | `noise_010` | -0.024 ± 0.025 | 0.409 | 0.994 ± 0.000 | 0.013 | -0.000 | -0.006 | n/a | 0.99 | 3 |  |
| contact | `noise_030` | -0.024 ± 0.025 | 0.409 | 0.964 ± 0.004 | 0.033 | -0.000 | -0.036 | n/a | 0.96 | 3 |  |
| contact | `noise_100` | -0.024 ± 0.025 | 0.409 | 0.755 ± 0.014 | 0.088 | -0.000 | -0.244 | n/a | 0.76 | 3 |  |
| pixels [shuffled labels] | `static` | -0.024 ± 0.025 | 0.409 | -0.001 ± 0.001 | 0.177 | -0.000 | -1.001 | n/a | -0.00 | 3 | control |
| pixels [shuffled labels] | `last` | -0.024 ± 0.025 | 0.409 | 0.008 ± 0.015 | 0.176 | +0.000 | -0.992 | n/a | 0.01 | 3 | control |
| pixels [shuffled labels] | `clip_seq` | -0.024 ± 0.025 | 0.409 | 0.010 ± 0.020 | 0.176 | -0.000 | -0.990 | n/a | 0.01 | 3 | control |
| track [shuffled labels] | `clip` | -0.025 ± 0.023 | 0.409 | -0.002 ± 0.001 | 0.177 | -0.001 | -1.001 | n/a | -0.00 | 3 | control |
| dinov2_b_random [shuffled labels] | `static` | -0.024 ± 0.025 | 0.409 | -0.001 ± 0.001 | 0.177 | -0.000 | -1.001 | n/a | -0.00 | 3 | control |
| dinov2_b_random [shuffled labels] | `last` | -0.047 ± 0.036 | 0.414 | -0.005 ± 0.018 | 0.178 | -0.023 | -1.005 | n/a | -0.01 | 3 | control |
| dinov2_b_random [shuffled labels] | `clip_mean` | -0.045 ± 0.033 | 0.413 | -0.005 ± 0.021 | 0.178 | -0.021 | -1.005 | n/a | -0.00 | 3 | control |
| dinov2_b_random [shuffled labels] | `clip_seq` | -0.031 ± 0.018 | 0.410 | 0.005 ± 0.029 | 0.177 | -0.007 | -0.995 | n/a | 0.01 | 3 | control |
| dinov2_b [shuffled labels] | `static` | -0.024 ± 0.025 | 0.409 | -0.001 ± 0.001 | 0.177 | -0.000 | -1.001 | n/a | -0.00 | 3 | control |
| dinov2_b [shuffled labels] | `last` | -0.036 ± 0.025 | 0.411 | 0.002 ± 0.010 | 0.177 | -0.011 | -0.998 | n/a | 0.00 | 3 | control |
| dinov2_b [shuffled labels] | `clip_mean` | -0.033 ± 0.024 | 0.411 | 0.004 ± 0.010 | 0.177 | -0.009 | -0.996 | n/a | 0.00 | 3 | control |
| dinov2_b [shuffled labels] | `clip_seq` | -0.027 ± 0.013 | 0.409 | 0.007 ± 0.032 | 0.176 | -0.002 | -0.993 | n/a | 0.01 | 3 | control |
| vjepa2_l_random [shuffled labels] | `static` | -0.024 ± 0.025 | 0.409 | -0.001 ± 0.001 | 0.177 | -0.000 | -1.001 | n/a | -0.00 | 3 | control |
| vjepa2_l_random [shuffled labels] | `clip_mean` | -0.042 ± 0.027 | 0.412 | -0.007 ± 0.019 | 0.178 | -0.017 | -1.007 | n/a | -0.01 | 3 | control |
| vjepa2_l_random [shuffled labels] | `clip_seq` | -0.042 ± 0.023 | 0.412 | 0.006 ± 0.028 | 0.177 | -0.018 | -0.994 | n/a | 0.01 | 3 | control |
| vjepa2_l [shuffled labels] | `static` | -0.024 ± 0.025 | 0.409 | -0.001 ± 0.001 | 0.177 | -0.000 | -1.001 | n/a | -0.00 | 3 | control |
| vjepa2_l [shuffled labels] | `clip_mean_L08` | -0.030 ± 0.024 | 0.410 | -0.006 ± 0.007 | 0.178 | -0.005 | -1.006 | n/a | -0.01 | 3 | control |
| vjepa2_l [shuffled labels] | `clip_mean_L16` | -0.024 ± 0.024 | 0.409 | -0.005 ± 0.008 | 0.178 | +0.000 | -1.005 | n/a | -0.01 | 3 | control |
| vjepa2_l [shuffled labels] | `clip_mean` | -0.024 ± 0.024 | 0.409 | -0.003 ± 0.005 | 0.177 | +0.000 | -1.003 | n/a | -0.00 | 3 | control |
| vjepa2_l [shuffled labels] | `clip_seq_L08` | -0.029 ± 0.014 | 0.410 | -0.009 ± 0.032 | 0.178 | -0.005 | -1.009 | n/a | -0.01 | 3 | control |
| vjepa2_l [shuffled labels] | `clip_seq_L16` | -0.029 ± 0.016 | 0.410 | -0.002 ± 0.029 | 0.177 | -0.005 | -1.002 | n/a | -0.00 | 3 | control |
| vjepa2_l [shuffled labels] | `clip_seq` | -0.026 ± 0.016 | 0.409 | 0.005 ± 0.027 | 0.177 | -0.002 | -0.995 | n/a | 0.00 | 3 | control |
| cosmos_cv8 [shuffled labels] | `clip_mean` | -0.024 ± 0.023 | 0.409 | -0.001 ± 0.001 | 0.177 | +0.000 | -1.001 | n/a | -0.00 | 3 | control |
| cosmos_cv8 [shuffled labels] | `clip_seq` | -0.025 ± 0.024 | 0.409 | -0.001 ± 0.001 | 0.177 | -0.001 | -1.001 | n/a | -0.00 | 3 | control |
| cosmos_cv8 [shuffled labels] | `clip_grid4` | -0.025 ± 0.024 | 0.409 | -0.001 ± 0.012 | 0.177 | -0.000 | -1.001 | n/a | -0.00 | 3 | control |
| combo [shuffled labels] | `dinov2_b/clip_seq+contact/realistic` | -0.034 ± 0.025 | 0.411 | -0.001 ± 0.001 | 0.177 | -0.010 | -1.001 | n/a | -0.00 | 3 | control |
| combo [shuffled labels] | `vjepa2_l/clip_mean_L16+contact/realistic` | -0.031 ± 0.025 | 0.410 | -0.001 ± 0.001 | 0.177 | -0.007 | -1.001 | n/a | -0.00 | 3 | control |
| combo [shuffled labels] | `vjepa2_l/clip_seq+contact/realistic` | -0.029 ± 0.024 | 0.410 | -0.001 ± 0.001 | 0.177 | -0.004 | -1.001 | n/a | -0.00 | 3 | control |
| state [shuffled labels] | `oracle_xc` | -0.025 ± 0.024 | 0.409 | -0.001 ± 0.001 | 0.177 | -0.000 | -1.001 | n/a | -0.00 | 3 | **ORACLE** control |
| state [shuffled labels] | `oracle_x` | -0.024 ± 0.024 | 0.409 | -0.001 ± 0.001 | 0.177 | +0.001 | -1.001 | n/a | -0.00 | 3 | **ORACLE** pixel-quantised oracle (not an upper bound) control |
| state [shuffled labels] | `oracle_fit` | -0.024 ± 0.025 | 0.409 | -0.001 ± 0.001 | 0.177 | -0.000 | -1.001 | n/a | -0.00 | 3 | **ORACLE** control |
| contact [shuffled labels] | `raw` | -0.031 ± 0.026 | 0.410 | -0.014 ± 0.021 | 0.178 | -0.006 | -1.014 | n/a | -0.01 | 3 | **ORACLE** control |
| contact [shuffled labels] | `realistic` | -0.031 ± 0.026 | 0.410 | -0.001 ± 0.001 | 0.177 | -0.007 | -1.001 | n/a | -0.00 | 3 | control |
| contact [shuffled labels] | `noise_000` | -0.024 ± 0.024 | 0.409 | -0.001 ± 0.001 | 0.177 | +0.000 | -1.001 | n/a | -0.00 | 3 | control |
| contact [shuffled labels] | `noise_010` | -0.026 ± 0.023 | 0.409 | -0.001 ± 0.001 | 0.177 | -0.001 | -1.001 | n/a | -0.00 | 3 | control |
| contact [shuffled labels] | `noise_030` | -0.024 ± 0.024 | 0.409 | -0.001 ± 0.001 | 0.177 | -0.000 | -1.001 | n/a | -0.00 | 3 | control |
| contact [shuffled labels] | `noise_100` | -0.025 ± 0.024 | 0.409 | 0.002 ± 0.005 | 0.177 | -0.001 | -0.998 | n/a | 0.00 | 3 | control |

## Identifiability check: slide (log m should be ≈ 0)

n/a (no slide probes.csv)

