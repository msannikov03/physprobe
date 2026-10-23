# PhysProbe results: `push`

- Splits — s0: n_train=1400, n_val=300, n_test=300, n_excluded=0; s1: n_train=1400, n_val=300, n_test=300, n_excluded=0; s2: n_train=1400, n_val=300, n_test=300, n_excluded=0
- git commit: `c5fe74b`
- runs: 2026-10-23T00:59:41+00:00 … 2026-10-23T00:59:41+00:00
- R² = test R², mean ± std over seeds (ridge probe unless stated). RMSE for log m is log-RMSE.
- Δ = R² − R²(pixels/clip_seq); frac = R² / R²(state/oracle_xc) (per target); a reference that is beaten (frac > 1.001) is printed as `n/a (>1)`, a tie as 1.00; `state/oracle_xc` = unquantised x(t), the linear x(t) read-out reference: a ceiling for μ, not for log m (a linear probe on x(t) does not invert d = J²/(2m²μg), so frac for log m is not a fraction of a ceiling); `state/oracle_x` (1-px quantised) is shown for information only.
- ★ = mode chosen by validation R² (mean r2_val over targets; candidates clip_mean*, clip_seq*, clip_grid*), used in headline.png.

**Gates:** mean PASS, static PASS, null PASS, oracle PASS, slide PASS, poscheck PASS, contact_label PASS; plan v1 rules (informational): shuffle_v1 PASS, oracle_v1 FAIL (details in `gates.md`)

**Mode / layer selection (by val, not test):** `track` → `clip` (val); `dinov2_b_random` → `clip_seq` (val); `dinov2_b` → `clip_seq` (val); `vjepa2_l_random` → `clip_seq` (val); `vjepa2_l` → `clip_mean_L16` (val); `cosmos_cv8` → `clip_grid4` (val); `combo` → `vjepa2_l/clip_mean_L16+contact/realistic` (val); best `vjepa2_l` layer for clip_mean: L16 (`clip_mean_L16`, by val); best `vjepa2_l` layer for clip_seq: L24 (last) (`clip_seq`, by val)

## Main table (ridge)

| Representation | Mode | log m R² | log m RMSE | μ R² | μ RMSE | Δ vs pixels/clip_seq (log m) | Δ vs pixels/clip_seq (μ) | frac of state/oracle_xc (log m) | frac of state/oracle_xc (μ) | seeds | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mean (predict train mean) | - | -0.006 ± 0.010 | 0.404 | -0.003 ± 0.001 | 0.173 | -1.005 | -0.985 | -0.01 | -0.00 | 3 |  |
| pixels | `static` | -0.006 ± 0.010 | 0.404 | -0.003 ± 0.001 | 0.173 | -1.005 | -0.985 | -0.01 | -0.00 | 3 |  |
| pixels | `last` | 0.819 ± 0.010 | 0.172 | 0.205 ± 0.055 | 0.154 | -0.180 | -0.777 | 0.84 | 0.21 | 3 |  |
| pixels | `clip_seq` | 0.999 ± 0.000 | 0.013 | 0.982 ± 0.002 | 0.023 | +0.000 | +0.000 | n/a (>1) | 0.98 | 3 | ★ |
| track | `clip` | 0.974 ± 0.001 | 0.065 | 0.939 ± 0.009 | 0.043 | -0.025 | -0.043 | 1.00 | 0.94 | 3 | ★ |
| dinov2_b_random | `static` | -0.006 ± 0.010 | 0.404 | -0.003 ± 0.001 | 0.173 | -1.005 | -0.985 | -0.01 | -0.00 | 3 |  |
| dinov2_b_random | `last` | 0.811 ± 0.013 | 0.175 | 0.180 ± 0.068 | 0.157 | -0.188 | -0.802 | 0.83 | 0.18 | 3 |  |
| dinov2_b_random | `clip_mean` | 0.837 ± 0.009 | 0.163 | 0.169 ± 0.062 | 0.158 | -0.162 | -0.813 | 0.86 | 0.17 | 3 |  |
| dinov2_b_random | `clip_seq` | 0.956 ± 0.007 | 0.085 | 0.684 ± 0.026 | 0.097 | -0.043 | -0.298 | 0.98 | 0.68 | 3 | ★ |
| dinov2_b | `static` | -0.006 ± 0.010 | 0.404 | -0.003 ± 0.001 | 0.173 | -1.005 | -0.985 | -0.01 | -0.00 | 3 |  |
| dinov2_b | `last` | 0.815 ± 0.010 | 0.174 | 0.191 ± 0.064 | 0.156 | -0.184 | -0.791 | 0.83 | 0.19 | 3 |  |
| dinov2_b | `clip_mean` | 0.838 ± 0.008 | 0.162 | 0.176 ± 0.060 | 0.157 | -0.161 | -0.807 | 0.86 | 0.18 | 3 |  |
| dinov2_b | `clip_seq` | 0.983 ± 0.001 | 0.053 | 0.842 ± 0.007 | 0.069 | -0.016 | -0.140 | n/a (>1) | 0.84 | 3 | ★ |
| vjepa2_l_random | `static` | -0.006 ± 0.010 | 0.404 | -0.003 ± 0.001 | 0.173 | -1.005 | -0.985 | -0.01 | -0.00 | 3 |  |
| vjepa2_l_random | `clip_mean` | 0.979 ± 0.011 | 0.057 | 0.850 ± 0.046 | 0.067 | -0.020 | -0.132 | n/a (>1) | 0.85 | 3 |  |
| vjepa2_l_random | `clip_seq` | 0.997 ± 0.000 | 0.022 | 0.883 ± 0.004 | 0.059 | -0.002 | -0.099 | n/a (>1) | 0.88 | 3 | ★ |
| vjepa2_l | `static` | -0.006 ± 0.010 | 0.404 | -0.003 ± 0.001 | 0.173 | -1.005 | -0.985 | -0.01 | -0.00 | 3 |  |
| vjepa2_l | `clip_mean_L08` | 0.994 ± 0.000 | 0.032 | 0.958 ± 0.002 | 0.036 | -0.005 | -0.025 | n/a (>1) | 0.96 | 3 |  |
| vjepa2_l | `clip_mean_L16` | 0.997 ± 0.000 | 0.023 | 0.977 ± 0.001 | 0.027 | -0.002 | -0.006 | n/a (>1) | 0.98 | 3 | ★ |
| vjepa2_l | `clip_mean` | 0.996 ± 0.000 | 0.026 | 0.970 ± 0.001 | 0.030 | -0.003 | -0.013 | n/a (>1) | 0.97 | 3 |  |
| vjepa2_l | `clip_seq_L08` | 0.993 ± 0.000 | 0.035 | 0.945 ± 0.002 | 0.041 | -0.006 | -0.038 | n/a (>1) | 0.94 | 3 |  |
| vjepa2_l | `clip_seq_L16` | 0.995 ± 0.000 | 0.028 | 0.967 ± 0.001 | 0.031 | -0.004 | -0.015 | n/a (>1) | 0.97 | 3 |  |
| vjepa2_l | `clip_seq` | 0.996 ± 0.000 | 0.026 | 0.967 ± 0.002 | 0.031 | -0.003 | -0.015 | n/a (>1) | 0.97 | 3 |  |
| cosmos_cv8 | `clip_mean` | 0.680 ± 0.009 | 0.228 | 0.176 ± 0.030 | 0.157 | -0.319 | -0.806 | 0.70 | 0.18 | 3 |  |
| cosmos_cv8 | `clip_seq` | 0.908 ± 0.001 | 0.122 | 0.413 ± 0.022 | 0.133 | -0.091 | -0.569 | 0.93 | 0.41 | 3 |  |
| cosmos_cv8 | `clip_grid4` | 0.943 ± 0.002 | 0.096 | 0.682 ± 0.008 | 0.098 | -0.056 | -0.300 | 0.97 | 0.68 | 3 | ★ |
| combo | `dinov2_b/clip_seq+contact/realistic` | 0.992 ± 0.000 | 0.036 | 0.948 ± 0.003 | 0.039 | -0.007 | -0.034 | n/a (>1) | 0.95 | 3 |  |
| combo | `vjepa2_l/clip_mean_L16+contact/realistic` | 0.997 ± 0.000 | 0.022 | 0.982 ± 0.001 | 0.023 | -0.002 | -0.000 | n/a (>1) | 0.98 | 3 | ★ |
| combo | `vjepa2_l/clip_seq+contact/realistic` | 0.995 ± 0.000 | 0.027 | 0.975 ± 0.003 | 0.027 | -0.004 | -0.007 | n/a (>1) | 0.98 | 3 |  |
| state | `oracle_xc` | 0.976 ± 0.001 | 0.062 | 1.000 ± 0.000 | 0.001 | -0.023 | +0.018 | 1.00 | 1.00 | 3 | **ORACLE** ★ |
| state | `oracle_x` | 0.974 ± 0.001 | 0.065 | 0.865 ± 0.013 | 0.063 | -0.025 | -0.117 | 1.00 | 0.87 | 3 | **ORACLE** pixel-quantised oracle (not an upper bound) |
| state | `oracle_fit` | 1.000 ± 0.000 | 0.000 | 1.000 ± 0.000 | 0.000 | +0.001 | +0.018 | n/a (>1) | 1.00 | 3 | **ORACLE** |
| contact | `raw` | 1.000 ± 0.000 | 0.007 | 1.000 ± 0.000 | 0.000 | +0.001 | +0.018 | n/a (>1) | 1.00 | 3 | **ORACLE** ★ |
| contact | `realistic` | 0.521 ± 0.038 | 0.279 | 0.452 ± 0.043 | 0.128 | -0.478 | -0.530 | 0.53 | 0.45 | 3 |  |
| contact | `noise_000` | 0.527 ± 0.036 | 0.277 | 0.457 ± 0.041 | 0.128 | -0.472 | -0.525 | 0.54 | 0.46 | 3 |  |
| contact | `noise_010` | 0.519 ± 0.034 | 0.280 | 0.450 ± 0.044 | 0.128 | -0.480 | -0.533 | 0.53 | 0.45 | 3 |  |
| contact | `noise_030` | 0.499 ± 0.029 | 0.285 | 0.434 ± 0.050 | 0.130 | -0.500 | -0.548 | 0.51 | 0.43 | 3 |  |
| contact | `noise_100` | 0.378 ± 0.021 | 0.318 | 0.336 ± 0.065 | 0.141 | -0.621 | -0.646 | 0.39 | 0.34 | 3 |  |
| pixels [shuffled labels] | `static` | -0.006 ± 0.010 | 0.404 | -0.003 ± 0.001 | 0.173 | -1.005 | -0.985 | -0.01 | -0.00 | 3 | control |
| pixels [shuffled labels] | `last` | -0.022 ± 0.055 | 0.407 | 0.004 ± 0.011 | 0.173 | -1.021 | -0.978 | -0.02 | 0.00 | 3 | control |
| pixels [shuffled labels] | `clip_seq` | -0.021 ± 0.061 | 0.407 | 0.002 ± 0.010 | 0.173 | -1.020 | -0.980 | -0.02 | 0.00 | 3 | control |
| track [shuffled labels] | `clip` | -0.024 ± 0.028 | 0.408 | -0.002 ± 0.002 | 0.173 | -1.023 | -0.984 | -0.02 | -0.00 | 3 | control |
| dinov2_b_random [shuffled labels] | `static` | -0.006 ± 0.010 | 0.404 | -0.003 ± 0.001 | 0.173 | -1.005 | -0.985 | -0.01 | -0.00 | 3 | control |
| dinov2_b_random [shuffled labels] | `last` | -0.021 ± 0.038 | 0.407 | -0.003 ± 0.003 | 0.173 | -1.019 | -0.986 | -0.02 | -0.00 | 3 | control |
| dinov2_b_random [shuffled labels] | `clip_mean` | -0.023 ± 0.044 | 0.407 | -0.003 ± 0.002 | 0.173 | -1.022 | -0.985 | -0.02 | -0.00 | 3 | control |
| dinov2_b_random [shuffled labels] | `clip_seq` | -0.018 ± 0.033 | 0.407 | -0.009 ± 0.002 | 0.174 | -1.017 | -0.991 | -0.02 | -0.01 | 3 | control |
| dinov2_b [shuffled labels] | `static` | -0.006 ± 0.010 | 0.404 | -0.003 ± 0.001 | 0.173 | -1.005 | -0.985 | -0.01 | -0.00 | 3 | control |
| dinov2_b [shuffled labels] | `last` | -0.017 ± 0.037 | 0.406 | -0.005 ± 0.004 | 0.174 | -1.016 | -0.988 | -0.02 | -0.01 | 3 | control |
| dinov2_b [shuffled labels] | `clip_mean` | -0.022 ± 0.060 | 0.407 | -0.005 ± 0.004 | 0.173 | -1.021 | -0.987 | -0.02 | -0.00 | 3 | control |
| dinov2_b [shuffled labels] | `clip_seq` | -0.017 ± 0.062 | 0.406 | -0.012 ± 0.015 | 0.174 | -1.016 | -0.994 | -0.02 | -0.01 | 3 | control |
| vjepa2_l_random [shuffled labels] | `static` | -0.006 ± 0.010 | 0.404 | -0.003 ± 0.001 | 0.173 | -1.005 | -0.985 | -0.01 | -0.00 | 3 | control |
| vjepa2_l_random [shuffled labels] | `clip_mean` | -0.030 ± 0.054 | 0.409 | -0.003 ± 0.001 | 0.173 | -1.029 | -0.985 | -0.03 | -0.00 | 3 | control |
| vjepa2_l_random [shuffled labels] | `clip_seq` | -0.032 ± 0.046 | 0.409 | -0.012 ± 0.003 | 0.174 | -1.031 | -0.994 | -0.03 | -0.01 | 3 | control |
| vjepa2_l [shuffled labels] | `static` | -0.006 ± 0.010 | 0.404 | -0.003 ± 0.001 | 0.173 | -1.005 | -0.985 | -0.01 | -0.00 | 3 | control |
| vjepa2_l [shuffled labels] | `clip_mean_L08` | -0.015 ± 0.033 | 0.406 | -0.005 ± 0.003 | 0.173 | -1.014 | -0.987 | -0.02 | -0.00 | 3 | control |
| vjepa2_l [shuffled labels] | `clip_mean_L16` | -0.014 ± 0.036 | 0.406 | -0.006 ± 0.011 | 0.174 | -1.013 | -0.989 | -0.01 | -0.01 | 3 | control |
| vjepa2_l [shuffled labels] | `clip_mean` | -0.016 ± 0.043 | 0.406 | -0.005 ± 0.009 | 0.173 | -1.015 | -0.987 | -0.02 | -0.00 | 3 | control |
| vjepa2_l [shuffled labels] | `clip_seq_L08` | -0.008 ± 0.071 | 0.405 | -0.015 ± 0.016 | 0.174 | -1.007 | -0.997 | -0.01 | -0.01 | 3 | control |
| vjepa2_l [shuffled labels] | `clip_seq_L16` | -0.007 ± 0.069 | 0.404 | -0.022 ± 0.025 | 0.175 | -1.006 | -1.004 | -0.01 | -0.02 | 3 | control |
| vjepa2_l [shuffled labels] | `clip_seq` | -0.012 ± 0.074 | 0.405 | -0.012 ± 0.019 | 0.174 | -1.011 | -0.994 | -0.01 | -0.01 | 3 | control |
| cosmos_cv8 [shuffled labels] | `clip_mean` | -0.015 ± 0.063 | 0.406 | -0.003 ± 0.001 | 0.173 | -1.014 | -0.985 | -0.02 | -0.00 | 3 | control |
| cosmos_cv8 [shuffled labels] | `clip_seq` | 0.002 ± 0.049 | 0.402 | -0.003 ± 0.001 | 0.173 | -0.997 | -0.985 | 0.00 | -0.00 | 3 | control |
| cosmos_cv8 [shuffled labels] | `clip_grid4` | -0.005 ± 0.046 | 0.404 | -0.002 ± 0.007 | 0.173 | -1.004 | -0.984 | -0.01 | -0.00 | 3 | control |
| combo [shuffled labels] | `dinov2_b/clip_seq+contact/realistic` | -0.015 ± 0.055 | 0.406 | -0.010 ± 0.013 | 0.174 | -1.014 | -0.992 | -0.02 | -0.01 | 3 | control |
| combo [shuffled labels] | `vjepa2_l/clip_mean_L16+contact/realistic` | -0.021 ± 0.034 | 0.407 | -0.009 ± 0.011 | 0.174 | -1.020 | -0.991 | -0.02 | -0.01 | 3 | control |
| combo [shuffled labels] | `vjepa2_l/clip_seq+contact/realistic` | -0.017 ± 0.042 | 0.406 | -0.008 ± 0.010 | 0.174 | -1.016 | -0.991 | -0.02 | -0.01 | 3 | control |
| state [shuffled labels] | `oracle_xc` | -0.023 ± 0.036 | 0.407 | -0.021 ± 0.032 | 0.175 | -1.022 | -1.003 | -0.02 | -0.02 | 3 | **ORACLE** control |
| state [shuffled labels] | `oracle_x` | -0.020 ± 0.035 | 0.407 | -0.004 ± 0.003 | 0.173 | -1.019 | -0.986 | -0.02 | -0.00 | 3 | **ORACLE** pixel-quantised oracle (not an upper bound) control |
| state [shuffled labels] | `oracle_fit` | -0.005 ± 0.023 | 0.404 | -0.009 ± 0.009 | 0.174 | -1.004 | -0.991 | -0.01 | -0.01 | 3 | **ORACLE** control |
| contact [shuffled labels] | `raw` | 0.002 ± 0.060 | 0.402 | -0.015 ± 0.022 | 0.174 | -0.997 | -0.997 | 0.00 | -0.01 | 3 | **ORACLE** control |
| contact [shuffled labels] | `realistic` | -0.020 ± 0.034 | 0.407 | -0.009 ± 0.011 | 0.174 | -1.019 | -0.991 | -0.02 | -0.01 | 3 | control |
| contact [shuffled labels] | `noise_000` | -0.027 ± 0.026 | 0.408 | -0.006 ± 0.007 | 0.174 | -1.026 | -0.989 | -0.03 | -0.01 | 3 | control |
| contact [shuffled labels] | `noise_010` | -0.019 ± 0.023 | 0.407 | -0.008 ± 0.009 | 0.174 | -1.018 | -0.990 | -0.02 | -0.01 | 3 | control |
| contact [shuffled labels] | `noise_030` | -0.013 ± 0.018 | 0.405 | -0.004 ± 0.003 | 0.173 | -1.012 | -0.986 | -0.01 | -0.00 | 3 | control |
| contact [shuffled labels] | `noise_100` | -0.006 ± 0.010 | 0.404 | -0.004 ± 0.001 | 0.173 | -1.005 | -0.986 | -0.01 | -0.00 | 3 | control |

## Identifiability check: slide (log m should be ≈ 0)

Dataset `slide`. Splits — s0: n_train=700, n_val=150, n_test=150, n_excluded=0; s1: n_train=700, n_val=150, n_test=150, n_excluded=0; s2: n_train=700, n_val=150, n_test=150, n_excluded=0 git commit: `c5fe74b`

| Representation | Mode | probe | log m R² | log m RMSE | note |
|---|---|---|---|---|---|
| mean | `-` | mean | -0.024 ± 0.025 | 0.409 |  |
| pixels | `static` | ridge | -0.024 ± 0.025 | 0.409 |  |
| pixels | `last` | ridge | -0.024 ± 0.025 | 0.409 |  |
| pixels | `clip_seq` | ridge | -0.024 ± 0.025 | 0.409 |  |
| track | `clip` | ridge | -0.024 ± 0.025 | 0.409 |  |
| dinov2_b_random | `static` | ridge | -0.024 ± 0.025 | 0.409 |  |
| dinov2_b_random | `last` | ridge | -0.025 ± 0.024 | 0.409 |  |
| dinov2_b_random | `clip_mean` | ridge | -0.026 ± 0.024 | 0.409 |  |
| dinov2_b_random | `clip_seq` | ridge | -0.028 ± 0.023 | 0.410 |  |
| dinov2_b | `static` | ridge | -0.024 ± 0.025 | 0.409 |  |
| dinov2_b | `last` | ridge | -0.025 ± 0.024 | 0.409 |  |
| dinov2_b | `clip_mean` | ridge | -0.025 ± 0.024 | 0.409 |  |
| dinov2_b | `clip_seq` | ridge | -0.031 ± 0.031 | 0.410 |  |
| vjepa2_l_random | `static` | ridge | -0.024 ± 0.025 | 0.409 |  |
| vjepa2_l_random | `clip_mean` | ridge | -0.026 ± 0.023 | 0.409 |  |
| vjepa2_l_random | `clip_seq` | ridge | -0.031 ± 0.021 | 0.410 |  |
| vjepa2_l | `static` | ridge | -0.024 ± 0.025 | 0.409 |  |
| vjepa2_l | `clip_mean_L08` | ridge | -0.025 ± 0.026 | 0.409 |  |
| vjepa2_l | `clip_mean_L16` | ridge | -0.025 ± 0.025 | 0.409 |  |
| vjepa2_l | `clip_mean` | ridge | -0.025 ± 0.024 | 0.409 |  |
| vjepa2_l | `clip_seq_L08` | ridge | -0.032 ± 0.037 | 0.410 |  |
| vjepa2_l | `clip_seq_L16` | ridge | -0.030 ± 0.033 | 0.410 |  |
| vjepa2_l | `clip_seq` | ridge | -0.029 ± 0.026 | 0.410 |  |
| cosmos_cv8 | `clip_mean` | ridge | -0.025 ± 0.024 | 0.409 |  |
| cosmos_cv8 | `clip_seq` | ridge | -0.024 ± 0.024 | 0.409 |  |
| cosmos_cv8 | `clip_grid4` | ridge | -0.024 ± 0.025 | 0.409 |  |
| combo | `dinov2_b/clip_seq+contact/realistic` | ridge | -0.024 ± 0.025 | 0.409 |  |
| combo | `vjepa2_l/clip_mean_L16+contact/realistic` | ridge | -0.024 ± 0.025 | 0.409 |  |
| combo | `vjepa2_l/clip_seq+contact/realistic` | ridge | -0.024 ± 0.025 | 0.409 |  |
| state | `oracle_xc` | ridge | -0.024 ± 0.025 | 0.409 | state oracle (reported separately) |
| state | `oracle_x` | ridge | -0.024 ± 0.025 | 0.409 | state oracle (reported separately) |
| state | `oracle_fit` | ridge | -0.024 ± 0.025 | 0.409 | state oracle (reported separately) |
| contact | `raw` | ridge | 0.971 ± 0.003 | 0.068 | **ORACLE**; leaks m via N=mg (by construction) |
| contact | `realistic` | ridge | -0.024 ± 0.025 | 0.409 |  |
| contact | `noise_000` | ridge | -0.024 ± 0.025 | 0.409 |  |
| contact | `noise_010` | ridge | -0.024 ± 0.025 | 0.409 |  |
| contact | `noise_030` | ridge | -0.024 ± 0.025 | 0.409 |  |
| contact | `noise_100` | ridge | -0.024 ± 0.025 | 0.409 |  |
| pixels | `static` | shuffle | -0.024 ± 0.025 | 0.409 |  |
| pixels | `last` | shuffle | -0.024 ± 0.025 | 0.409 |  |
| pixels | `clip_seq` | shuffle | -0.024 ± 0.025 | 0.409 |  |
| track | `clip` | shuffle | -0.025 ± 0.023 | 0.409 |  |
| dinov2_b_random | `static` | shuffle | -0.024 ± 0.025 | 0.409 |  |
| dinov2_b_random | `last` | shuffle | -0.047 ± 0.036 | 0.414 |  |
| dinov2_b_random | `clip_mean` | shuffle | -0.045 ± 0.033 | 0.413 |  |
| dinov2_b_random | `clip_seq` | shuffle | -0.031 ± 0.018 | 0.410 |  |
| dinov2_b | `static` | shuffle | -0.024 ± 0.025 | 0.409 |  |
| dinov2_b | `last` | shuffle | -0.036 ± 0.025 | 0.411 |  |
| dinov2_b | `clip_mean` | shuffle | -0.033 ± 0.024 | 0.411 |  |
| dinov2_b | `clip_seq` | shuffle | -0.027 ± 0.013 | 0.409 |  |
| vjepa2_l_random | `static` | shuffle | -0.024 ± 0.025 | 0.409 |  |
| vjepa2_l_random | `clip_mean` | shuffle | -0.042 ± 0.027 | 0.412 |  |
| vjepa2_l_random | `clip_seq` | shuffle | -0.042 ± 0.023 | 0.412 |  |
| vjepa2_l | `static` | shuffle | -0.024 ± 0.025 | 0.409 |  |
| vjepa2_l | `clip_mean_L08` | shuffle | -0.030 ± 0.024 | 0.410 |  |
| vjepa2_l | `clip_mean_L16` | shuffle | -0.024 ± 0.024 | 0.409 |  |
| vjepa2_l | `clip_mean` | shuffle | -0.024 ± 0.024 | 0.409 |  |
| vjepa2_l | `clip_seq_L08` | shuffle | -0.029 ± 0.014 | 0.410 |  |
| vjepa2_l | `clip_seq_L16` | shuffle | -0.029 ± 0.016 | 0.410 |  |
| vjepa2_l | `clip_seq` | shuffle | -0.026 ± 0.016 | 0.409 |  |
| cosmos_cv8 | `clip_mean` | shuffle | -0.024 ± 0.023 | 0.409 |  |
| cosmos_cv8 | `clip_seq` | shuffle | -0.025 ± 0.024 | 0.409 |  |
| cosmos_cv8 | `clip_grid4` | shuffle | -0.025 ± 0.024 | 0.409 |  |
| combo | `dinov2_b/clip_seq+contact/realistic` | shuffle | -0.034 ± 0.025 | 0.411 |  |
| combo | `vjepa2_l/clip_mean_L16+contact/realistic` | shuffle | -0.031 ± 0.025 | 0.410 |  |
| combo | `vjepa2_l/clip_seq+contact/realistic` | shuffle | -0.029 ± 0.024 | 0.410 |  |
| state | `oracle_xc` | shuffle | -0.025 ± 0.024 | 0.409 | state oracle (reported separately) |
| state | `oracle_x` | shuffle | -0.024 ± 0.024 | 0.409 | state oracle (reported separately) |
| state | `oracle_fit` | shuffle | -0.024 ± 0.025 | 0.409 | state oracle (reported separately) |
| contact | `raw` | shuffle | -0.031 ± 0.026 | 0.410 | **ORACLE**; leaks m via N=mg (by construction) |
| contact | `realistic` | shuffle | -0.031 ± 0.026 | 0.410 |  |
| contact | `noise_000` | shuffle | -0.024 ± 0.024 | 0.409 |  |
| contact | `noise_010` | shuffle | -0.026 ± 0.023 | 0.409 |  |
| contact | `noise_030` | shuffle | -0.024 ± 0.024 | 0.409 |  |
| contact | `noise_100` | shuffle | -0.025 ± 0.024 | 0.409 |  |

