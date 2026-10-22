# Quick-look linear probes (ridge), test R², mean ± std over split seeds

Std is the sample std (ddof 1) over the seeds. Ridge = RidgeCV(alphas=logspace(-3,5,17)) on train after train-only z-scoring; inputs above 4096 dims go through PCA to 512 (train-only, randomised, seed 0).

## push (n_train 1400, n_test 300, seeds [0, 1, 2]; sorted by R²(mu))

| input | dim | R²(log_m) | R²(mu) |
|---|---|---|---|
| state/oracle_fit | 2 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| contact/raw | 192 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| state/oracle_xc | 64 | 0.976 ± 0.001 | 1.000 ± 0.000 |
| vjepa2_l/clip_seq + contact/realistic | 32960→512 | 0.997 ± 0.000 | 0.984 ± 0.002 |
| pixels/clip_seq | 8192→512 | 0.999 ± 0.000 | 0.982 ± 0.001 |
| vjepa2_l/clip_seq_L16 | 32768→512 | 0.997 ± 0.000 | 0.979 ± 0.002 |
| vjepa2_l/clip_mean_L16 | 1024 | 0.997 ± 0.000 | 0.977 ± 0.000 |
| vjepa2_l/clip_seq | 32768→512 | 0.997 ± 0.000 | 0.975 ± 0.001 |
| dinov2_b/clip_seq + contact/realistic | 24768→512 | 0.997 ± 0.000 | 0.972 ± 0.002 |
| vjepa2_l/clip_mean | 1024 | 0.996 ± 0.000 | 0.969 ± 0.002 |
| vjepa2_l/clip_seq_L08 | 32768→512 | 0.995 ± 0.000 | 0.966 ± 0.001 |
| vjepa2_l/clip_mean_L08 | 1024 | 0.994 ± 0.000 | 0.958 ± 0.002 |
| track/clip | 128 | 0.974 ± 0.001 | 0.938 ± 0.009 |
| vjepa2_l_random/clip_seq | 32768→512 | 0.999 ± 0.000 | 0.936 ± 0.007 |
| dinov2_b/clip_seq | 24576→512 | 0.991 ± 0.000 | 0.891 ± 0.009 |
| state/oracle_x | 64 | 0.974 ± 0.001 | 0.872 ± 0.012 |
| vjepa2_l_random/clip_mean | 1024 | 0.979 ± 0.011 | 0.851 ± 0.043 |
| dinov2_b_random/clip_seq | 24576→512 | 0.981 ± 0.002 | 0.801 ± 0.010 |
| cosmos_cv8/clip_grid4 | 2304 | 0.959 ± 0.006 | 0.748 ± 0.018 |
| contact/noise_000 | 192 | 0.527 ± 0.036 | 0.457 ± 0.041 |
| contact/realistic | 192 | 0.520 ± 0.033 | 0.452 ± 0.037 |
| contact/noise_010 | 192 | 0.518 ± 0.033 | 0.450 ± 0.042 |
| contact/noise_030 | 192 | 0.497 ± 0.028 | 0.435 ± 0.047 |
| cosmos_cv8/clip_seq | 144 | 0.908 ± 0.001 | 0.410 ± 0.021 |
| contact/noise_100 | 192 | 0.378 ± 0.021 | 0.336 ± 0.065 |
| pixels/last | 1024 | 0.819 ± 0.009 | 0.205 ± 0.051 |
| dinov2_b/last | 1536 | 0.815 ± 0.010 | 0.191 ± 0.064 |
| dinov2_b_random/last | 1536 | 0.811 ± 0.013 | 0.181 ± 0.068 |
| cosmos_cv8/clip_mean | 16 | 0.680 ± 0.009 | 0.176 ± 0.030 |
| dinov2_b/clip_mean | 1536 | 0.839 ± 0.009 | 0.176 ± 0.060 |
| dinov2_b_random/clip_mean | 1536 | 0.837 ± 0.009 | 0.169 ± 0.062 |
| dinov2_b/static | 1536 | -0.006 ± 0.010 | -0.003 ± 0.001 |
| dinov2_b_random/static | 1536 | -0.006 ± 0.010 | -0.003 ± 0.001 |
| pixels/static | 1024 | -0.006 ± 0.009 | -0.003 ± 0.001 |
| vjepa2_l/static | 1024 | -0.006 ± 0.010 | -0.003 ± 0.001 |
| vjepa2_l_random/static | 1024 | -0.006 ± 0.010 | -0.003 ± 0.001 |

Controls (push): shuffle R² in [-0.084, 0.023] over 258 fits; mean R² in [-0.018, -0.000].

## slide (n_train 700, n_test 150, seeds [0, 1, 2]; sorted by R²(mu))

| input | dim | R²(log_m) | R²(mu) |
|---|---|---|---|
| contact/raw | 192 | 0.972 ± 0.002 | 1.000 ± 0.000 |
| state/oracle_xc | 64 | -0.024 ± 0.025 | 1.000 ± 0.000 |
| state/oracle_fit | 2 | -0.024 ± 0.025 | 1.000 ± 0.000 |
| contact/noise_000 | 192 | -0.024 ± 0.025 | 1.000 ± 0.000 |
| pixels/clip_seq | 8192→512 | -0.025 ± 0.022 | 1.000 ± 0.000 |
| dinov2_b_random/clip_mean | 1536 | -0.029 ± 0.021 | 1.000 ± 0.000 |
| vjepa2_l_random/clip_seq | 32768→512 | -0.042 ± 0.024 | 1.000 ± 0.000 |
| dinov2_b_random/clip_seq | 24576→512 | -0.042 ± 0.022 | 1.000 ± 0.000 |
| dinov2_b_random/last | 1536 | -0.028 ± 0.021 | 1.000 ± 0.000 |
| dinov2_b/clip_seq | 24576→512 | -0.055 ± 0.046 | 1.000 ± 0.000 |
| dinov2_b/clip_seq + contact/realistic | 24768→512 | -0.024 ± 0.025 | 1.000 ± 0.000 |
| vjepa2_l/clip_seq_L16 | 32768→512 | -0.056 ± 0.036 | 1.000 ± 0.000 |
| vjepa2_l/clip_seq_L08 | 32768→512 | -0.054 ± 0.036 | 1.000 ± 0.000 |
| vjepa2_l/clip_seq | 32768→512 | -0.053 ± 0.023 | 1.000 ± 0.000 |
| track/clip | 128 | -0.024 ± 0.022 | 1.000 ± 0.000 |
| dinov2_b/clip_mean | 1536 | -0.030 ± 0.028 | 1.000 ± 0.000 |
| vjepa2_l/clip_mean_L16 | 1024 | -0.028 ± 0.025 | 1.000 ± 0.000 |
| vjepa2_l/clip_seq + contact/realistic | 32960→512 | -0.024 ± 0.025 | 1.000 ± 0.000 |
| vjepa2_l_random/clip_mean | 1024 | -0.027 ± 0.020 | 1.000 ± 0.000 |
| vjepa2_l/clip_mean | 1024 | -0.027 ± 0.020 | 0.999 ± 0.000 |
| vjepa2_l/clip_mean_L08 | 1024 | -0.028 ± 0.029 | 0.999 ± 0.000 |
| dinov2_b/last | 1536 | -0.029 ± 0.027 | 0.999 ± 0.000 |
| cosmos_cv8/clip_grid4 | 2304 | -0.024 ± 0.025 | 0.999 ± 0.000 |
| state/oracle_x | 64 | -0.024 ± 0.022 | 0.999 ± 0.000 |
| pixels/last | 1024 | -0.024 ± 0.022 | 0.999 ± 0.000 |
| contact/realistic | 192 | -0.024 ± 0.022 | 0.999 ± 0.000 |
| contact/noise_010 | 192 | -0.025 ± 0.025 | 0.994 ± 0.000 |
| cosmos_cv8/clip_seq | 144 | -0.025 ± 0.023 | 0.967 ± 0.003 |
| contact/noise_030 | 192 | -0.025 ± 0.025 | 0.964 ± 0.003 |
| contact/noise_100 | 192 | -0.025 ± 0.025 | 0.761 ± 0.013 |
| cosmos_cv8/clip_mean | 16 | -0.024 ± 0.016 | 0.761 ± 0.012 |
| dinov2_b/static | 1536 | -0.024 ± 0.025 | -0.001 ± 0.001 |
| dinov2_b_random/static | 1536 | -0.024 ± 0.025 | -0.001 ± 0.001 |
| pixels/static | 1024 | -0.024 ± 0.022 | -0.001 ± 0.001 |
| vjepa2_l/static | 1024 | -0.024 ± 0.025 | -0.001 ± 0.001 |
| vjepa2_l_random/static | 1024 | -0.024 ± 0.025 | -0.001 ± 0.001 |

Controls (slide): shuffle R² in [-0.104, 0.083] over 258 fits; mean R² in [-0.051, -0.000].

## poscheck (n_train 210, n_test 45, seeds [0, 1, 2]; sorted by R²(x0_m))

| input | dim | R²(x0_m) |
|---|---|---|
| state/oracle_xc | 64 | 1.000 ± 0.000 |
| track/clip | 128 | 1.000 ± 0.000 |
| state/oracle_x | 64 | 1.000 ± 0.000 |
| pixels/last | 1024 | 1.000 ± 0.000 |
| pixels/static | 1024 | 1.000 ± 0.000 |
| pixels/clip_seq | 8192→209 | 1.000 ± 0.000 |
| dinov2_b/static | 1536 | 0.999 ± 0.000 |
| dinov2_b/clip_seq | 24576→209 | 0.999 ± 0.000 |
| dinov2_b/last | 1536 | 0.999 ± 0.000 |
| dinov2_b/clip_mean | 1536 | 0.999 ± 0.000 |
| vjepa2_l/clip_mean | 1024 | 0.999 ± 0.000 |
| vjepa2_l/static | 1024 | 0.999 ± 0.000 |
| vjepa2_l_random/clip_mean | 1024 | 0.999 ± 0.000 |
| vjepa2_l_random/static | 1024 | 0.999 ± 0.000 |
| vjepa2_l/clip_seq | 32768→209 | 0.999 ± 0.001 |
| dinov2_b_random/clip_mean | 1536 | 0.998 ± 0.000 |
| dinov2_b_random/last | 1536 | 0.998 ± 0.000 |
| dinov2_b_random/static | 1536 | 0.998 ± 0.000 |
| vjepa2_l_random/clip_seq | 32768→209 | 0.998 ± 0.001 |
| vjepa2_l/clip_mean_L16 | 1024 | 0.997 ± 0.001 |
| dinov2_b_random/clip_seq | 24576→209 | 0.997 ± 0.001 |
| dinov2_b/clip_seq + contact/realistic | 24768→209 | 0.996 ± 0.002 |
| vjepa2_l/clip_seq + contact/realistic | 32960→209 | 0.995 ± 0.001 |
| vjepa2_l/clip_seq_L16 | 32768→209 | 0.993 ± 0.004 |
| cosmos_cv8/clip_grid4 | 2304 | 0.986 ± 0.003 |
| vjepa2_l/clip_seq_L08 | 32768→209 | 0.985 ± 0.007 |
| vjepa2_l/clip_mean_L08 | 1024 | 0.984 ± 0.006 |
| cosmos_cv8/clip_mean | 16 | 0.191 ± 0.092 |
| cosmos_cv8/clip_seq | 144 | 0.176 ± 0.016 |
| contact/realistic | 192 | -0.016 ± 0.018 |
| state/oracle_fit | 2 | -0.017 ± 0.017 |
| contact/raw | 192 | -0.018 ± 0.014 |

Controls (poscheck): shuffle R² in [-0.234, 0.216] over 117 fits; mean R² in [-0.035, -0.001].

