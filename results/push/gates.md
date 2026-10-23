# Sanity gates: `push`

- Splits — s0: n_train=1400, n_val=300, n_test=300, n_excluded=0; s1: n_train=1400, n_val=300, n_test=300, n_excluded=0; s2: n_train=1400, n_val=300, n_test=300, n_excluded=0
- git commit: `c5fe74b`
- runs: 2026-10-23T00:59:41+00:00 … 2026-10-23T00:59:41+00:00
- All numbers: seed-mean test R² (perm: seed-mean of the per-seed null p95 / null mean).

| Gate | Status |
|---|---|
| mean R² ∈ [3·E − 0.01, 0.01], E = −(1 + n_test/n_train)/(n_test − 3) | **PASS** |
| static (frame 0) R² < 0.05 for all reps (log m, μ) | **PASS** |
| label-permutation null: p95 ≤ max(0.10, 3.3/√n_train), mean ≤ 0.02 | **PASS** |
| plan v1 rule, informational: shuffled-label control R² ≤ 0.05 | **PASS** (info) |
| state/oracle_xc on push: R²(log m) > 0.95, R²(μ) > 0.9 | **PASS** |
| plan v1 rule, informational: state/oracle_x on push: R²(log m) > 0.95, R²(μ) > 0.9 | **FAIL** (info) |
| slide: R²(log m) < 0.05 for all video/pixel reps | **PASS** |
| poscheck: R²(x0) ≥ 0.5 per encoder for its val-chosen pooling (clip_mean shown) | **PASS** |
| contact/raw labelled ORACLE | **PASS** |

## PASS — mean R² ∈ [3·E − 0.01, 0.01], E = −(1 + n_test/n_train)/(n_test − 3)

- log_m: R² = -0.0065, n_test = 300, n_train = 1400, bound [-0.0223, 0.01] (E ≈ -0.0041) → ok
- mu: R² = -0.0029, n_test = 300, n_train = 1400, bound [-0.0223, 0.01] (E ≈ -0.0041) → ok
- bound: R² ∈ [3·E − 0.01, 0.01] with E = −(1 + n_test/n_train)/(n_test − 3), the expected test R² of the train-mean predictor for Gaussian y (R² = −n_test·(ȳ_test − ȳ_train)²/SS_test, so R²/E ≈ χ²₁: one seed leaves the bound with approximate p ≈ 0.08; the 3-seed mean with approximate p ≈ 0.03, optimistic because the splits reshuffle the same clips)

## PASS — static (frame 0) R² < 0.05 for all reps (log m, μ)

- pixels/static log_m: R² = -0.0065
- pixels/static mu: R² = -0.0029
- dinov2_b_random/static log_m: R² = -0.0065
- dinov2_b_random/static mu: R² = -0.0029
- dinov2_b/static log_m: R² = -0.0065
- dinov2_b/static mu: R² = -0.0029
- vjepa2_l_random/static log_m: R² = -0.0065
- vjepa2_l_random/static mu: R² = -0.0029
- vjepa2_l/static log_m: R² = -0.0065
- vjepa2_l/static mu: R² = -0.0029

## PASS — label-permutation null: p95 ≤ max(0.10, 3.3/√n_train), mean ≤ 0.02

- pixels/static log_m: null p95 = -0.0065 (bound 0.100), null mean = -0.0065; real ridge R² = -0.0065, p ≤ 1.000 (max over seeds)
- pixels/static mu: null p95 = -0.0029 (bound 0.100), null mean = -0.0029; real ridge R² = -0.0029, p ≤ 1.000 (max over seeds)
- pixels/last log_m: null p95 = 0.0602 (bound 0.100), null mean = -0.0181; real ridge R² = 0.8186, p ≤ 0.010 (max over seeds)
- pixels/last mu: null p95 = 0.0277 (bound 0.100), null mean = -0.0152; real ridge R² = 0.2053, p ≤ 0.010 (max over seeds)
- pixels/clip_seq log_m: null p95 = -0.0466 (bound 0.100), null mean = -0.2083; real ridge R² = 0.9990, p ≤ 0.010 (max over seeds)
- pixels/clip_seq mu: null p95 = -0.0449 (bound 0.100), null mean = -0.1889; real ridge R² = 0.9821, p ≤ 0.010 (max over seeds)
- track/clip log_m: null p95 = 0.0446 (bound 0.100), null mean = -0.0622; real ridge R² = 0.9738, p ≤ 0.010 (max over seeds)
- track/clip mu: null p95 = 0.0294 (bound 0.100), null mean = -0.0704; real ridge R² = 0.9390, p ≤ 0.010 (max over seeds)
- dinov2_b_random/static log_m: null p95 = -0.0065 (bound 0.100), null mean = -0.0065; real ridge R² = -0.0065, p ≤ 1.000 (max over seeds)
- dinov2_b_random/static mu: null p95 = -0.0029 (bound 0.100), null mean = -0.0029; real ridge R² = -0.0029, p ≤ 1.000 (max over seeds)
- dinov2_b_random/last log_m: null p95 = 0.0307 (bound 0.100), null mean = -0.0566; real ridge R² = 0.8107, p ≤ 0.010 (max over seeds)
- dinov2_b_random/last mu: null p95 = 0.0191 (bound 0.100), null mean = -0.0240; real ridge R² = 0.1801, p ≤ 0.010 (max over seeds)
- dinov2_b_random/clip_mean log_m: null p95 = 0.0340 (bound 0.100), null mean = -0.0497; real ridge R² = 0.8367, p ≤ 0.010 (max over seeds)
- dinov2_b_random/clip_mean mu: null p95 = 0.0193 (bound 0.100), null mean = -0.0214; real ridge R² = 0.1692, p ≤ 0.010 (max over seeds)
- dinov2_b_random/clip_seq log_m: null p95 = -0.0513 (bound 0.100), null mean = -0.1719; real ridge R² = 0.9555, p ≤ 0.010 (max over seeds)
- dinov2_b_random/clip_seq mu: null p95 = -0.0249 (bound 0.100), null mean = -0.1402; real ridge R² = 0.6842, p ≤ 0.010 (max over seeds)
- dinov2_b/static log_m: null p95 = -0.0065 (bound 0.100), null mean = -0.0065; real ridge R² = -0.0065, p ≤ 1.000 (max over seeds)
- dinov2_b/static mu: null p95 = -0.0029 (bound 0.100), null mean = -0.0029; real ridge R² = -0.0029, p ≤ 1.000 (max over seeds)
- dinov2_b/last log_m: null p95 = 0.0507 (bound 0.100), null mean = -0.0346; real ridge R² = 0.8146, p ≤ 0.010 (max over seeds)
- dinov2_b/last mu: null p95 = 0.0182 (bound 0.100), null mean = -0.0227; real ridge R² = 0.1907, p ≤ 0.010 (max over seeds)
- dinov2_b/clip_mean log_m: null p95 = 0.0494 (bound 0.100), null mean = -0.0359; real ridge R² = 0.8385, p ≤ 0.010 (max over seeds)
- dinov2_b/clip_mean mu: null p95 = 0.0196 (bound 0.100), null mean = -0.0181; real ridge R² = 0.1756, p ≤ 0.010 (max over seeds)
- dinov2_b/clip_seq log_m: null p95 = -0.0464 (bound 0.100), null mean = -0.1640; real ridge R² = 0.9827, p ≤ 0.010 (max over seeds)
- dinov2_b/clip_seq mu: null p95 = -0.0407 (bound 0.100), null mean = -0.1425; real ridge R² = 0.8417, p ≤ 0.010 (max over seeds)
- vjepa2_l_random/static log_m: null p95 = -0.0065 (bound 0.100), null mean = -0.0065; real ridge R² = -0.0065, p ≤ 1.000 (max over seeds)
- vjepa2_l_random/static mu: null p95 = -0.0029 (bound 0.100), null mean = -0.0029; real ridge R² = -0.0029, p ≤ 1.000 (max over seeds)
- vjepa2_l_random/clip_mean log_m: null p95 = -0.3846 (bound 0.100), null mean = -0.7048; real ridge R² = 0.9793, p ≤ 0.010 (max over seeds)
- vjepa2_l_random/clip_mean mu: null p95 = -0.3189 (bound 0.100), null mean = -0.5172; real ridge R² = 0.8503, p ≤ 0.010 (max over seeds)
- vjepa2_l_random/clip_seq log_m: null p95 = -0.0302 (bound 0.100), null mean = -0.1562; real ridge R² = 0.9971, p ≤ 0.010 (max over seeds)
- vjepa2_l_random/clip_seq mu: null p95 = -0.0284 (bound 0.100), null mean = -0.1319; real ridge R² = 0.8829, p ≤ 0.010 (max over seeds)
- vjepa2_l/static log_m: null p95 = -0.0065 (bound 0.100), null mean = -0.0065; real ridge R² = -0.0065, p ≤ 1.000 (max over seeds)
- vjepa2_l/static mu: null p95 = -0.0029 (bound 0.100), null mean = -0.0029; real ridge R² = -0.0029, p ≤ 1.000 (max over seeds)
- vjepa2_l/clip_mean_L08 log_m: null p95 = -0.2021 (bound 0.100), null mean = -0.3735; real ridge R² = 0.9937, p ≤ 0.010 (max over seeds)
- vjepa2_l/clip_mean_L08 mu: null p95 = -0.2218 (bound 0.100), null mean = -0.3739; real ridge R² = 0.9576, p ≤ 0.010 (max over seeds)
- vjepa2_l/clip_mean_L16 log_m: null p95 = -0.2916 (bound 0.100), null mean = -0.4426; real ridge R² = 0.9967, p ≤ 0.010 (max over seeds)
- vjepa2_l/clip_mean_L16 mu: null p95 = -0.2091 (bound 0.100), null mean = -0.3686; real ridge R² = 0.9765, p ≤ 0.010 (max over seeds)
- vjepa2_l/clip_mean log_m: null p95 = -0.2050 (bound 0.100), null mean = -0.3645; real ridge R² = 0.9958, p ≤ 0.010 (max over seeds)
- vjepa2_l/clip_mean mu: null p95 = -0.1538 (bound 0.100), null mean = -0.3145; real ridge R² = 0.9695, p ≤ 0.010 (max over seeds)
- vjepa2_l/clip_seq_L08 log_m: null p95 = -0.0385 (bound 0.100), null mean = -0.1608; real ridge R² = 0.9926, p ≤ 0.010 (max over seeds)
- vjepa2_l/clip_seq_L08 mu: null p95 = -0.0324 (bound 0.100), null mean = -0.1452; real ridge R² = 0.9446, p ≤ 0.010 (max over seeds)
- vjepa2_l/clip_seq_L16 log_m: null p95 = -0.0323 (bound 0.100), null mean = -0.1405; real ridge R² = 0.9953, p ≤ 0.010 (max over seeds)
- vjepa2_l/clip_seq_L16 mu: null p95 = -0.0223 (bound 0.100), null mean = -0.1252; real ridge R² = 0.9675, p ≤ 0.010 (max over seeds)
- vjepa2_l/clip_seq log_m: null p95 = -0.0153 (bound 0.100), null mean = -0.1303; real ridge R² = 0.9958, p ≤ 0.010 (max over seeds)
- vjepa2_l/clip_seq mu: null p95 = -0.0279 (bound 0.100), null mean = -0.1313; real ridge R² = 0.9671, p ≤ 0.010 (max over seeds)
- cosmos_cv8/clip_mean log_m: null p95 = 0.0614 (bound 0.100), null mean = -0.0166; real ridge R² = 0.6798, p ≤ 0.010 (max over seeds)
- cosmos_cv8/clip_mean mu: null p95 = 0.0260 (bound 0.100), null mean = -0.0124; real ridge R² = 0.1759, p ≤ 0.010 (max over seeds)
- cosmos_cv8/clip_seq log_m: null p95 = 0.0370 (bound 0.100), null mean = -0.0586; real ridge R² = 0.9078, p ≤ 0.010 (max over seeds)
- cosmos_cv8/clip_seq mu: null p95 = 0.0193 (bound 0.100), null mean = -0.0443; real ridge R² = 0.4133, p ≤ 0.010 (max over seeds)
- cosmos_cv8/clip_grid4 log_m: null p95 = -0.0085 (bound 0.100), null mean = -0.1337; real ridge R² = 0.9427, p ≤ 0.010 (max over seeds)
- cosmos_cv8/clip_grid4 mu: null p95 = -0.0402 (bound 0.100), null mean = -0.1476; real ridge R² = 0.6822, p ≤ 0.010 (max over seeds)
- combo/dinov2_b/clip_seq+contact/realistic log_m: null p95 = -0.0466 (bound 0.100), null mean = -0.1763; real ridge R² = 0.9922, p ≤ 0.010 (max over seeds)
- combo/dinov2_b/clip_seq+contact/realistic mu: null p95 = -0.0477 (bound 0.100), null mean = -0.1567; real ridge R² = 0.9480, p ≤ 0.010 (max over seeds)
- combo/vjepa2_l/clip_mean_L16+contact/realistic log_m: null p95 = -0.3466 (bound 0.100), null mean = -0.5394; real ridge R² = 0.9971, p ≤ 0.010 (max over seeds)
- combo/vjepa2_l/clip_mean_L16+contact/realistic mu: null p95 = -0.2528 (bound 0.100), null mean = -0.4191; real ridge R² = 0.9819, p ≤ 0.010 (max over seeds)
- combo/vjepa2_l/clip_seq+contact/realistic log_m: null p95 = -0.0431 (bound 0.100), null mean = -0.1653; real ridge R² = 0.9954, p ≤ 0.010 (max over seeds)
- combo/vjepa2_l/clip_seq+contact/realistic mu: null p95 = -0.0577 (bound 0.100), null mean = -0.1585; real ridge R² = 0.9750, p ≤ 0.010 (max over seeds)
- state/oracle_xc log_m: null p95 = 0.0795 (bound 0.100), null mean = -0.0117; real ridge R² = 0.9763, p ≤ 0.010 (max over seeds)
- state/oracle_xc mu: null p95 = 0.0642 (bound 0.100), null mean = -0.0140; real ridge R² = 1.0000, p ≤ 0.010 (max over seeds)
- state/oracle_x log_m: null p95 = 0.0718 (bound 0.100), null mean = -0.0144; real ridge R² = 0.9740, p ≤ 0.010 (max over seeds)
- state/oracle_x mu: null p95 = 0.0568 (bound 0.100), null mean = -0.0131; real ridge R² = 0.8654, p ≤ 0.010 (max over seeds)
- state/oracle_fit log_m: null p95 = 0.0808 (bound 0.100), null mean = -0.0068; real ridge R² = 1.0000, p ≤ 0.010 (max over seeds)
- state/oracle_fit mu: null p95 = 0.0743 (bound 0.100), null mean = -0.0033; real ridge R² = 1.0000, p ≤ 0.010 (max over seeds)
- contact/raw log_m: null p95 = 0.0432 (bound 0.100), null mean = -0.0732; real ridge R² = 0.9997, p ≤ 0.010 (max over seeds)
- contact/raw mu: null p95 = -0.0176 (bound 0.100), null mean = -0.3822; real ridge R² = 1.0000, p ≤ 0.010 (max over seeds)
- contact/realistic log_m: null p95 = 0.0447 (bound 0.100), null mean = -0.0180; real ridge R² = 0.5214, p ≤ 0.010 (max over seeds)
- contact/realistic mu: null p95 = 0.0365 (bound 0.100), null mean = -0.0133; real ridge R² = 0.4523, p ≤ 0.010 (max over seeds)
- contact/noise_000 log_m: null p95 = 0.0487 (bound 0.100), null mean = -0.0167; real ridge R² = 0.5268, p ≤ 0.010 (max over seeds)
- contact/noise_000 mu: null p95 = 0.0368 (bound 0.100), null mean = -0.0129; real ridge R² = 0.4571, p ≤ 0.010 (max over seeds)
- contact/noise_010 log_m: null p95 = 0.0404 (bound 0.100), null mean = -0.0212; real ridge R² = 0.5187, p ≤ 0.010 (max over seeds)
- contact/noise_010 mu: null p95 = 0.0351 (bound 0.100), null mean = -0.0171; real ridge R² = 0.4495, p ≤ 0.010 (max over seeds)
- contact/noise_030 log_m: null p95 = 0.0370 (bound 0.100), null mean = -0.0233; real ridge R² = 0.4991, p ≤ 0.010 (max over seeds)
- contact/noise_030 mu: null p95 = 0.0357 (bound 0.100), null mean = -0.0195; real ridge R² = 0.4342, p ≤ 0.010 (max over seeds)
- contact/noise_100 log_m: null p95 = 0.0056 (bound 0.100), null mean = -0.0502; real ridge R² = 0.3782, p ≤ 0.010 (max over seeds)
- contact/noise_100 mu: null p95 = 0.0091 (bound 0.100), null mean = -0.0463; real ridge R² = 0.3365, p ≤ 0.010 (max over seeds)
- gate: seed-mean null p95 ≤ max(0.10, 3.3/√n_train) AND seed-mean null mean ≤ 0.02 per (input, target); null = ridge with the real fit's alpha on permuted train labels. The √n term: for an input that encodes y almost exactly, a permuted fit returns ≈ ȳ + r·(y − ȳ) with chance correlation r ~ N(0, 1/n_train), so null R² ≈ 2r − r² and its p95 ≈ 3.3/√n_train without any leak
- (info, not gated) single shuffled-label fits: max seed-mean R² = 0.0039 over 74 (input, target) cells

## PASS (info) — plan v1 rule, informational: shuffled-label control R² ≤ 0.05

- 74/74 (input, target) cells pass (seed-mean R²); failing cells listed

## PASS — state/oracle_xc on push: R²(log m) > 0.95, R²(μ) > 0.9

- log m: R² = 0.9763 (need > 0.95)
- μ: R² = 1.0000 (need > 0.9)
- (info, not gated) state/oracle_x, pixel-quantised oracle (not an upper bound): R²(log m) = 0.9740, R²(μ) = 0.8654

## FAIL (info) — plan v1 rule, informational: state/oracle_x on push: R²(log m) > 0.95, R²(μ) > 0.9

- log m: R² = 0.9740 (need > 0.95)
- μ: R² = 0.8654 (need > 0.9)  ← FAIL

## PASS — slide: R²(log m) < 0.05 for all video/pixel reps

dataset `slide`; video/pixel reps only (contact, state excluded)
- pixels/static: R²(log m) = -0.0244
- pixels/last: R²(log m) = -0.0244
- pixels/clip_seq: R²(log m) = -0.0244
- track/clip: R²(log m) = -0.0243
- dinov2_b_random/static: R²(log m) = -0.0244
- dinov2_b_random/last: R²(log m) = -0.0247
- dinov2_b_random/clip_mean: R²(log m) = -0.0261
- dinov2_b_random/clip_seq: R²(log m) = -0.0279
- dinov2_b/static: R²(log m) = -0.0244
- dinov2_b/last: R²(log m) = -0.0249
- dinov2_b/clip_mean: R²(log m) = -0.0250
- dinov2_b/clip_seq: R²(log m) = -0.0312
- vjepa2_l_random/static: R²(log m) = -0.0244
- vjepa2_l_random/clip_mean: R²(log m) = -0.0256
- vjepa2_l_random/clip_seq: R²(log m) = -0.0310
- vjepa2_l/static: R²(log m) = -0.0244
- vjepa2_l/clip_mean_L08: R²(log m) = -0.0252
- vjepa2_l/clip_mean_L16: R²(log m) = -0.0249
- vjepa2_l/clip_mean: R²(log m) = -0.0246
- vjepa2_l/clip_seq_L08: R²(log m) = -0.0320
- vjepa2_l/clip_seq_L16: R²(log m) = -0.0297
- vjepa2_l/clip_seq: R²(log m) = -0.0287
- cosmos_cv8/clip_mean: R²(log m) = -0.0248
- cosmos_cv8/clip_seq: R²(log m) = -0.0244
- cosmos_cv8/clip_grid4: R²(log m) = -0.0243
- (info, not gated) state/oracle_xc on slide: R²(log m) = -0.0244 (expected ≈ 0)
- (info, not gated) state/oracle_x on slide: R²(log m) = -0.0244 (expected ≈ 0)
- (info, not gated) contact/raw ORACLE on slide: R²(log m) = 0.9715 (high by construction, N = mg)

## PASS — poscheck: R²(x0) ≥ 0.5 per encoder for its val-chosen pooling (clip_mean shown)

dataset `poscheck`, target x0_m
- dinov2_b_random/clip_mean: R²(x0) = 0.9982; other poolings: clip_seq=0.997
- dinov2_b/clip_mean: R²(x0) = 0.9993; other poolings: clip_seq=0.999; val-chosen pooling clip_seq = 0.999 (test)
- vjepa2_l_random/clip_mean: R²(x0) = 0.9987; other poolings: clip_seq=0.998
- vjepa2_l/clip_mean: R²(x0) = 0.9992 (other layers: clip_mean_L08=0.984, clip_mean_L16=0.997); other poolings: clip_seq_L08=0.985, clip_seq_L16=0.993, clip_seq=0.999
- cosmos_cv8/clip_mean: R²(x0) = 0.1583; other poolings: clip_seq=0.166, clip_grid4=0.986  ← global mean loses position for cosmos_cv8; position-preserving pooling (chosen on val): clip_grid4 = 0.986 (test)

## PASS — contact/raw labelled ORACLE

- contact/raw rows are labelled **ORACLE** in table.md and drawn hatched ('Contact oracle') in headline.png
- probes.csv `oracle` column for contact/raw: True

