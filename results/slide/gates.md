# Sanity gates: `slide`

- Splits — s0: n_train=700, n_val=150, n_test=150, n_excluded=0; s1: n_train=700, n_val=150, n_test=150, n_excluded=0; s2: n_train=700, n_val=150, n_test=150, n_excluded=0
- git commit: `fce21d1`
- runs: 2026-10-22T22:45:46+00:00 … 2026-10-22T22:45:46+00:00
- All numbers: seed-mean test R² (perm: seed-mean of the per-seed null p95 / null mean).

| Gate | Status |
|---|---|
| mean R² ∈ [3·E − 0.01, 0.01], E = −(1 + n_test/n_train)/(n_test − 1) | **PASS** |
| static (frame 0) R² < 0.05 for all reps (log m, μ) | **PASS** |
| label-permutation null: p95 ≤ max(0.10, 3.3/√n_train), mean ≤ 0.02 | **PASS** |
| plan v1 rule, informational: shuffled-label control R² ≤ 0.05 | **PASS** (info) |
| state/oracle_xc on slide: R²(μ) > 0.9 (μ only on slide) | **PASS** |
| slide: R²(log m) < 0.05 for all video/pixel reps | **PASS** |
| poscheck: R²(x0) ≥ 0.5 per encoder for its val-chosen pooling (clip_mean shown) | **PASS** |
| contact/raw labelled ORACLE | **PASS** |

## PASS — mean R² ∈ [3·E − 0.01, 0.01], E = −(1 + n_test/n_train)/(n_test − 1)

- log_m: R² = -0.0244, n_test = 150, n_train = 700, bound [-0.0344, 0.01] (E ≈ -0.0081) → ok
- mu: R² = -0.0012, n_test = 150, n_train = 700, bound [-0.0344, 0.01] (E ≈ -0.0081) → ok
- bound: R² ∈ [3·E − 0.01, 0.01] with E = −(1 + n_test/n_train)/(n_test − 1), the expected test R² of the train-mean predictor (R² = −n_test·(ȳ_test − ȳ_train)²/SS_test, so R²/E ≈ χ²₁: one seed leaves the bound with p ≈ 0.08, the 3-seed mean with p ≈ 0.03)

## PASS — static (frame 0) R² < 0.05 for all reps (log m, μ)

- pixels/static log_m: R² = -0.0244
- pixels/static mu: R² = -0.0012
- dinov2_b_random/static log_m: R² = -0.0244
- dinov2_b_random/static mu: R² = -0.0012
- dinov2_b/static log_m: R² = -0.0244
- dinov2_b/static mu: R² = -0.0012
- vjepa2_l_random/static log_m: R² = -0.0244
- vjepa2_l_random/static mu: R² = -0.0012
- vjepa2_l/static log_m: R² = -0.0244
- vjepa2_l/static mu: R² = -0.0012

## PASS — label-permutation null: p95 ≤ max(0.10, 3.3/√n_train), mean ≤ 0.02

- pixels/static log_m: null p95 = -0.0244 (bound 0.125), null mean = -0.0244; real ridge R² = -0.0244, p ≤ 1.000 (max over seeds)
- pixels/static mu: null p95 = -0.0012 (bound 0.125), null mean = -0.0012; real ridge R² = -0.0012, p ≤ 1.000 (max over seeds)
- pixels/last log_m: null p95 = -0.0243 (bound 0.125), null mean = -0.0244; real ridge R² = -0.0244, p ≤ 0.733 (max over seeds)
- pixels/last mu: null p95 = 0.0953 (bound 0.125), null mean = -0.0424; real ridge R² = 0.9988, p ≤ 0.010 (max over seeds)
- pixels/clip_seq log_m: null p95 = -0.0237 (bound 0.125), null mean = -0.0244; real ridge R² = -0.0244, p ≤ 0.663 (max over seeds)
- pixels/clip_seq mu: null p95 = 0.0129 (bound 0.125), null mean = -0.2088; real ridge R² = 0.9999, p ≤ 0.010 (max over seeds)
- track/clip log_m: null p95 = -0.0240 (bound 0.125), null mean = -0.0244; real ridge R² = -0.0243, p ≤ 0.525 (max over seeds)
- track/clip mu: null p95 = 0.0577 (bound 0.125), null mean = -0.1000; real ridge R² = 0.9997, p ≤ 0.010 (max over seeds)
- dinov2_b_random/static log_m: null p95 = -0.0244 (bound 0.125), null mean = -0.0244; real ridge R² = -0.0244, p ≤ 1.000 (max over seeds)
- dinov2_b_random/static mu: null p95 = -0.0012 (bound 0.125), null mean = -0.0012; real ridge R² = -0.0012, p ≤ 1.000 (max over seeds)
- dinov2_b_random/last log_m: null p95 = -0.0216 (bound 0.125), null mean = -0.0247; real ridge R² = -0.0247, p ≤ 0.960 (max over seeds)
- dinov2_b_random/last mu: null p95 = -0.2242 (bound 0.125), null mean = -0.5322; real ridge R² = 0.9998, p ≤ 0.010 (max over seeds)
- dinov2_b_random/clip_mean log_m: null p95 = -0.0207 (bound 0.125), null mean = -0.0248; real ridge R² = -0.0261, p ≤ 0.960 (max over seeds)
- dinov2_b_random/clip_mean mu: null p95 = -0.2108 (bound 0.125), null mean = -0.4815; real ridge R² = 0.9999, p ≤ 0.010 (max over seeds)
- dinov2_b_random/clip_seq log_m: null p95 = -0.0131 (bound 0.125), null mean = -0.0291; real ridge R² = -0.0279, p ≤ 0.931 (max over seeds)
- dinov2_b_random/clip_seq mu: null p95 = -0.1139 (bound 0.125), null mean = -0.3049; real ridge R² = 0.9999, p ≤ 0.010 (max over seeds)
- dinov2_b/static log_m: null p95 = -0.0244 (bound 0.125), null mean = -0.0244; real ridge R² = -0.0244, p ≤ 1.000 (max over seeds)
- dinov2_b/static mu: null p95 = -0.0012 (bound 0.125), null mean = -0.0012; real ridge R² = -0.0012, p ≤ 1.000 (max over seeds)
- dinov2_b/last log_m: null p95 = -0.0222 (bound 0.125), null mean = -0.0246; real ridge R² = -0.0249, p ≤ 0.980 (max over seeds)
- dinov2_b/last mu: null p95 = -0.2462 (bound 0.125), null mean = -0.4823; real ridge R² = 0.9994, p ≤ 0.010 (max over seeds)
- dinov2_b/clip_mean log_m: null p95 = -0.0220 (bound 0.125), null mean = -0.0246; real ridge R² = -0.0250, p ≤ 0.990 (max over seeds)
- dinov2_b/clip_mean mu: null p95 = -0.3016 (bound 0.125), null mean = -0.5550; real ridge R² = 0.9997, p ≤ 0.010 (max over seeds)
- dinov2_b/clip_seq log_m: null p95 = -0.0126 (bound 0.125), null mean = -0.0287; real ridge R² = -0.0312, p ≤ 1.000 (max over seeds)
- dinov2_b/clip_seq mu: null p95 = -0.1435 (bound 0.125), null mean = -0.3464; real ridge R² = 0.9998, p ≤ 0.010 (max over seeds)
- vjepa2_l_random/static log_m: null p95 = -0.0244 (bound 0.125), null mean = -0.0244; real ridge R² = -0.0244, p ≤ 1.000 (max over seeds)
- vjepa2_l_random/static mu: null p95 = -0.0012 (bound 0.125), null mean = -0.0012; real ridge R² = -0.0012, p ≤ 1.000 (max over seeds)
- vjepa2_l_random/clip_mean log_m: null p95 = -0.0220 (bound 0.125), null mean = -0.0246; real ridge R² = -0.0256, p ≤ 0.990 (max over seeds)
- vjepa2_l_random/clip_mean mu: null p95 = -0.6845 (bound 0.125), null mean = -1.1023; real ridge R² = 0.9995, p ≤ 0.010 (max over seeds)
- vjepa2_l_random/clip_seq log_m: null p95 = -0.0135 (bound 0.125), null mean = -0.0297; real ridge R² = -0.0310, p ≤ 0.960 (max over seeds)
- vjepa2_l_random/clip_seq mu: null p95 = -0.0920 (bound 0.125), null mean = -0.2799; real ridge R² = 0.9999, p ≤ 0.010 (max over seeds)
- vjepa2_l/static log_m: null p95 = -0.0244 (bound 0.125), null mean = -0.0244; real ridge R² = -0.0244, p ≤ 1.000 (max over seeds)
- vjepa2_l/static mu: null p95 = -0.0012 (bound 0.125), null mean = -0.0012; real ridge R² = -0.0012, p ≤ 1.000 (max over seeds)
- vjepa2_l/clip_mean_L08 log_m: null p95 = -0.0218 (bound 0.125), null mean = -0.0245; real ridge R² = -0.0252, p ≤ 0.802 (max over seeds)
- vjepa2_l/clip_mean_L08 mu: null p95 = -1.4708 (bound 0.125), null mean = -2.4865; real ridge R² = 0.9995, p ≤ 0.010 (max over seeds)
- vjepa2_l/clip_mean_L16 log_m: null p95 = -0.0221 (bound 0.125), null mean = -0.0244; real ridge R² = -0.0249, p ≤ 0.782 (max over seeds)
- vjepa2_l/clip_mean_L16 mu: null p95 = -0.2704 (bound 0.125), null mean = -0.5562; real ridge R² = 0.9996, p ≤ 0.010 (max over seeds)
- vjepa2_l/clip_mean log_m: null p95 = -0.0218 (bound 0.125), null mean = -0.0244; real ridge R² = -0.0246, p ≤ 0.733 (max over seeds)
- vjepa2_l/clip_mean mu: null p95 = -0.2687 (bound 0.125), null mean = -0.5410; real ridge R² = 0.9995, p ≤ 0.010 (max over seeds)
- vjepa2_l/clip_seq_L08 log_m: null p95 = -0.0131 (bound 0.125), null mean = -0.0300; real ridge R² = -0.0320, p ≤ 0.921 (max over seeds)
- vjepa2_l/clip_seq_L08 mu: null p95 = -0.1459 (bound 0.125), null mean = -0.4007; real ridge R² = 0.9997, p ≤ 0.010 (max over seeds)
- vjepa2_l/clip_seq_L16 log_m: null p95 = -0.0139 (bound 0.125), null mean = -0.0298; real ridge R² = -0.0297, p ≤ 0.723 (max over seeds)
- vjepa2_l/clip_seq_L16 mu: null p95 = -0.0768 (bound 0.125), null mean = -0.2864; real ridge R² = 0.9997, p ≤ 0.010 (max over seeds)
- vjepa2_l/clip_seq log_m: null p95 = -0.0133 (bound 0.125), null mean = -0.0290; real ridge R² = -0.0287, p ≤ 0.733 (max over seeds)
- vjepa2_l/clip_seq mu: null p95 = -0.0571 (bound 0.125), null mean = -0.2492; real ridge R² = 0.9996, p ≤ 0.010 (max over seeds)
- cosmos_cv8/clip_mean log_m: null p95 = -0.0236 (bound 0.125), null mean = -0.0251; real ridge R² = -0.0248, p ≤ 0.426 (max over seeds)
- cosmos_cv8/clip_mean mu: null p95 = 0.0767 (bound 0.125), null mean = -0.0336; real ridge R² = 0.7610, p ≤ 0.010 (max over seeds)
- cosmos_cv8/clip_seq log_m: null p95 = -0.0241 (bound 0.125), null mean = -0.0244; real ridge R² = -0.0244, p ≤ 0.673 (max over seeds)
- cosmos_cv8/clip_seq mu: null p95 = 0.0183 (bound 0.125), null mean = -0.1351; real ridge R² = 0.9666, p ≤ 0.010 (max over seeds)
- cosmos_cv8/clip_grid4 log_m: null p95 = -0.0214 (bound 0.125), null mean = -0.0247; real ridge R² = -0.0243, p ≤ 0.535 (max over seeds)
- cosmos_cv8/clip_grid4 mu: null p95 = -0.1278 (bound 0.125), null mean = -0.3218; real ridge R² = 0.9988, p ≤ 0.010 (max over seeds)
- combo/dinov2_b/clip_seq+contact/realistic log_m: null p95 = -0.0244 (bound 0.125), null mean = -0.0244; real ridge R² = -0.0244, p ≤ 0.881 (max over seeds)
- combo/dinov2_b/clip_seq+contact/realistic mu: null p95 = -0.1932 (bound 0.125), null mean = -0.4250; real ridge R² = 0.9996, p ≤ 0.010 (max over seeds)
- combo/vjepa2_l/clip_mean_L16+contact/realistic log_m: null p95 = -0.0244 (bound 0.125), null mean = -0.0244; real ridge R² = -0.0244, p ≤ 0.743 (max over seeds)
- combo/vjepa2_l/clip_mean_L16+contact/realistic mu: null p95 = -0.7791 (bound 0.125), null mean = -1.2452; real ridge R² = 0.9997, p ≤ 0.010 (max over seeds)
- combo/vjepa2_l/clip_seq+contact/realistic log_m: null p95 = -0.0244 (bound 0.125), null mean = -0.0244; real ridge R² = -0.0244, p ≤ 0.624 (max over seeds)
- combo/vjepa2_l/clip_seq+contact/realistic mu: null p95 = -0.1534 (bound 0.125), null mean = -0.3533; real ridge R² = 0.9995, p ≤ 0.010 (max over seeds)
- state/oracle_xc log_m: null p95 = -0.0241 (bound 0.125), null mean = -0.0244; real ridge R² = -0.0244, p ≤ 0.584 (max over seeds)
- state/oracle_xc mu: null p95 = 0.0876 (bound 0.125), null mean = -0.0247; real ridge R² = 1.0000, p ≤ 0.010 (max over seeds)
- state/oracle_x log_m: null p95 = -0.0241 (bound 0.125), null mean = -0.0244; real ridge R² = -0.0244, p ≤ 0.584 (max over seeds)
- state/oracle_x mu: null p95 = 0.0924 (bound 0.125), null mean = -0.0306; real ridge R² = 0.9988, p ≤ 0.010 (max over seeds)
- state/oracle_fit log_m: null p95 = -0.0209 (bound 0.125), null mean = -0.0245; real ridge R² = -0.0242, p ≤ 0.535 (max over seeds)
- state/oracle_fit mu: null p95 = 0.1037 (bound 0.125), null mean = -0.0099; real ridge R² = 1.0000, p ≤ 0.010 (max over seeds)
- contact/raw log_m: null p95 = 0.0633 (bound 0.125), null mean = -0.0453; real ridge R² = 0.9715, p ≤ 0.010 (max over seeds)
- contact/raw mu: null p95 = -0.0069 (bound 0.125), null mean = -0.2651; real ridge R² = 1.0000, p ≤ 0.010 (max over seeds)
- contact/realistic log_m: null p95 = -0.0242 (bound 0.125), null mean = -0.0244; real ridge R² = -0.0244, p ≤ 0.713 (max over seeds)
- contact/realistic mu: null p95 = -0.0685 (bound 0.125), null mean = -0.2411; real ridge R² = 0.9986, p ≤ 0.010 (max over seeds)
- contact/noise_000 log_m: null p95 = -0.0241 (bound 0.125), null mean = -0.0244; real ridge R² = -0.0244, p ≤ 0.634 (max over seeds)
- contact/noise_000 mu: null p95 = 0.0015 (bound 0.125), null mean = -0.2050; real ridge R² = 1.0000, p ≤ 0.010 (max over seeds)
- contact/noise_010 log_m: null p95 = -0.0242 (bound 0.125), null mean = -0.0244; real ridge R² = -0.0245, p ≤ 0.752 (max over seeds)
- contact/noise_010 mu: null p95 = -0.0425 (bound 0.125), null mean = -0.1976; real ridge R² = 0.9943, p ≤ 0.010 (max over seeds)
- contact/noise_030 log_m: null p95 = -0.0243 (bound 0.125), null mean = -0.0244; real ridge R² = -0.0245, p ≤ 0.842 (max over seeds)
- contact/noise_030 mu: null p95 = -0.0079 (bound 0.125), null mean = -0.1544; real ridge R² = 0.9644, p ≤ 0.010 (max over seeds)
- contact/noise_100 log_m: null p95 = -0.0243 (bound 0.125), null mean = -0.0244; real ridge R² = -0.0245, p ≤ 0.901 (max over seeds)
- contact/noise_100 mu: null p95 = -0.0028 (bound 0.125), null mean = -0.1246; real ridge R² = 0.7555, p ≤ 0.010 (max over seeds)
- gate: seed-mean null p95 ≤ max(0.10, 3.3/√n_train) AND seed-mean null mean ≤ 0.02 per (input, target); null = ridge with the real fit's alpha on permuted train labels. The √n term: for an input that encodes y almost exactly, a permuted fit returns ≈ ȳ + r·(y − ȳ) with chance correlation r ~ N(0, 1/n_train), so null R² ≈ 2r − r² and its p95 ≈ 3.3/√n_train without any leak
- (info, not gated) single shuffled-label fits: max seed-mean R² = 0.0099 over 74 (input, target) cells

## PASS (info) — plan v1 rule, informational: shuffled-label control R² ≤ 0.05

- 74/74 (input, target) cells pass (seed-mean R²); failing cells listed

## PASS — state/oracle_xc on slide: R²(μ) > 0.9 (μ only on slide)

- μ: R² = 1.0000 (need > 0.9)
- (info, not gated) log m: R² = -0.0244 (m is not identifiable on slide; expected ≈ 0)
- (info, not gated) state/oracle_x, pixel-quantised oracle (not an upper bound): R²(μ) = 0.9988

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

