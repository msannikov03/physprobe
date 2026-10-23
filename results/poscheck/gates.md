# Sanity gates: `poscheck`

- Splits — s0: n_train=210, n_val=45, n_test=45, n_excluded=0; s1: n_train=210, n_val=45, n_test=45, n_excluded=0; s2: n_train=210, n_val=45, n_test=45, n_excluded=0
- git commit: `c5fe74b`
- runs: 2026-10-23T00:58:59+00:00 … 2026-10-23T00:58:59+00:00
- All numbers: seed-mean test R² (perm: seed-mean of the per-seed null p95 / null mean).

| Gate | Status |
|---|---|
| mean R² ∈ [3·E − 0.01, 0.01], E = −(1 + n_test/n_train)/(n_test − 3) | **PASS** |
| static (frame 0) R² < 0.05 for all reps (log m, μ) | **N/A** |
| label-permutation null: p95 ≤ max(0.10, 3.3/√n_train), mean ≤ 0.02 | **PASS** |
| plan v1 rule, informational: shuffled-label control R² ≤ 0.05 | **FAIL** (info) |
| state/oracle_xc on poscheck: R²(log m) > 0.95, R²(μ) > 0.9 | **N/A** |
| slide: R²(log m) < 0.05 for all video/pixel reps | **PASS** |
| poscheck: R²(x0) ≥ 0.5 per encoder for its val-chosen pooling (clip_mean shown) | **PASS** |
| contact/raw labelled ORACLE | **PASS** |

## PASS — mean R² ∈ [3·E − 0.01, 0.01], E = −(1 + n_test/n_train)/(n_test − 3)

- x0_m: R² = -0.0169, n_test = 45, n_train = 210, bound [-0.0967, 0.01] (E ≈ -0.0289) → ok
- bound: R² ∈ [3·E − 0.01, 0.01] with E = −(1 + n_test/n_train)/(n_test − 3), the expected test R² of the train-mean predictor for Gaussian y (R² = −n_test·(ȳ_test − ȳ_train)²/SS_test, so R²/E ≈ χ²₁: one seed leaves the bound with approximate p ≈ 0.08; the 3-seed mean with approximate p ≈ 0.03, optimistic because the splits reshuffle the same clips)

## N/A — static (frame 0) R² < 0.05 for all reps (log m, μ)

- not applicable: no log_m/mu targets in `poscheck` (frame 0 encodes x0 by design)

## PASS — label-permutation null: p95 ≤ max(0.10, 3.3/√n_train), mean ≤ 0.02

- pixels/static x0_m: null p95 = 0.1616 (bound 0.228), null mean = -0.1658; real ridge R² = 0.9999, p ≤ 0.010 (max over seeds)
- pixels/last x0_m: null p95 = 0.1616 (bound 0.228), null mean = -0.1658; real ridge R² = 0.9999, p ≤ 0.010 (max over seeds)
- pixels/clip_seq x0_m: null p95 = 0.1571 (bound 0.228), null mean = -0.1733; real ridge R² = 0.9999, p ≤ 0.010 (max over seeds)
- track/clip x0_m: null p95 = 0.1854 (bound 0.228), null mean = -0.0282; real ridge R² = 1.0000, p ≤ 0.010 (max over seeds)
- dinov2_b_random/static x0_m: null p95 = -0.6067 (bound 0.228), null mean = -1.4906; real ridge R² = 0.9982, p ≤ 0.010 (max over seeds)
- dinov2_b_random/last x0_m: null p95 = -0.6058 (bound 0.228), null mean = -1.4912; real ridge R² = 0.9982, p ≤ 0.010 (max over seeds)
- dinov2_b_random/clip_mean x0_m: null p95 = -0.6058 (bound 0.228), null mean = -1.4912; real ridge R² = 0.9982, p ≤ 0.010 (max over seeds)
- dinov2_b_random/clip_seq x0_m: null p95 = -0.2865 (bound 0.228), null mean = -0.9671; real ridge R² = 0.9967, p ≤ 0.010 (max over seeds)
- dinov2_b/static x0_m: null p95 = -0.6707 (bound 0.228), null mean = -1.4767; real ridge R² = 0.9993, p ≤ 0.010 (max over seeds)
- dinov2_b/last x0_m: null p95 = -0.6736 (bound 0.228), null mean = -1.4606; real ridge R² = 0.9993, p ≤ 0.010 (max over seeds)
- dinov2_b/clip_mean x0_m: null p95 = -0.6736 (bound 0.228), null mean = -1.4606; real ridge R² = 0.9993, p ≤ 0.010 (max over seeds)
- dinov2_b/clip_seq x0_m: null p95 = -0.6938 (bound 0.228), null mean = -1.4844; real ridge R² = 0.9993, p ≤ 0.010 (max over seeds)
- vjepa2_l_random/static x0_m: null p95 = -1.4848 (bound 0.228), null mean = -3.0419; real ridge R² = 0.9987, p ≤ 0.010 (max over seeds)
- vjepa2_l_random/clip_mean x0_m: null p95 = -1.4849 (bound 0.228), null mean = -3.0418; real ridge R² = 0.9987, p ≤ 0.010 (max over seeds)
- vjepa2_l_random/clip_seq x0_m: null p95 = -0.2743 (bound 0.228), null mean = -0.9984; real ridge R² = 0.9980, p ≤ 0.010 (max over seeds)
- vjepa2_l/static x0_m: null p95 = -0.5001 (bound 0.228), null mean = -1.2216; real ridge R² = 0.9992, p ≤ 0.010 (max over seeds)
- vjepa2_l/clip_mean_L08 x0_m: null p95 = -1.0626 (bound 0.228), null mean = -2.3007; real ridge R² = 0.9840, p ≤ 0.010 (max over seeds)
- vjepa2_l/clip_mean_L16 x0_m: null p95 = -0.7680 (bound 0.228), null mean = -1.6519; real ridge R² = 0.9973, p ≤ 0.010 (max over seeds)
- vjepa2_l/clip_mean x0_m: null p95 = -0.5230 (bound 0.228), null mean = -1.2461; real ridge R² = 0.9992, p ≤ 0.010 (max over seeds)
- vjepa2_l/clip_seq_L08 x0_m: null p95 = -0.4488 (bound 0.228), null mean = -1.0414; real ridge R² = 0.9845, p ≤ 0.010 (max over seeds)
- vjepa2_l/clip_seq_L16 x0_m: null p95 = -0.1545 (bound 0.228), null mean = -0.6055; real ridge R² = 0.9930, p ≤ 0.010 (max over seeds)
- vjepa2_l/clip_seq x0_m: null p95 = -0.0726 (bound 0.228), null mean = -0.4856; real ridge R² = 0.9988, p ≤ 0.010 (max over seeds)
- cosmos_cv8/clip_mean x0_m: null p95 = 0.0742 (bound 0.228), null mean = -0.0803; real ridge R² = 0.1583, p ≤ 0.040 (max over seeds)
- cosmos_cv8/clip_seq x0_m: null p95 = 0.0759 (bound 0.228), null mean = -0.0702; real ridge R² = 0.1664, p ≤ 0.020 (max over seeds)
- cosmos_cv8/clip_grid4 x0_m: null p95 = -0.4027 (bound 0.228), null mean = -1.0442; real ridge R² = 0.9861, p ≤ 0.010 (max over seeds)
- state/oracle_xc x0_m: null p95 = 0.1966 (bound 0.228), null mean = -0.0217; real ridge R² = 1.0000, p ≤ 0.010 (max over seeds)
- state/oracle_x x0_m: null p95 = 0.1971 (bound 0.228), null mean = -0.0217; real ridge R² = 1.0000, p ≤ 0.010 (max over seeds)
- state/oracle_fit x0_m: null p95 = -0.0169 (bound 0.228), null mean = -0.0169; real ridge R² = -0.0169, p ≤ 0.010 (max over seeds)
- contact/raw x0_m: null p95 = -0.0165 (bound 0.228), null mean = -0.0169; real ridge R² = -0.0170, p ≤ 0.851 (max over seeds)
- contact/realistic x0_m: null p95 = -0.0119 (bound 0.228), null mean = -0.0172; real ridge R² = -0.0169, p ≤ 0.485 (max over seeds)
- gate: seed-mean null p95 ≤ max(0.10, 3.3/√n_train) AND seed-mean null mean ≤ 0.02 per (input, target); null = ridge with the real fit's alpha on permuted train labels. The √n term: for an input that encodes y almost exactly, a permuted fit returns ≈ ȳ + r·(y − ȳ) with chance correlation r ~ N(0, 1/n_train), so null R² ≈ 2r − r² and its p95 ≈ 3.3/√n_train without any leak
- (info, not gated) single shuffled-label fits: max seed-mean R² = 0.0654 over 30 (input, target) cells

## FAIL (info) — plan v1 rule, informational: shuffled-label control R² ≤ 0.05

- 28/30 (input, target) cells pass (seed-mean R²); failing cells listed
- state/oracle_xc x0_m: R² = 0.0654  ← FAIL
- state/oracle_x x0_m: R² = 0.0654  ← FAIL

## N/A — state/oracle_xc on poscheck: R²(log m) > 0.95, R²(μ) > 0.9

- not applicable: defined for push- and slide-type datasets only (`poscheck` is poscheck)
- (info) state/oracle_xc: R²(x0) = 1.0000
- (info) state/oracle_x: R²(x0) = 1.0000

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

