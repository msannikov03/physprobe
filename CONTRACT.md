# CONTRACT v1.3 (2026-10-22, after the review)

Interface contract between the two halves of the project: the formats of data, features, baseline blocks and results, and the protocol that produces the reported numbers. Newer amendments override older text where they differ; everything else stands. Items in older sections marked "→ v1.3" are superseded below. Any change: agree with the other author first, then bump the version here.

## v1.3 amendments (review fixes, 2026-10-22, private commit fce21d1)

**Release v1.1 edits (2026-10-23), applied in place below.** The mean gate's E uses n_test − 3 and its p-values are called approximate; a "frac" tie prints 1.00; `io.git_commit()` ignores `results/` when deciding `-dirty` and is now used by `probes.run` and `baselines.build`; without clips, `scripts/probe_all.sh` rebuilds only the state and contact blocks and keeps the shipped `pixels` and `track` blocks. No released R² value changes; the printed mean-gate bounds widen slightly.

**Realistic contact (`contact/realistic`, replaces v1 and v1.2).** Per clip, each force channel (fn, ft) goes through three steps: values with |x| < 10⁻³ · (clip max |x|) are set to 0, the channel is divided by its clip max, and additive N(0, 0.05) noise is added on that normalised scale. The third channel is the slip flag 1[slip > 10⁻³ m/s], kept binary and noise-free. 64 × 3 = 192 features, not an oracle. The floor is new. After the cube stops, MuJoCo leaves a friction residual of about 10⁻⁸ N that is proportional to m. Divided by the clip max μmg and z-scored on train, that residual became an exact 1/μ code, which is how the v1.2 block reached R² 0.95 (ln m) / 0.86 (μ) on push. With the floor, what survives is mainly the stop time t_stop = J/(mμg), carried by the ft step and the slip flag. The noise is additive because a multiplicative N(1, σ) keeps tiny values tiny.

**Noise curve (`contact/noise_<p>`, replaces v1.2).** `contact/noise_000`, `noise_010`, `noise_030` and `noise_100` are the noise-free realistic block (floored fn/max, floored ft/max, slip flag) plus additive N(0, p) on every row of all three channels, p ∈ {0, 0.1, 0.3, 1.0}. The same rng seed is used at every level, so the draws are identical up to scale and the curve is monotone in expectation; CV-tuned ridge R² is not monotone by construction, but the released curve is (observed). These blocks are not oracles: they carry no absolute force scale and no raw slip speed, which (|v_xy| = v0 − μgt) would be an exact kinematic oracle. Built by `baselines.build --noise-curve` on push and slide.

**State blocks.** `state/oracle_fit` = [ln v0, a] fitted on the unquantised x(t) (v1.2 fitted the 1-px quantised x(t)). `state/oracle_xc` (unquantised x(t), 64 values) is the linear x(t) read-out reference. It is a ceiling for μ only: ln m = ln J − ln v0 is nonlinear in x(t), and pixels, V-JEPA 2 and contact/raw beat it on ln m. In the "frac of state/oracle_xc" column a value above 1.001 is printed as `n/a (>1)` and a tie as 1.00. `state/oracle_x` (1-px quantised) stays informational.

**Gates (`physprobe/probes/report.py`; replaces v1.2 where they differ).**
- mean: R² ∈ [3·E − 0.01, 0.01] with E = −(1 + n_test/n_train)/(n_test − 3), the expected test R² of the train-mean predictor for Gaussian targets (ȳ_test is independent of SS_test and E[1/χ²_k] = 1/(k − 2)). R²/E is approximately χ²₁, so the printed p-values (≈ 0.08 per seed, ≈ 0.03 for the 3-seed mean) are approximate; the 3-seed value is optimistic because the splits reshuffle the same clips.
- oracle: `state/oracle_xc` R²(ln m) > 0.95 and R²(μ) > 0.9 on push. On slide the gate checks μ only, because m is not identifiable there.
- poscheck: each encoder passes when the pooling chosen on val among its `clip_mean*`, `clip_seq*` and `clip_grid*` modes gives R²(x0) ≥ 0.5. The `clip_mean` value is printed alongside, so an encoder whose global mean loses position (Cosmos) is flagged.
- static, null, slide and contact_label are as in v1.2.
- `table.md` and `gates.md` print a warning when `probes.csv` mixes rows from more than one git commit.

**Canonical run (`scripts/probe_all.sh`, wrapped by `scripts/reproduce.sh`).** `probes.run --fresh`, which deletes `results/<ds>/probes.csv` and `preds/` first. On push and slide the run adds `baselines.build --noise-curve` and three combined inputs: `vjepa2_l/clip_seq+contact/realistic`, `dinov2_b/clip_seq+contact/realistic` and `vjepa2_l/clip_mean_L16+contact/realistic`. Datasets run in the order slide, poscheck, push, so the push report sees the other two for its cross-dataset gates. Extra `probes.csv` columns beyond v1: `r2_val`, `r2_train`, `oracle`, `p_value`, `null_mean`.

**Feature extraction (`physprobe/feats/`; replaces v1 where it differs).**
- dtype: bf16 for pretrained reps and fp32 for random-init reps (`--dtype auto`). In fp32, TF32 is off for matmuls and cuDNN convolutions. A random network's between-clip variance is tiny, so bf16 rounding noise was of the same size as the signal.
- DINOv2 runs every forward pass on exactly 64 frames; the last chunk is padded by repeating its last frame and the padding outputs are dropped. A frame's feature therefore does not depend on `--batch` or on which modes share the pass.
- Stored `X`: float16 for every `clip_seq*` and `clip_grid*` mode and for `pixels/clip_seq`, float32 otherwise. Extraction refuses values that do not fit float16.
- `info` keys added to v1: `dtype`, `init` (pretrained or random), `seq_shape` (flatten order of sequence modes), `code_hash` (sha1 of `encoders.py` + `extract.py`, 12 hex characters), `fp32_precision`, `device`, `batch`, `torch`, `transformers`, `sec_per_clip`, `gpu_sec_per_clip`.
- Resume: an existing feature file is reused only when its ids, dtype, revision and `code_hash` match the current run; otherwise it is re-extracted. `--ignore-code-hash` relaxes only the hash.
- Flatten orders: V-JEPA 2 `clip_seq*` is (slot, channel), 32 × 1024, slot k = frames 2k and 2k+1. DINOv2 `clip_seq` is (frame, feature), 16 × 1536, frames 0, 4, …, 60.

**Cosmos (`cosmos_cv8`, moved from optional to the minimum).** NVIDIA Cosmos-Tokenize1 CV8×8×8 continuous video tokenizer, `nvidia/Cosmos-Tokenize1-CV8x8x8-720p`, `encoder.jit` only (TorchScript), bf16 only because the graph does not run in fp32. Input: the 64 frames plus frame 63 repeated once (65 frames = 1 + 8·8) at native 256 × 256, scaled x/127.5 − 1. Latent: (16 channels, 9 slots, 32, 32). Slot 0 encodes frame 0 alone and slot j ≥ 1 ends at frame 8j; the temporal convolutions are causal. Raw latents are used, without the shipped `mean_std.pt` normalisation. Modes, with no `static` or `last`:
- `clip_mean`: mean over (slot, row, col), 16 values, float32.
- `clip_seq`: spatial mean per slot, (9, 16) slot-major, 144 values, float16.
- `clip_grid4`: adaptive average pool of each slot to 4 × 4, (16, 9, 4, 4) channel-major, 2304 values, float16. Per-slot slicing must reshape to (16, 9, 4, 4) and index axis 1.

**Provenance.** `io.git_commit()` returns the short HEAD with `-dirty` appended when tracked files outside `results/` differ from HEAD (the rule of `git describe --dirty`, with the tracked canonical outputs excluded so that a canonical run does not mark the next one dirty). It is written to `dataset.json`, to feature `info` and to every `probes.csv` row (in `probes.run` and `baselines.build` from release v1.1 on; before that they stamped the HEAD without the dirty check). The first generation of a dataset stores `provenance` = {`git_commit`, `sim_code_sha256` (sha256 of `physprobe/sim/*.py`), `created`, `parts_tag`} in `data/<ds>/.parts/config.json`, and `dataset.json` copies `git_commit`, `created` and `sim_code_sha256` from it. Resuming a dataset with clips left to generate is refused when the sim code hash has changed; `--overwrite` regenerates. A dataset whose `config.json` predates provenance can be re-assembled but not extended; its `dataset.json` keeps the old `git_commit` and has `sim_code_sha256: null` with a `provenance` note. The released datasets predate these keys: their `dataset.json` carries `git_commit` c3dac34 and no `sim_code_sha256`. Hashes inside results/, the feature info and the notebook refer to the private development history that was squashed into the public v1.0 commit d41bcfa; its physprobe/ and tests/ trees are identical. The data were generated by the simulator code first committed as a124216 (physics identical to the release).

**Seeds and splits (added to v1).** Base seeds are assigned by dataset name first: `push_ood` 200000 and `push_nuis` 400000, so they never reuse push seeds. Otherwise by scenario: push 0, slide 100000, poscheck 300000. Datasets named `*_ood` get split `ood`, `smoke*` get split `smoke`, and all others the canonical `seed % 20` split.

**Validation (`io.validate`, added to v1).** Dataset level: `ok_frac` and `in_frame_frac` ≥ 0.95; the required camera keys; for push and slide, train/val/test fractions within 0.02 + 10/n of 70/15/15 and a single `frame0_hash`. Per row: split equal to the canonical split of its seed; values inside the `dataset.json` ranges; seed = base_seed + id, with (m, μ, x0) reproduced by `default_rng(seed)`; push v0 = J/m; contact fn > 0 in every row and |fn/(mg) − 1| ≤ 10⁻⁴; state x non-decreasing on push and slide; state row 0 = [x0, 0, rest_z, v0, 0, 0]. Sampled clips (10 random plus min m and max m): frame 0 is not constant, is identical across push and slide clips, and the last frame differs from frame 0 when `moved_px` ≥ 3. The CLI exits with code 1 on any problem.

**Environment.** `env.sh` is portable. `PHYSPROBE_ROOT` defaults to the clone. `SIM_PY` and `FEAT_PY` default to `python3` and must be exported when the environments differ. `MUJOCO_GL=egl` is set on Linux only. `requirements-feats.txt` (torch, transformers, huggingface_hub plus the probe requirements) specifies `FEAT_PY`.

**Corrections to earlier text, applied in place below.**
- v1.1 friction model: the stopping distance matches the semi-implicit Euler value v0²/(2μg) − v0·dt/2 to about 10⁻⁴ relative. The gap of up to 1.3 % to the continuous formula is that Euler offset (≤ 0.8 mm), not a physics error, and `tests/test_sim.py` asserts 5 %.
- v1.1 camera: the tracker mask includes the visible side faces, so neither pixel-scale pair maps the tracker centroid to metric x without bias.
- v1.2 protocol sensitivity: the Cosmos-grid comparison was PCA-256 against no PCA, not 256 against 512, and the "within 0.013 on all of slide" claim was false for slide ln m.

## v1.2 amendments (probes-v1, 2026-10-22 late, after the cross-check of side B)

**Extra baseline blocks (Ilya, `physprobe/baselines/build.py`).** `state/oracle_xc` = unquantised x(t) of the cube centre, 64 values, used by the oracle gate and by "fraction of oracle" in tables (→ v1.3: a ceiling for μ only, not for ln m). `state/oracle_x` (x(t) quantised to 1 px with the `_com` scale) stays as an informational row only, because block-averaged pixels keep sub-pixel position and beat it. `state/oracle_fit` = [log v0, a] fitted to x(t) (→ v1.3: on the unquantised x(t)). `contact/noise_<p>` = realistic contact with force noise σ = p·m_ref·g, p ∈ {0, 0.1, 0.3, 1.0}, m_ref = geometric mean of the mass range (→ v1.3: additive N(0, p) on the normalised block). Colour tracker mask = saturation > 0.4 and chroma > 60 (includes the shaded cube sides). Grey = luma.

**Probe protocol (canonical, `physprobe/probes/run.py`).** Exclude clips with `ok = false` (count in `n_excluded`). z-score on train; PCA to 256 components (fitted on train) when D > 2048; ridge with α ∈ logspace(−3, 6, 28) chosen by contiguous 5-fold CV on train only; val is used only to choose the mode/layer per representation; test is reported. Combined inputs "A + B": each block z-scored on train and scaled by 1/√D_block. Probes: `ridge`, `mean`, `shuffle` (one permutation), `perm` (100 label permutations fitted with the ridge's chosen α; the csv row stores the null p95 in `r2`, plus `null_mean` and `p_value`), optional `mlp`.

**Gates (`physprobe/probes/report.py`).** mean: R² ∈ [−3/n_test − 0.01, 0.01] (→ v1.3: [3·E − 0.01, 0.01]). static: R² < 0.05 for every representation (push/slide only). null: permutation p95 ≤ max(0.10, 3.3/√n_train) and null mean ≤ 0.02 (replaces the original single-shuffle ≤ 0.05 rule, which fails by chance correlation when n_train is small and the input encodes the target exactly). oracle: `state/oracle_xc` R²(log m) > 0.95 and R²(μ) > 0.9 on push; μ only on slide. slide: R²(log m) < 0.05 for every video/pixel input. poscheck: R²(x0) ≥ 0.5 per encoder (→ v1.3: for its val-chosen pooling). The original v1 verdicts (quantised oracle ≥ 0.9, shuffle ≤ 0.05) are printed next to the new ones as informational.

**Protocol sensitivity (documented, not a bug; corrected in v1.3).** The quick-look pass (`scripts/quicklook_probes.py`: PCA to 512 only when D > 4096, α ≤ 10⁵ by GCV), run on the 22 Oct features before the v1.3 fixes, agreed with the canonical pipeline within 0.013 on every input of ≤ 2048 dims, and within 0.0012 on slide μ. On slide ln m the high-dimensional sequence inputs differed by up to 0.026 (V-JEPA 2 `clip_seq_L16`: −0.030 canonical, −0.056 quick-look), with every value ≤ 0. On push R²(μ), inputs that go through PCA lost at 256 components against 512: V-JEPA 2 `clip_seq` −0.008, `clip_seq_L08` −0.022, DINOv2 `clip_seq` −0.047, random-init DINOv2 `clip_seq` −0.072, random-init V-JEPA 2 `clip_seq` −0.109. The Cosmos grid (2304 dims) is reduced to 256 components in the canonical run and not reduced at all in the quick-look (D ≤ 4096): 0.682 against 0.748. Both runs are regenerated by `scripts/reproduce.sh`, and the v1.3 rerun changes the random-init and DINOv2 features, so `docs/RESULTS_2026-10-22.md` carries the re-measured gaps. The canonical numbers are the reported ones.

## v1.1 amendments (sim-v1, 2026-10-22)

These override the v1 text where they differ. Everything else in v1 stands.

**Friction model (sim-v1).** MuJoCo's soft friction cones make a fast-sliding cube hop (normal force up to 10 mg, tilt up to 11° at mu = 0.8), so sim-v1 uses MuJoCo dry friction instead: `dof_frictionloss = max(mu, 0.01)·m·g` on the cube's x and y degrees of freedom (`solreffriction 0.002 1`), and the floor contacts are frictionless (`condim=1`). Kinematics are Coulomb: a = mu·g while sliding. The final distance matches the semi-implicit Euler value d = v0²/(2·mu·g) − v0·dt/2 to about 10⁻⁴ relative; the v0·dt/2 term (≤ 0.8 mm, up to 1.3 % of the shortest slides) is the integrator's offset, not a physics error. `tests/test_sim.py` asserts 5 % against the continuous formula. `dataset.json` records this under `friction_model`.

**Contact channel (`contact_<id>.npy`, float32 (64,3) = [fn_N, ft_N, slip_mps]).** fn = sum over cube–floor contacts of `mj_contactForce` component 0 (= m·g in every row). ft = magnitude of the floor's tangential reaction on the cube = |dof friction force| (= mu·m·g while sliding, ≈ 0 once stopped). slip = mean |v_xy| of the cube centre over the 20 substeps of the frame interval. fn and ft are therefore an oracle for (m, mu); the `contact/realistic` baseline must normalise per clip and add noise (→ v1.3 defines the block).

**State (`state_<id>.npy`).** Row 0 holds the velocity just after the push; frame 0 is rendered at rest before the velocity is set. Rest height z = 0.0498922 m for every mass (`mj_setConst` after setting mass keeps resting penetration identical across m; frame 0 has one hash across smoke_push, push and slide).

**Stability keys.** Under sim-v1 `max_tilt_deg = max_yaw_deg = 0` always, so `ok` reduces to `in_frame ∧ no NaN`.

**Visibility keys.** `moved_px` = pixel distance between the projected cube centre at frame 0 and frame 63. `frames_moving` = number of frames k in 1..63 with centre speed > 1 mm/s. `t_stop_s` = first substep time with speed < 1 mm/s (simulated past 1 s if needed). `in_frame` = all 8 cube corners inside the image in all 64 frames. `z_pen_max_m` = maximum contact penetration. Extra optional meta keys: `moving_at_end` (bool), `frame0_hash` (16 hex chars).

**Camera (`dataset.json.camera`, identical in every dataset).** Perspective camera, `pos [0.57, 0, 2.799]`, `xyaxes [1,0,0, 0,1,0]`, `fovy 30`, image 256×256, floor view 1.5 m wide, world origin at 12 % of the width. Pixel convention: u = column, v = row, pixel centres at integer indices, row 0 at the top; world +x → +u, world +y → −v; `u = origin_px[0] + x / m_per_px`. Three scales are given: `m_per_px_floor` / `origin_px` (floor plane, z = 0), `m_per_px_com` / `origin_px_com` (cube centre height), `m_per_px_top` / `origin_px_top` (top face). The colour tracker's mask (v1.2) includes the visible side faces, so its centroid is offset from the projected top-face centre by +1.2 px at x = 0 to −1.1 px at x = 0.94 m, about a 1.4 % compression of the path. Linear probes absorb this. Use the `_top` pair only with a top-face-only mask; otherwise treat tracker units as arbitrary.

**Previews.** `previews/<dataset>/grid.mp4` (16 fps, 4× slow motion, labels burned in) and `previews/<dataset>/sheet.png`. Not in git.

**Generation.** `scripts/gen_all.sh` (workers capped at 10, nice 10). Each dataset keeps its config in `data/<ds>/.parts/config.json`; a mismatched config is refused, `--overwrite` regenerates. Extra `dataset.json` keys: `friction_model, sim_version, base_seed, split_mode, g, path_m, rest_z_m, v_stop_mps, cube_rgba, floor_rgba, frame0`.

**Datasets generated 2026-10-22 on our GPU machine** (`$PHYSPROBE_ROOT/data/`): push n=2000, slide n=1000, poscheck n=300, smoke_push n=20; all `ok_frac = 1.0`, `in_frame_frac = 1.0`, canonical splits 70/15/15, validate reports no problems.

## v1 text (the original contract, 2026-10-22, verbatim, in Russian)
### 4.1 Интерфейсный контракт (CONTRACT v1, фиксируется в T0)

Любое изменение контракта — только через сообщение второму и повышение версии в `CONTRACT.md`. Потребитель игнорирует незнакомые ключи.

**Корень и окружение**
- `ROOT = $PHYSPROBE_ROOT`. Если переменная не задана, корень — это корень клона. На GPU-машине: `$PHYSPROBE_ROOT`, он же git-клон.
- `data/`, `features/`, `.cache/`, `results/*/preds/` в git не попадают (gitignore).
- `env.sh` (на GPU-машине делать `source` всегда):
  ```bash
  export PHYSPROBE_ROOT=/path/to/physprobe
  export HF_HOME=$PHYSPROBE_ROOT/.cache/hf
  export TORCH_HOME=$PHYSPROBE_ROOT/.cache/torch
  export PIP_CACHE_DIR=$PHYSPROBE_ROOT/.cache/pip
  export XDG_CACHE_HOME=$PHYSPROBE_ROOT/.cache
  export MPLCONFIGDIR=$PHYSPROBE_ROOT/.cache/mpl
  export TMPDIR=$PHYSPROBE_ROOT/.cache/tmp
  export MUJOCO_GL=egl
  export SIM_PY=/path/to/venv-sim/bin/python   # mujoco, без torch
  export FEAT_PY=/path/to/venv-feats/bin/python    # torch, transformers 5.3, sklearn 1.8, без mujoco
  export PYTHONPATH=$PHYSPROBE_ROOT
  mkdir -p $HF_HOME $TORCH_HOME $PIP_CACHE_DIR $MPLCONFIGDIR $TMPDIR
  ```
- Оба venv используем только на чтение, в них ничего не ставим. `HF_HUB_ENABLE_HF_TRANSFER` не выставляем.

**Правила импортов**
- `physprobe/io.py`, `paths.py`: только numpy и stdlib.
- `physprobe/sim/*`: numpy + mujoco (SIM_PY).
- `physprobe/feats/*`: torch + transformers (FEAT_PY), без mujoco.
- `physprobe/baselines/*`, `physprobe/probes/*`, `physprobe/tools/*`: numpy, scipy, sklearn, pandas, matplotlib, без torch и mujoco. Запускаются в FEAT_PY на GPU-машине и в локальном CPU-venv.

**Данные: `data/<dataset>/`**
Наборы: `push`, `slide`, `poscheck`, `smoke_push`, `dummy`; позже `push_nuis`, `push_ood`.

| Файл | Формат |
|---|---|
| `clip_<id>.npy` | uint8 `(64,256,256,3)` RGB; кадр k в момент t = k/64 с; кадр 0 рендерится в покое до задания скорости |
| `contact_<id>.npy` | float32 `(64,3)`, колонки `[fn_N, ft_N, slip_mps]`: fn — сумма нормальных сил по всем контактам куб–пол (`mj_contactForce`, компонента 0), ft — сумма норм компонент (1,2), slip — \|v_xy\| центра масс; строка k — среднее по 20 подшагам интервала [k/64, (k+1)/64) |
| `state_<id>.npy` | float32 `(64,6)`, центр масс `[x,y,z,vx,vy,vz]` в мировых координатах в моменты кадров |
| `meta.jsonl` | одна JSON-строка на клип, отсортирована по id (ключи ниже) |
| `dataset.json` | см. ниже |

Ключи `meta.jsonl`:
- `id`: строка из 6 цифр, `"000123"`.
- `scenario`: `push | slide | poscheck | dummy`.
- Физика: `m`, `mu`, `J` (null для slide), `v0`, `x0_m` (для poscheck — случайное положение).
- `seed`, `split`: `train | val | test | ood | smoke`.
- Видимость: `in_frame` (bool), `disp_m`, `moved_px`, `frames_moving`, `t_stop_s`.
- Устойчивость: `max_tilt_deg`, `max_yaw_deg`, `z_pen_max_m`.
- `ok`: bool, `= in_frame ∧ tilt<5° ∧ yaw<5° ∧ нет NaN`.

`dataset.json`:
- `name`, `scenario`, `n`, `T=64`, `H=W=256`, `fps=64`, `dt=1/1280`, `substeps=20`.
- `contact_channels`, `state_channels`.
- `ranges {m:[lo,hi,"loguniform"], mu:[lo,hi,"uniform"]}`, `J`, `v0_fixed`, `cube_half=0.05`, `floor_friction=0.01`.
- `camera {pos, fovy, m_per_px_floor, origin_px}`: нужно для пиксельного оракула.
- `mujoco_version`, `git_commit`, `created`.

**Конфигурация по умолчанию:** m ∈ [0.5, 2] (лог-равномерно), μ ∈ [0.2, 0.8], J = 1.0 Н·с, slide v₀ = 1.2 м/с. Генератор принимает флаги для всех этих параметров.

**Сиды и разбиения**
- Сид клипа = `base_seed + index`. `rng = np.random.default_rng(seed)`; сначала сэмплируется m, затем μ.
- Каноническое разбиение по `seed % 20`: 0–13 → train, 14–16 → val, 17–19 → test.
- `base_seed`: push 0, slide 100000, poscheck 300000, smoke_push 0 (n = 20).
- `io.split_ids(meta, seed)`: seed 0 возвращает каноническое разбиение; seeds 1 и 2 перемешивают train/val/test в пропорции 70/15/15. Пробы гоняются на seeds {0, 1, 2}.

**API `io.py`** (пишет Михаил в T0):
- Пути и метаданные: `root()`, `dataset_dir(name)`, `write_clip(ds_dir,id,clip,contact,state)` (атомарно), `write_meta`, `write_dataset_json`, `load_meta(name)`.
- Чтение данных: `load_clip(name,id,mmap=True)`, `load_contact`, `load_state`.
- Признаки: `save_features(name, rep, mode, ids, X, info)`, `load_features(name, rep, mode) -> (ids, X, info)`, `split_ids(meta, seed)`.
- `validate(name)`: проверяет наличие файлов, dtype и shape, отсутствие NaN, диапазоны, доли разбиений, долю ok/in_frame и идентичность кадра 0 (max\|Δ\| = 0 для 10 случайных клипов + клипов с min m и max m).
- CLI: `python -m physprobe.io validate <ds>`.

**Признаки: `features/<dataset>/<rep>/<mode>.npz`**
- Ключи: `ids` (`<U16`, порядок meta), `X` (float32; float16 для `clip_seq`), `info` (JSON-строка: rep, hf_id, revision, mode, layer, pooling, n_frames, input_res, sec_per_clip, git_commit).
- **Нейросетевые представления (Михаил):** `vjepa2_l`, `vjepa2_l_random`, `dinov2_b`, `dinov2_b_random`; [ОПЦ] `cosmos_cv8`, `siglip2_b`, `dinov2_l`.
  - `static`: кадр 0. Для V-JEPA это кадр 0, повторённый 64 раза. Если хеши кадров 0 у всех клипов равны, считаем один раз и размножаем.
  - `last`: последний кадр. Только для покадровых моделей; для V-JEPA [ОПЦ].
  - `clip_mean`: глобальное среднее, D = 1024 / 1536.
  - `clip_seq`: для V-JEPA — среднее по пространству в каждом из 32 временных слотов, `(32×1024)` в плоском виде; для DINOv2 — 16 равномерных кадров × [CLS ‖ среднее патчей], `(16×1536)`.
  - Слои V-JEPA: `clip_mean_L08`, `clip_mean_L16`, `clip_seq_L08`, `clip_seq_L16`. Без суффикса — последний слой (L24).
- **Опорные уровни (Илья):**
  - `pixels/{static,last,clip_seq}`: серый 32×32; для `clip_seq` — 8 временных бинов, D = 8192.
  - `track/clip`: центроид куба по порогу цвета, 64×2.
  - `state/oracle_x`: x(t) в пикселях, квантовано до 1 px, 64 значения.
  - `state/oracle_fit`: `[log v0, a]`, подгонка по x(t).
  - `contact/raw`: 192 признака, ОРАКУЛ.
  - `contact/realistic`: fn и ft, нормированные на максимум по клипу, шум 5 %, плюс бинарный флаг slip > 1 мм/с; 192 признака.
- Комбинированные входы (например, `vjepa2_l/clip_seq + contact/realistic`) собираются во время пробы из блоков, стандартизованных по отдельности. Каждый блок масштабируется к единичной суммарной дисперсии.

**Результаты: `results/<dataset>/`**
- `probes.csv` (long format), колонки: `dataset, rep, mode, inputs, target(log_m|mu|x0_m), probe(ridge|mlp|mean|shuffle), seed, n_train, n_val, n_test, n_excluded, feat_dim, pca_dim, alpha, r2, rmse, git_commit, timestamp`.
- `preds/<rep>__<mode>__<probe>__<target>__s<seed>.npz`: ids, y_true, y_pred. В git не попадает.
- `table.md`: среднее ± std по 3 разбиениям.
- `gates.md`: санити-гейты PASS/FAIL.
- `figs/headline.png`, `figs/scatter_<target>.png`, `figs/r2_by_moved_px.png`.

