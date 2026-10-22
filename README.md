# physprobe: do frozen video encoders and a generative video tokenizer see hidden object physics?

A visually identical cube is pushed across a floor in MuJoCo with a fixed impulse. Its mass and friction never show in a single frame; they only change how it moves. We ask whether frozen video representations encode these hidden parameters linearly, and whether a touch signal adds to them. We probe V-JEPA 2 (a predictive self-supervised video model), DINOv2 (per-frame image features), random-weight copies of both, and the latent of a generative video tokenizer (NVIDIA Cosmos-Tokenize1). Every probe is compared against raw pixels, a colour tracker, the true trajectory and the floor contact forces, with identifiability controls.

Course project, Modern Methods and Algorithms of Generative AI (Skoltech, 2026). Mikhail Sannikov, Ilya Mikhalchuk.

![Two visually identical cubes pushed with the same impulse: a light one (top, 0.60 kg) slides far, a heavy one (bottom, 1.80 kg) barely moves. Frame 0 is pixel-identical; only the motion carries mass and friction.](assets/demo_strip.png)

## Answer

Mass and friction are linearly readable from frozen video features: V-JEPA 2 reaches test R² 0.997 for ln m and 0.977 for μ on 2000 push clips, and the controls show that the probe reads the cube's kinematics, not artefacts. Raw pixels do as well (0.999 / 0.982), so this clean scene does not rank encoders; the Cosmos tokenizer latent is weaker at the poolings we tried (0.938 / 0.670), and a realistic touch signal reaches only 0.521 / 0.452 on its own, adds little on top of V-JEPA 2 (+0.005 for μ) and more on top of the weaker DINOv2 sequence (+0.11).

![Test R² for ln m and μ per representation on push, and ln m on the slide control](results/push/figs/headline.png)


### Results at a glance (push, test R² over 3 splits; full tables in `results/push/table.md`)

| input | R²(ln m) | R²(μ) |
|---|---|---|
| raw pixels, 32×32 grey × 8 time bins | 0.999 | 0.982 |
| V-JEPA 2 ViT-L, block 16, mean-pooled | 0.997 | 0.977 |
| V-JEPA 2 ViT-L, random weights, sequence | 0.997 | 0.883 |
| DINOv2-B, 16-frame sequence | 0.983 | 0.842 |
| DINOv2-B, mean over frames | 0.838 | 0.176 |
| Cosmos CV8×8×8 tokenizer, 4×4 grid × 9 slots | 0.938 | 0.670 |
| realistic touch signal alone | 0.521 | 0.452 |
| true trajectory x(t) (reference) | 0.976 | 1.000 |
| exact floor contact (oracle) | 1.000 | 1.000 |

Controls: every first-frame input scores ≈ 0 (the frame is identical across clips); on the fixed-speed slide set every video input scores ≤ 0 for mass (no leak); pooled features of the pretrained encoders still locate the cube (R² 0.99 on the position check). All sanity gates pass.

- Write-up with all claims, hypotheses, limitations and figures: [`docs/RESULTS_2026-10-22.md`](docs/RESULTS_2026-10-22.md).
- Full tables, every pooling mode, shuffled-label controls: [`results/push/table.md`](results/push/table.md), [`results/slide/table.md`](results/slide/table.md), [`results/poscheck/table.md`](results/poscheck/table.md).
- Sanity gates (mean predictor, static frame, permutation null, state reference, slide leak, position check): [`results/push/gates.md`](results/push/gates.md) and the same file per dataset.
- Features for CPU-only reproduction (1.5 GB) and the per-clip contact and state signals: Hugging Face dataset [`businesslion/physprobe-features`](https://huggingface.co/datasets/businesslion/physprobe-features). With it, `scripts/reproduce.sh --from probes` reproduces every number in about 10 minutes without a GPU (see the dataset card).
- Interface contract (data, feature and result formats; probe protocol): [`CONTRACT.md`](CONTRACT.md).
- Proposal, in Russian: [`docs/proposal_v2.md`](docs/proposal_v2.md).

## Demo

`notebooks/demo_two_cubes.ipynb` (executed, outputs embedded) takes the two cubes above, shows their frames, trajectories and contact forces, and runs the canonical ridge probe on three inputs. From the identical first frame the probe can only return the training mean; from the clip it recovers both masses; adding the realistic touch signal changes little.

| input | pred. m A (kg) | pred. μ A | pred. m B (kg) | pred. μ B |
|---|---|---|---|---|
| truth | 0.605 | 0.212 | 1.798 | 0.233 |
| frame 0 (V-JEPA 2 static) | 0.989 | 0.499 | 0.989 | 0.499 |
| clip (V-JEPA 2, block 16) | 0.610 | 0.217 | 1.761 | 0.237 |
| clip + realistic touch | 0.627 | 0.193 | 1.767 | 0.224 |

Re-execute on a machine with the data and features: `$FEAT_PY -m jupyter nbconvert --to notebook --execute --inplace notebooks/demo_two_cubes.ipynb` (CPU, about 10 s).

## Environment

The pipeline uses three Python environments, because MuJoCo and torch pin different numpy versions.

| variable | requirements | contents | used for |
|---|---|---|---|
| `SIM_PY` | `requirements-sim.txt` | MuJoCo 3.16, numpy, pillow | simulation, data generation |
| `FEAT_PY` | `requirements-feats.txt` | torch 2.10 (CUDA), transformers 5.3, huggingface_hub, plus the probe requirements | feature extraction and probes |
| probe venv (optional) | `requirements-probe.txt` | numpy, scipy, scikit-learn, pandas, matplotlib | probes and reports on CPU, no torch |

```bash
python3.11 -m venv .venv-sim   && .venv-sim/bin/pip install -r requirements-sim.txt
python3.13 -m venv .venv-feats && .venv-feats/bin/pip install -r requirements-feats.txt
export SIM_PY=$PWD/.venv-sim/bin/python FEAT_PY=$PWD/.venv-feats/bin/python
source env.sh
```

`env.sh` sets `PHYSPROBE_ROOT` (default: this clone), keeps every cache under `$PHYSPROBE_ROOT/.cache`, sets `MUJOCO_GL=egl` on Linux and puts the clone on `PYTHONPATH`. It defaults `SIM_PY` and `FEAT_PY` to `python3`, so export both when your environments differ. The scripts source `env.sh` themselves and keep values you exported. On our compute machine the setup is:

```bash
export PHYSPROBE_ROOT=/path/to/physprobe               # the clone itself
export SIM_PY=/path/to/venv-sim/bin/python   # Python 3.11, mujoco 3.16, no torch
export FEAT_PY=/path/to/venv-feats/bin/python    # Python 3.13, torch 2.10, transformers 5.3, sklearn 1.8
source $PHYSPROBE_ROOT/env.sh
```

Model weights are downloaded from Hugging Face on first use at pinned revisions; none of them is gated. Feature extraction needs a CUDA GPU. Disk: about 40 GB for the generated clips and 1.5 GB for the features. `ffmpeg` is optional and only used for preview videos.

## Reproduce

```bash
scripts/reproduce.sh                  # everything, about 45 min on one RTX 4090
scripts/reproduce.sh --from probes    # rerun probes and reports on existing features
```

| step | what it runs | time | hardware |
|---|---|---|---|
| `tests` | `tests/test_sim.py`: scene physics, frame-0 identity, generator refusals | short | CPU |
| `data` | `scripts/gen_all.sh`: push 2000, slide 1000, poscheck 300 clips, then validation | ~2 min | CPU, 10 workers |
| `features` | `scripts/extract_all.sh push slide poscheck`: DINOv2, V-JEPA 2, both random-init controls, then Cosmos | ~25 min, plus ~3 min for Cosmos | one RTX 4090 |
| `validate` | `python -m physprobe.io validate` for the three datasets | seconds | CPU |
| `probes` | `scripts/probe_all.sh` for slide, poscheck, push: baselines and noise curve, ridge probes from scratch, combined video + touch inputs, tables, gates, figures | ~10 min | CPU, 8 threads |
| `quicklook` | `scripts/quicklook_probes.py`: an independent second implementation of the probes, used as a cross-check | ~3 min | CPU |

The script stops at the first failing step, prints the time of each step and can resume with `--from <step>`. Finished clips and features whose code hash matches are skipped. Outputs go to `results/<dataset>/` (`probes.csv`, `table.md`, `gates.md`, `figs/`) and `results/quicklook/`. Generated clips and per-clip predictions are not in git. The features (1.5 GB) and the contact/state signals are published on Hugging Face (`businesslion/physprobe-features`); download them into `features/` and `data/` and run `scripts/reproduce.sh --from probes` to reproduce every number on a CPU.

## Tests

```bash
python3 -m pytest tests/ -q           # skips every module whose dependencies the interpreter lacks
$SIM_PY  tests/test_sim.py             # needs mujoco
$FEAT_PY tests/test_feats.py           # needs torch + transformers; PHYSPROBE_GPU_TESTS=1 adds two GPU tests on real weights
```

`tests/test_io.py` needs only numpy. `tests/test_pipeline.py` runs baselines, probes and the report end to end on a synthetic dataset and needs the probe requirements. `tests/test_sim.py` needs MuJoCo, and `tests/test_feats.py` needs torch with tiny random configs; its two real-weight tests also need CUDA. pytest is not pinned in the requirement files.

## Repository layout

```
physprobe/io.py, paths.py      data and feature I/O, dataset validation (numpy + stdlib)
physprobe/sim/                 MuJoCo scene, clip generator, dataset statistics       (SIM_PY)
physprobe/feats/               frozen encoders and the resumable feature extractor     (FEAT_PY)
physprobe/baselines/build.py   pixel, tracker, state and contact baseline blocks
physprobe/probes/run.py        ridge probes, permutation null, combined inputs
physprobe/probes/report.py     tables, gates, figures
physprobe/tools/make_dummy.py  synthetic dataset for CPU-only tests
scripts/                       reproduce.sh, gen_all.sh, extract_all.sh, probe_all.sh, quicklook_probes.py
notebooks/demo_two_cubes.ipynb two-cubes demo (executed)
assets/                        images used by this README
tests/                         test_io, test_sim, test_feats, test_pipeline
results/{push,slide,poscheck}/ canonical probes.csv, table.md, gates.md, figs/
results/quicklook/             cross-check implementation
docs/RESULTS_2026-10-22.md     results write-up
docs/proposal_v2.md            project proposal (Russian)
CONTRACT.md                    interface contract and probe protocol
env.sh, requirements-*.txt     environment
```

## Authors and contributions

- **Mikhail Sannikov:** simulation and data generation, feature extraction for all encoders including Cosmos, the interface contract, the cross-check implementation, the post-review fixes and the documentation.
- **Ilya Mikhalchuk:** baseline blocks, probes and permutation null, the report with tables, gates and figures, the pipeline tests, and proposal v2.
- **Both:** the experimental design.

## Models, data and licences

| component | source | licence |
|---|---|---|
| V-JEPA 2 ViT-L | `facebook/vjepa2-vitl-fpc64-256` | MIT (Hugging Face model card) |
| DINOv2-base | `facebook/dinov2-base` | Apache-2.0 |
| Cosmos-Tokenize1 CV8×8×8 | `nvidia/Cosmos-Tokenize1-CV8x8x8-720p` | NVIDIA Open Model License |
| MuJoCo 3.16 | `mujoco` on PyPI | Apache-2.0 |

Cosmos-Tokenize1 is licensed by NVIDIA Corporation under the NVIDIA Open Model License; the licence requires this notice with any redistribution of the model or of a model derived from it. Model weights are downloaded at run time and are not redistributed here. All data is generated by the simulator in this repository; no external dataset is used. Project code licence: MIT (see `LICENSE`).

## References

1. A. Mehta, R. Baviskar. CALIPER: Clean Scenes Cannot Rank Physical Inference in Pretrained Visual Representations. arXiv:2609.08250, 2026.
2. M. Assran et al. V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning. arXiv:2506.09985, 2025.
3. M. Oquab et al. DINOv2: Learning Robust Visual Features without Supervision. arXiv:2304.07193, 2023.
4. NVIDIA (N. Agarwal et al.). Cosmos World Foundation Model Platform for Physical AI. arXiv:2501.03575, 2025. Tokenizer code: github.com/NVIDIA/Cosmos-Tokenizer.
5. E. Todorov, T. Erez, Y. Tassa. MuJoCo: A physics engine for model-based control. IROS 2012.
