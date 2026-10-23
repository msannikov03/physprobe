#!/usr/bin/env bash
# Reproduce every number and figure in docs/RESULTS_2026-10-22.md, end to end.
#   scripts/reproduce.sh                   # all steps
#   scripts/reproduce.sh --from probes     # resume at a step; also the CPU-only path on the Hugging Face
#                                          # download (features/ + data/ without clips), about 15 min
# Steps, in order:
#   tests      $SIM_PY tests/test_sim.py (scene physics, frame-0 identity, generator refusals)
#   data       scripts/gen_all.sh: push 2000, slide 1000, poscheck 300 clips + validate   (~2 min, CPU)
#   features   scripts/extract_all.sh push slide poscheck: DINOv2, V-JEPA 2, both random-init
#              controls, Cosmos CV8                         (~40 min, one RTX 4090)
#   validate   python -m physprobe.io validate for push, slide, poscheck
#   probes     scripts/probe_all.sh slide, poscheck, push: baselines (+ noise curve), --fresh ridge
#              probes with the three video + contact/realistic inputs, tables, gates, figures  (~10 min, CPU)
#              Without clips, only the state/contact blocks are rebuilt; pixels/track come from features/.
#   quicklook  scripts/quicklook_probes.py: the independent cross-check implementation  (~3 min, CPU)
#              Without clips, its pixel/track rows also come from features/.
# Check a rerun against the committed numbers: python scripts/compare_results.py <ref>/probes.csv results/push/probes.csv
# Interpreters (README, "Environment"): SIM_PY = requirements-sim.txt (mujoco), FEAT_PY =
# requirements-feats.txt (torch + CUDA). The probes run in FEAT_PY unless PROBE_PY is set.
# env.sh defaults SIM_PY and FEAT_PY to python3: export them before running when your venvs differ.
# Every step is resumable (finished clips and features with a matching code hash are skipped).
# The script stops at the first failing step and prints the time of every step.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$here/env.sh"
cd "$here"
PROBE_PY=${PROBE_PY:-$FEAT_PY}
DATASETS=(push slide poscheck)
STEPS=(tests data features validate probes quicklook)

usage() { echo "usage: $0 [--from STEP]   STEP in: ${STEPS[*]}" >&2; exit 2; }
FROM=${STEPS[0]}
case "${1:-}" in
  "") ;;
  --from) FROM=${2:-}; [ -n "$FROM" ] || usage ;;
  -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
  *) usage ;;
esac
case " ${STEPS[*]} " in *" $FROM "*) ;; *) usage ;; esac

need() {  # interpreter, module list, hint
  "$1" -c "import $2" 2>/dev/null || { echo "$1 cannot import $2; $3 (see README, Environment)" >&2; exit 2; }
}

# Each step returns non-zero on the first failure inside it (errexit does not apply inside `if`).
step_tests()     { need "$SIM_PY" mujoco "export SIM_PY"; "$SIM_PY" tests/test_sim.py; }
step_data()      { need "$SIM_PY" mujoco "export SIM_PY"; bash scripts/gen_all.sh; }
step_features()  { need "$FEAT_PY" "torch, transformers" "export FEAT_PY"; bash scripts/extract_all.sh "${DATASETS[@]}"; }
step_validate()  {
  local ds
  for ds in "${DATASETS[@]}"; do "$PROBE_PY" -m physprobe.io validate "$ds" || return 1; done
}
step_probes()    {   # slide and poscheck first: the push report reads them for its cross-dataset gates
  local ds
  need "$PROBE_PY" "sklearn, pandas, matplotlib" "export FEAT_PY or PROBE_PY"
  for ds in slide poscheck push; do PY="$PROBE_PY" bash scripts/probe_all.sh "$ds" || return 1; done
}
step_quicklook() {
  OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} nice -n 10 "$PROBE_PY" scripts/quicklook_probes.py \
    --datasets "${DATASETS[@]}" --seeds 0 1 2 --out results/quicklook
}

fmt() { printf '%dm%02ds' $(($1 / 60)) $(($1 % 60)); }
echo "== reproduce: root=$PHYSPROBE_ROOT SIM_PY=$SIM_PY FEAT_PY=$FEAT_PY PROBE_PY=$PROBE_PY from=$FROM"
T0=$(date +%s)
SUMMARY=()
started=
for s in "${STEPS[@]}"; do
  [ "$s" = "$FROM" ] && started=1
  if [ -z "$started" ]; then echo "== skip $s"; continue; fi
  echo "== $(date '+%F %T') start $s"
  t=$(date +%s)
  if ! "step_$s"; then
    echo "== FAILED at step '$s' after $(fmt $(( $(date +%s) - t ))); rerun with: $0 --from $s" >&2
    exit 1
  fi
  d=$(( $(date +%s) - t ))
  SUMMARY+=("$(printf '%-10s %s' "$s" "$(fmt "$d")")")
  echo "== $(date '+%F %T') done $s in $(fmt "$d")"
done
echo "== timings"
printf '   %s\n' "${SUMMARY[@]}"
echo "   total      $(fmt $(( $(date +%s) - T0 )))"
echo "== results: results/{push,slide,poscheck}/{table.md,gates.md,figs/}, results/quicklook/"
