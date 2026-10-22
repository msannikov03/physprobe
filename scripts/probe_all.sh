#!/usr/bin/env bash
# Baselines -> probes -> report for one dataset. Canonical and fresh: probes.run --fresh deletes
# results/<dataset>/probes.csv and preds/ first, so no row survives from an older commit.
# push*/slide*: also builds contact/noise_{000,010,030,100} (--noise-curve) and probes the
# video + contact/realistic combinations (block-scaled, CONTRACT v1.2).
# Usage: scripts/probe_all.sh <dataset> [extra args for probes.run]
# Python: $PY, else $FEAT_PY (compute), else ./.venv/bin/python, else python3.
# Shared box: BLAS/OpenMP capped at 8 threads (override via OMP_NUM_THREADS), every step at nice 10.
set -euo pipefail
D=${1:?usage: probe_all.sh <dataset> [probes.run args]}
shift || true
cd "$(dirname "$0")/.."
PY=${PY:-${FEAT_PY:-}}
if [ -z "$PY" ]; then
  if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
fi
export PYTHONPATH=${PYTHONPATH:-$PWD}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$OMP_NUM_THREADS}
RUN="nice -n 10 $PY"

TARGETS="log_m mu"
BUILD_ARGS=()
INPUTS=()
case "$D" in
  poscheck*) TARGETS="x0_m" ;;
  push*|slide*)
    BUILD_ARGS=(--noise-curve)
    INPUTS=(--inputs "vjepa2_l/clip_seq+contact/realistic" "dinov2_b/clip_seq+contact/realistic"
            "vjepa2_l/clip_mean_L16+contact/realistic") ;;
esac

echo "== baselines.build $D ${BUILD_ARGS[*]-}"
$RUN -m physprobe.baselines.build --dataset "$D" ${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"}
echo "== probes.run $D ($TARGETS) --fresh"
$RUN -m physprobe.probes.run --dataset "$D" --fresh --targets $TARGETS --threads "$OMP_NUM_THREADS" \
  ${INPUTS[@]+"${INPUTS[@]}"} "$@"
echo "== probes.report $D"
case "$D" in
  push|slide|poscheck) $RUN -m physprobe.probes.report --all ;;   # re-renders push last
  *) $RUN -m physprobe.probes.report --dataset "$D" ;;
esac
