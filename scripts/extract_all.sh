#!/usr/bin/env bash
# Extract all feature reps and modes for each dataset, in order dinov2_b, dinov2_b_random, vjepa2_l, vjepa2_l_random,
# cosmos_cv8. Runs under nice 10 with CPU threads capped (OMP/MKL/OPENBLAS_NUM_THREADS, default 8).
# Usage: scripts/extract_all.sh <dataset>...
#   nohup scripts/extract_all.sh push slide poscheck > /dev/null 2>&1 &
# Env overrides: REPS="..." BATCH=4 THREADS=8 EXTRA_ARGS="--limit 20" (e.g. "--dtype fp32"; default dtype is
# bf16 for pretrained reps and fp32 for *_random). Needs FEAT_PY (requirements-feats.txt; env.sh defaults to python3).
# Logs: $PHYSPROBE_ROOT/logs/extract_<ds>_<rep>.log, summary in $PHYSPROBE_ROOT/logs/extract_all.log. Resumable: rerun the same command after a crash.
set -uo pipefail
if [ $# -lt 1 ]; then echo "usage: $0 <dataset>..." >&2; exit 2; fi
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$here/env.sh"
cd "$PHYSPROBE_ROOT" || exit 1
mkdir -p logs
"$FEAT_PY" -c "import torch, transformers" 2>/dev/null || { echo "FEAT_PY=$FEAT_PY cannot import torch/transformers; export FEAT_PY (see env.sh)" >&2; exit 2; }
REPS=${REPS:-"dinov2_b dinov2_b_random vjepa2_l vjepa2_l_random cosmos_cv8"}
THREADS=${THREADS:-8}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
BATCH=${BATCH:-4}
EXTRA_ARGS=${EXTRA_ARGS:-}
summary=logs/extract_all.log
fails=0
say() { echo "$(date '+%F %T') $*" | tee -a "$summary"; }
say "start datasets=[$*] reps=[$REPS] batch=$BATCH threads=$THREADS extra=[$EXTRA_ARGS] host=$(hostname)"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/gpu: /' | tee -a "$summary"
for ds in "$@"; do
  if [ ! -f "data/$ds/meta.jsonl" ]; then say "SKIP $ds: data/$ds/meta.jsonl missing"; fails=$((fails + 1)); continue; fi
  for rep in $REPS; do
    log="logs/extract_${ds}_${rep}.log"
    say "run $ds $rep -> $log"
    t0=$(date +%s)
    # shellcheck disable=SC2086
    if nice -n 10 "$FEAT_PY" -m physprobe.feats.extract --dataset "$ds" --rep "$rep" --batch "$BATCH" $EXTRA_ARGS >> "$log" 2>&1; then
      say "ok   $ds $rep in $(( $(date +%s) - t0 )) s"
    else
      say "FAIL $ds $rep (exit $?), see $log"; fails=$((fails + 1))
    fi
  done
done
say "end fails=$fails"
exit $(( fails > 0 ))
