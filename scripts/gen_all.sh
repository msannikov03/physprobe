#!/usr/bin/env bash
# A2 smoke + A4 full generation, then validate each dataset. Resumable: finished clips are skipped.
#   mkdir -p logs && nohup bash scripts/gen_all.sh > logs/gen_all.log 2>&1 &
# Needs SIM_PY (requirements-sim.txt; set it before running, env.sh defaults to python3) and ffmpeg for grid.mp4.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$here/env.sh"
cd "$PHYSPROBE_ROOT" || exit 1
mkdir -p logs
"$SIM_PY" -c "import mujoco" 2>/dev/null || { echo "SIM_PY=$SIM_PY cannot import mujoco; export SIM_PY (see env.sh)" >&2; exit 2; }
W=${WORKERS:-10}
gen() { nice -n 10 "$SIM_PY" -m physprobe.sim.generate --workers "$W" "$@" || exit 1; }

gen --scenario push --n 20 --dataset smoke_push
gen --scenario push --n 2000 --dataset push
gen --scenario slide --n 1000 --dataset slide
gen --scenario poscheck --n 300 --dataset poscheck

rc=0
for ds in smoke_push push slide poscheck; do
  nice -n 10 "$SIM_PY" -m physprobe.io validate "$ds" || rc=1
done
for ds in push slide poscheck; do
  echo "== stats $ds"; nice -n 10 "$SIM_PY" -m physprobe.sim.stats "$ds"
done
df -h "$PHYSPROBE_ROOT"
exit $rc
