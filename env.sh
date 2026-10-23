# shellcheck shell=bash disable=SC2296  # ${(%):-%x} is the zsh fallback for BASH_SOURCE
# source this before any physprobe command (bash or zsh). PHYSPROBE_ROOT, SIM_PY, FEAT_PY and MUJOCO_GL can be
# preset to override them; the caches always live under $PHYSPROBE_ROOT/.cache (keeps them off a full home partition).
# PHYSPROBE_ROOT defaults to the clone that holds this file (CONTRACT v1: unset -> clone root).
_pp_here="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")" && pwd)"
: "${PHYSPROBE_ROOT:=$_pp_here}"
export PHYSPROBE_ROOT
export HF_HOME=$PHYSPROBE_ROOT/.cache/hf
export TORCH_HOME=$PHYSPROBE_ROOT/.cache/torch
export PIP_CACHE_DIR=$PHYSPROBE_ROOT/.cache/pip
export XDG_CACHE_HOME=$PHYSPROBE_ROOT/.cache
export MPLCONFIGDIR=$PHYSPROBE_ROOT/.cache/mpl
export TMPDIR=$PHYSPROBE_ROOT/.cache/tmp
if [ "$(uname -s)" = Linux ]; then export MUJOCO_GL=${MUJOCO_GL:-egl}; fi   # headless EGL; macOS uses its default
# Interpreters: SIM_PY needs requirements-sim.txt (mujoco, no torch); FEAT_PY needs requirements-feats.txt
# (torch, transformers, sklearn, no mujoco). Default: python3 on PATH. On our GPU machine they are:
#   export SIM_PY=/path/to/venv-sim/bin/python   # mujoco 3.16, no torch
#   export FEAT_PY=/path/to/venv-feats/bin/python    # torch 2.10, transformers 5.3, sklearn 1.8, no mujoco
# with PHYSPROBE_ROOT=/path/to/physprobe (the clone itself).
export SIM_PY=${SIM_PY:-python3}
export FEAT_PY=${FEAT_PY:-python3}
export PYTHONPATH="$_pp_here${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$MPLCONFIGDIR" "$TMPDIR"
unset _pp_here
