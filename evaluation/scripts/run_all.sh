#!/bin/bash
# Run all 20 combinations of model, mode and agent of the repair evaluation (2 models x
# 2 modes x 5 agents), one after another, with the worker counts of the paper's runs.
# See run_agent.sh for the models' settings. Finished instances are skipped, so the same command resumes.
# Usage: bash evaluation/scripts/run_all.sh <runs dir> [haiku|qwen ...]
set -e
R=${1:?usage: run_all.sh <runs dir> [haiku|qwen ...]}
shift
models=("$@")
[ ${#models[@]} -gt 0 ] || models=(haiku qwen)
here=$(cd "$(dirname "$0")" && pwd)
for m in "${models[@]}"; do
  for mode in baseline forecast; do
    for a in mini-swe-agent swe-agent acr repair-agent agentless; do
      # Agentless spends most of its time running tests, not waiting for the model.
      if [ "$a" = agentless ]; then w=30; elif [ "$m" = qwen ]; then w=6; else w=10; fi
      echo "== $m $a $mode ($w workers)"
      bash "$here/run_agent.sh" "$R" "$m" "$a" "$mode" "$w"
    done
  done
done
