#!/bin/bash
# Repeat the evaluation of one agent with one model: both modes, all 300 instances, in a
# new folder that gets the evaluated diagnoses (variants.py full). The first evaluation
# stays untouched. Finished instances are skipped, so the same command resumes.
# MODES (optional): the modes to run, default "baseline forecast"; one mode alone repeats
# only that mode, and the other can follow later into the same folder.
# Usage: bash evaluation/revision/repeat.sh <runs dir> <repetition dir> <haiku|qwen> <agent> [workers, default 6]
set -e
R=${1:?usage: repeat.sh <runs dir> <repetition dir> <haiku|qwen> <agent> [workers]}
P=${2:?} m=${3:?} a=${4:?} workers=${5:-6}
here=$(cd "$(dirname "$0")" && pwd)
[ -f "$P/variant.json" ] || python "$here/variants.py" full "$R" "$P" --model "$m"
for mode in ${MODES:-baseline forecast}; do
  echo "== $m $a $mode"
  bash "$here/../scripts/run_agent.sh" "$P" "$m" "$a" "$mode" "$workers"
done
