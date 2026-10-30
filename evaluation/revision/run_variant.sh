#!/bin/bash
# Run one agent in FORECAST mode with the diagnoses of a variant folder (variants.py).
# If the folder has a preamble.txt, it replaces the sentence that introduces the diagnosis,
# and if it has a changed.txt, only those instances run. Results go to
# <variant dir>/<model>/<id>/{prompt,run,evaluate}/; finished instances are skipped, so the
# same command resumes. See evaluation/scripts/run_agent.sh for the models' settings.
# Usage: bash evaluation/revision/run_variant.sh <variant dir> <haiku|qwen> <agent> [workers, default 6]
set -e
V=${1:?usage: run_variant.sh <variant dir> <haiku|qwen> <agent> [workers]}
m=${2:?} a=${3:?} workers=${4:-6}
here=$(cd "$(dirname "$0")" && pwd)
[ -f "$V/$m/$(head -1 "$here/../scripts/instances.txt")/diagnosis/diagnosis.json" ] ||
  { echo "no $m diagnoses in $V; build them with variants.py --model $m" >&2; exit 2; }
# Only the instances whose prompt differs from the evaluated runs (variants.py, changed.txt).
if [ -f "$V/changed.txt" ]; then
  IDS=$V/changed.txt
  export IDS
fi
if [ -f "$V/preamble.txt" ]; then
  FORECAST_DIAGNOSIS_PREAMBLE=$(cat "$V/preamble.txt")
  export FORECAST_DIAGNOSIS_PREAMBLE
fi
exec bash "$here/../scripts/run_agent.sh" "$V" "$m" "$a" forecast "$workers"
