#!/bin/bash
# Compute FORECAST's diagnosis for every instance of SWE-bench Lite and put it where the
# repair runs expect it.
#
# For each instance, evaluation.diagnosis_batch (stage all) collects execution traces of the
# FAIL_TO_PASS and PASS_TO_PASS tests in the instance's SWE-bench image, builds the traced
# relevance tree, derives the cause-chain diagnosis, and ranks spectrum-based fault
# localization (used by ACR and RepairAgent). This happens in <work dir>, which keeps the
# traces and trees and needs a lot of disk. The diagnosis files and each instance's
# report.json (with the time of every step) are then copied to <runs dir>/<model>/<id>/,
# the same for both models.
#
# Building a tree takes much memory for the large instances (sympy); keep [jobs] low.
# Finished stages are skipped, so the same command resumes.
# Usage: bash evaluation/scripts/diagnose.sh <work dir> <runs dir> [jobs, default 4]
set -e
W=${1:?usage: diagnose.sh <work dir> <runs dir> [jobs]}
R=${2:?usage: diagnose.sh <work dir> <runs dir> [jobs]}
J=${3:-4}
here=$(cd "$(dirname "$0")" && pwd)
cd "$here/../.."
mkdir -p "$W"
W=$(cd "$W" && pwd)

xargs -P "$J" -I{} sh -c 'python -m evaluation.diagnosis_batch --subset lite --stage all \
    --instance-ids "$1" --output-dir "$2" > "$2/$1.log" 2>&1 || echo "$1: failed, see $2/$1.log"' \
    _ {} "$W" < "$here/instances.txt"

for m in haiku qwen; do
  for i in $(cat "$here/instances.txt"); do
    mkdir -p "$R/$m/$i/diagnosis"
    cp "$W/$i/diagnosis/"{diagnosis.json,diagnosis.txt,diagnosis.md,sbfl.json} "$R/$m/$i/diagnosis/"
    cp "$W/$i/report.json" "$R/$m/$i/"
  done
done
echo "diagnoses in $R/haiku and $R/qwen"
