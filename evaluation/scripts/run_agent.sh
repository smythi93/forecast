#!/bin/bash
# Run one agent of the repair evaluation in one mode with one model over all 300
# instances. Each instance runs in the instance's SWE-bench image and is judged by the
# SWE-bench harness; the results go to <runs dir>/<model>/<id>/{prompt,run,evaluate}/.
# Finished instances are skipped, so the same command resumes an interrupted run.
#
# haiku  Claude Haiku 4.5 through a LiteLLM proxy: set LITELLM_PROXY_API_BASE and
#        LITELLM_PROXY_API_KEY; the proxy must serve anthropic/claude-haiku-4-5-20251001.
# qwen   Qwen3.5 27B through Ollama: OLLAMA_ENDPOINTS is a comma-separated list of Ollama
#        servers serving qwen3.5:27b (default http://127.0.0.1:11434); workers are spread
#        over them. Costs are computed at Haiku's list prices.
#
# MIRRORS (optional): a folder made by mirrors.sh; git then clones from there instead of
# GitHub, which Agentless needs with many workers.
# IDS (optional): a file with the instance ids to run instead of all 300.
# Usage: bash evaluation/scripts/run_agent.sh <runs dir> <haiku|qwen> <agent> <baseline|forecast> [workers, default 10]
set -e
R=${1:?usage: run_agent.sh <runs dir> <haiku|qwen> <agent> <baseline|forecast> [workers]}
m=${2:?} a=${3:?} mode=${4:?} workers=${5:-10}
here=$(cd "$(dirname "$0")" && pwd)
cd "$here/../.."
mkdir -p "$R"
R=$(cd "$R" && pwd)

# Agents run in the published SWE-bench images (Docker Hub, swebench/sweb.eval.*).
export FORECAST_IMAGE_NAMESPACE=swebench
if [ -n "$MIRRORS" ]; then
  export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0="url.$(cd "$MIRRORS" && pwd)/.insteadOf" \
         GIT_CONFIG_VALUE_0=https://github.com/
fi
# Agentless commits in scratch repositories to compute diffs; give git an identity.
export GIT_AUTHOR_NAME=agentless GIT_AUTHOR_EMAIL=agentless@localhost \
       GIT_COMMITTER_NAME=agentless GIT_COMMITTER_EMAIL=agentless@localhost

case $m in
  haiku)
    export LITELLM_PROXY_CANONICAL_MODEL=claude-haiku-4-5-20251001
    exec python -m evaluation.parallel_repair --agent "$a" --mode "$mode" \
      --model proxy/anthropic/claude-haiku-4-5-20251001 \
      --instance-ids-file "${IDS:-$here/instances.txt}" --output-dir "$R/haiku" --workers "$workers" ;;
  qwen)
    export FORECAST_PRICE_AS=claude-haiku-4-5-20251001
    exec python -m evaluation.parallel_repair --agent "$a" --mode "$mode" \
      --model ollama/qwen3.5:27b --ollama-endpoints "${OLLAMA_ENDPOINTS:-http://127.0.0.1:11434}" \
      --instance-ids-file "${IDS:-$here/instances.txt}" --output-dir "$R/qwen" --workers "$workers" ;;
  *)
    echo "unknown model $m (haiku or qwen)" >&2; exit 2 ;;
esac
