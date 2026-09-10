#!/usr/bin/env bash
# Ask the proxy a question and show what the cache did with it.
#   demo/ask.sh "What is Redis used for?"
set -uo pipefail

PROMPT="${1:?usage: demo/ask.sh \"your question\"}"
BASE="${CACHELLM_URL:-http://127.0.0.1:8080}"
MODEL="${CACHELLM_MODEL:-bedrock/us.amazon.nova-micro-v1:0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HEADERS=$(mktemp); trap 'rm -f "$HEADERS"' EXIT

BODY=$(PROMPT="$PROMPT" MODEL="$MODEL" python3 -c '
import json, os
print(json.dumps({
    "model": os.environ["MODEL"],
    "temperature": 0,
    "max_tokens": 512,
    "messages": [
        {"role": "system", "content": "Answer concisely in at most three sentences."},
        {"role": "user", "content": os.environ["PROMPT"]},
    ],
}))')

ELAPSED=$(curl -s -o /dev/null -D "$HEADERS" -w '%{time_total}' \
  "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$BODY")

python3 "$HERE/_render.py" ask "$HEADERS" "$ELAPSED"
