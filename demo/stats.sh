#!/usr/bin/env bash
# One-line summary of what the cache has done so far.
set -uo pipefail
BASE="${CACHELLM_URL:-http://127.0.0.1:8080}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
curl -s "$BASE/admin/stats" | python3 "$HERE/_render.py" stats
