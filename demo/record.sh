#!/usr/bin/env bash
# Drives the demo shown in the README GIF: types each command out, runs it,
# pauses so a viewer can read the result.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DIM=$'\033[2m'; CYAN=$'\033[36m'; RESET=$'\033[0m'

type_out () {                       # simulate a person typing
  local text="$1"
  printf '%s$%s ' "$CYAN" "$RESET"
  for (( i=0; i<${#text}; i++ )); do
    printf '%s' "${text:$i:1}"
    sleep 0.028
  done
  printf '\n'
}

comment () { printf '%s# %s%s\n' "$DIM" "$1" "$RESET"; sleep 0.9; }
run ()     { type_out "$1"; eval "$1"; }

clear
sleep 0.6
comment "CacheLLM: a semantic cache for LLM APIs. Adopting it is one line."
run 'export OPENAI_BASE_URL=http://localhost:8080/v1'
sleep 1.1

comment "Ask something new. This one goes to the real model on AWS Bedrock."
run './demo/ask.sh "What is Redis used for?"'
sleep 1.6

comment "Ask the exact same thing again."
run './demo/ask.sh "What is Redis used for?"'
sleep 1.6

comment "Now reword it. Different words, same question."
run './demo/ask.sh "What is Redis typically used for?"'
sleep 1.8

comment "A genuinely different question still misses. No false positives."
run './demo/ask.sh "What is Kafka used for?"'
sleep 1.6

comment "Anything personal is never stored, however similar it looks."
run './demo/ask.sh "What is my order 987654321012 status?"'
sleep 1.8

comment "What did that buy us?"
run './demo/stats.sh'
sleep 4
