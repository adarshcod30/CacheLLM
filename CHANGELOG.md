# Changelog

## 0.2.2

Found by running the proxy against real Google Gemini and Groq for the first
time, instead of against simulated servers. Both now pass every live check, and
`bench/live_check.py` lets anyone repeat that with their own key.

### Fixed

- **Groq's main models work.** Groq names its two main chat models
  `openai/gpt-oss-20b` and `openai/gpt-oss-120b`. The proxy read the `openai/`
  as its own routing hint and removed it, so every request came back 404. The
  prefix is now removed only when the upstream is OpenAI itself. OpenRouter and
  Together ids such as `openai/gpt-4o` were broken the same way.
- **A failed stream reports the failure.** A streaming request for an unknown
  model returned HTTP 200 with an empty answer. The proxy now waits for the
  upstream's first event, so the caller gets the host's real status code, and a
  failure partway through ends the stream with the error event the OpenAI SDKs
  raise.
- **A `.env` file is respected.** Auto-detection only checked exported
  variables, so a base URL written in `.env` was replaced by whichever vendor
  key happened to be exported. `.env.example` now leaves the provider lines
  commented out, so copying it keeps detection on.
- **`CACHELLM_EMBEDDING_CACHE_SIZE=0` no longer turns caching off.** The
  embedder read its results back out of its own cache, so a cache of 0, or a
  batch bigger than the cache, failed. The admin threshold sweep could reach
  the second case with a large enough list.
- **Pricing picks the longest matching model name,** so a Flash-Lite preview
  is no longer priced as the dearer Flash.

### Clearer

- **A Redis that answers but cannot hold the cache is named.** A server without
  the search module, or a URL pointing at a database other than 0, now gets a
  startup warning and a note at the top of `cachellm stats`, with the fix.
  Before, it was one info-level line quoting a cryptic `FT.INFO` error. No
  Redis at all stays quiet, because that is the normal default.
- **The README explains how to get a working Redis:** Homebrew or the official
  Docker image, database 0, and what happens when either is missing.
- **Latency is reported apart for exact and reworded hits.** In the Bedrock
  run, reworded hits took 9.7 ms at p95 and exact ones 4.9 ms. The single
  5.7 ms figure blended the two. The model comparison's embed column timed
  short synthetic strings and is now labelled that way, and the benchmark now
  times real questions.
- **Gemini and Groq models have real prices,** checked against their pricing
  pages on 2026-09-11, so savings on those hosts no longer use a stand-in.
- **Catalog examples come from each host's live model list.** All three Groq
  examples and two of the three Gemini examples had been retired.
- **A download-size claim is corrected.** As fastembed fetches them, MiniLM is
  86 MB and bge-small 63 MB, so MiniLM is the larger of the two.

## 0.2.1

Fixes `cachellm stats` and `cachellm watch`, which were the point of 0.2.0 and
did not work with its own default backend.

Both commands built their own application state, which meant starting a second
process with a second, empty in-memory cache and reporting on that. Against a
running proxy they always said "no requests yet". They now read the running
proxy over HTTP, through a new `GET /admin/requests` endpoint, and say plainly
what to do when nothing is answering.

This only ever worked by accident with the redis backend, where the state
happens to be shared. With the default in-memory backend the cache lives inside
the serving process, so reading it has to go through the server.

```bash
cachellm stats                       # reads http://127.0.0.1:8080 by default
cachellm stats --url http://host:80  # or wherever it runs
```

## 0.2.0

The release that made it installable. Three things stood between `pip install`
and something useful, and all three are gone.

### It runs with nothing installed

The cache now lives in a numpy matrix inside the proxy by default, so Redis is
optional. That was not a compromise for speed: finding the nearest of 20,000
cached prompts by multiplying one matrix takes 0.85 ms, while a Redis round trip
alone costs 2 to 3 ms and the embedding step before every lookup costs about
9 ms. An approximate index only starts paying above roughly 100,000 entries.

```bash
pip install cachellm-proxy
cachellm serve
```

`CACHELLM_BACKEND` picks storage: `memory` needs no server, `redis` shares one
cache across workers and survives restarts, and `auto` (the default) uses Redis
when it can reach one. Asking for `redis` explicitly never falls back, because
a silent downgrade would hide an outage. `CACHELLM_MEMORY_SNAPSHOT_PATH`
persists the in-process cache across restarts.

### It configures itself

On startup, when nothing is set, CacheLLM looks for a provider this machine
already has: an API key in the conventional environment variable, Ollama
listening locally, or resolvable AWS credentials. It logs what it picked, why,
and the variable that overrides it. Setting any of `CACHELLM_DEFAULT_PROVIDER`,
`CACHELLM_OPENAI_BASE_URL` or `CACHELLM_OPENAI_API_KEY` turns detection off.

Seventeen hosts are catalogued with the environment variable each one's key
lives in, its base URL, any pip extra, and real example model ids. Every base
URL was checked against the live endpoint. `cachellm providers` prints the lot
alongside what this machine can already reach.

### The dashboard is a terminal command

```bash
cachellm stats      # hit rate, savings, latency, recent requests
cachellm watch      # follow requests live
```

Hit rate with a bar, money saved against money spent, cached and uncached
latency with the speedup, and a log of recent requests showing which hit, which
missed, which were bypassed and why, with the similarity score and the money
each one saved. Plain ANSI, no new dependency, colour dropped when piped.
Prometheus and Grafana remain for history and alerting; nothing requires them.

### You install only what you route to

A plain install went from 69 packages and 214 MB to 51 and 167 MB. AWS, Redis
and the `openai` SDK are extras now; the SDK was never imported by the source at
all, only by one test.

```
pip install cachellm-proxy              OpenAI-compatible endpoints
pip install "cachellm-proxy[aws]"       Bedrock
pip install "cachellm-proxy[redis]"     Redis instead of in-process
pip install "cachellm-proxy[all]"       everything
```

Asking for something whose extra is missing names the exact command to fix it.

### Fixes

- **Model ids containing a slash were mangled.** Any such id had its first
  segment stripped before being forwarded, which broke every OpenRouter and
  Together model: `meta-llama/Llama-3.3-70B-Instruct-Turbo` reached the upstream
  as `Llama-3.3-70B-Instruct-Turbo`. Only `bedrock/`, `openai/` and `fake/` are
  stripped now.
- **The default provider was Bedrock**, which a plain install cannot reach now
  that AWS is an extra. Someone running Ollama was told to install AWS. The
  fallback is the OpenAI-compatible adapter, which is always present and is
  where names like `llama3.2` actually live.
- **`/metrics` was unscrapable.** It advertised the OpenMetrics content type
  while emitting the Prometheus text format, so a live Prometheus rejected every
  scrape with "data does not end with # EOF".
- **Refused cache writes were silent.** A response stopping at `max_tokens` is
  correctly not cached, but with a low cap that drove the hit rate to 8.9% with
  nothing explaining it. Refusals are counted by reason and `/admin/stats`
  reports them in plain English. `CACHELLM_CACHE_TRUNCATED` opts in.
- **Bedrock vendor prefixes needed their trailing dot.** `moonshot` matched
  Moonshot's own `moonshot-v1-8k`. The dot is what separates
  `moonshot.kimi-k2-thinking` on Bedrock from a vendor's own hyphenated ids.
- **Bypassed requests logged a blank prompt**, which are the rows you most want
  to read.
- **Credentials from `aws login`** need `botocore[crt]`, now included in the
  `aws` extra, with an error that names the fix.
- **The Docker image would not build**, because `pyproject` declares a readme
  the Dockerfile never copied.

### Known limitation

`openai/`, `bedrock/` and `fake/` are reserved routing prefixes, so Groq's
`openai/gpt-oss-120b` is read as a routing instruction and reaches the upstream
as `gpt-oss-120b`.

### Under the hood

261 tests, up from 128. Storage tests run against both backends to prove they
behave identically. CI gained a job with no Redis service at all, which asserts
the cache still works, the semantic tier still fires, and no genuinely new
question is ever answered from cache.

## 0.1.0

First release. A drop-in semantic caching proxy for OpenAI-compatible APIs, with
a two-tier cache, per-model threshold calibration, cacheability policy, TTL
tiers, stampede protection and full observability.

Measured against Amazon Nova Micro on AWS Bedrock over 2,000 requests: 77.0% hit
rate against a 79.6% workload ceiling, zero false positives across 368 genuinely
new questions, 5.7 ms p95 on hits against 1,022.8 ms on misses, 78.0% cost
reduction.
