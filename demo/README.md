# Demo

Two small scripts and a recording tape, used to produce the GIF in the README.

```bash
# terminal 1: a proxy with a cold cache (no Redis needed; it uses memory)
CACHELLM_DEFAULT_PROVIDER=bedrock AWS_REGION=us-east-1 uv run cachellm serve
curl -X POST localhost:8080/admin/invalidate -H 'content-type: application/json' -d '{"all":true}'
curl -X POST localhost:8080/admin/reset-stats

# terminal 2
./demo/ask.sh "What is Redis used for?"
./demo/stats.sh
```

`ask.sh` sends one question and prints what the cache did with it: hit or miss,
which tier answered, the similarity score, and the modelled money saved. It
reads the `X-Cache-*` response headers, so it is also a compact reference for
what those headers contain.

Both scripts honour two environment variables:

| Variable | Default |
| --- | --- |
| `CACHELLM_URL` | `http://127.0.0.1:8080` |
| `CACHELLM_MODEL` | `bedrock/us.amazon.nova-micro-v1:0` |

To run against the built-in fake provider instead, so it costs nothing and needs
no cloud account:

```bash
CACHELLM_DEFAULT_PROVIDER=fake CACHELLM_FAKE_LATENCY_MS=600 uv run cachellm serve
CACHELLM_MODEL=fake/echo ./demo/ask.sh "What is Redis used for?"
```

## Re-recording the GIF

```bash
brew install vhs
vhs demo/demo.tape
```

The tape assumes a cold cache, so flush it first or the opening request will
already be a hit and the recording will not show the contrast.
