# Contributing to CacheLLM

Thanks for taking the time. Bug reports, fixes, new providers, better docs and
measurements that prove the cache wrong are all welcome.

## Set up

You need Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/adarshcod30/CacheLLM.git
cd CacheLLM
uv sync --all-extras
```

Run the proxy from your checkout, with auto-reload:

```bash
make dev
```

`make help` lists every other target.

## Test

```bash
make test       # the full suite
make test-cov   # with a coverage report
make lint       # ruff and mypy
make fix        # let ruff fix what it can
```

The suite needs no network, no account and no model download. It uses a
deterministic hashing embedder and simulated hosts. Tests that need Redis run
against `redis://localhost:6379/0`, which must be a Redis 8 with the search
module, and are skipped when no Redis answers there.

Checks against real providers live in `bench/live_check.py` and
`bench/live_routing.py`. They need keys and cost a fraction of a cent, so they
are not part of CI. Run one if your change touches a provider.

## Make a change

1. Open an issue first for anything large, so we can agree on the shape.
2. Create a branch named for the change, such as `fix-stream-errors`.
3. Write a test that fails without your change. Most bugs in this project were
   found by a real client or a real provider, not by a unit test, so a test
   that reproduces what actually went wrong is worth more than one that checks
   the happy path.
4. Keep the change focused, and run `make lint && make test`.
5. Update the README and `CHANGELOG.md` if behaviour or settings change.

**If you touch matching quality**, such as thresholds, normalisation, the
embedding model or the policy, include a threshold sweep or a model comparison
in the pull request. `make tune` and `make compare-models` produce them. The
numbers matter more than the argument, and a change that serves even one wrong
answer on the hard negatives will not be merged.

## Add a provider

Most hosts speak the OpenAI protocol, so a new one is usually a single entry in
`src/cachellm/providers/catalog.py`: its key, name, base URL, environment
variable and a few example models that you have seen work. Add a naming rule in
`NAME_CONVENTIONS` only if its model names are unique to it. Then run
`bench/live_check.py` against it and paste the result into the pull request.

## Commit messages

Say what changed and why, in plain words. The subject line should make sense on
its own, for example "Keep Groq's openai/ model ids whole".

## Be kind

Assume good intent, be patient with newcomers, and argue with evidence rather
than volume. Harassment of any kind is not welcome here.

## Releasing

Maintainers release by pushing a version tag. See
[docs/publishing.md](docs/publishing.md).
