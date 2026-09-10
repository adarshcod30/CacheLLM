# Evaluation

Everything here is reproducible from this repository. No number was copied from
a blog post, and the negative results are kept in on purpose: they are the ones
that changed the design.

```bash
make tune             # threshold sweep on the labelled corpus
make compare-models   # six embedding models, scored on safe recall
make bench            # 2,000-request replay against a running proxy
```

## The corpus

`eval/corpus.py` holds 40 question groups (120 prompts) plus 35 **hard
negatives**. Hard negatives are the whole point. They are pairs that are one
word apart and mean opposite things:

| Question A | Question B |
| --- | --- |
| How do I undo the last git **commit**? | How do I undo the last git **merge**? |
| How do I **enable** logging in Django? | How do I **disable** logging in Django? |
| Why do database indexes make **queries faster**? | Why do database indexes make **writes slower**? |
| Is this **spam**: you have won a free prize | Is this **phishing**: you have won a free prize |

Any cache scores well on paraphrases alone. Reporting a hit rate without
measuring hard negatives is how a semantic cache ends up confidently wrong, so
both numbers are always published together.

That gives 193 labelled pairs: 123 duplicates, 70 non-duplicates (35 hard, 35
random cross-topic).

## Finding 1: no single threshold separates duplicates from hard negatives

Sweeping thresholds with `BAAI/bge-small-en-v1.5`, the model most tutorials
reach for:

| Threshold | Recall | Precision | Hard negatives served |
| --------: | -----: | --------: | --------------------: |
| 0.90 | 48.0% | 0.952 | 3 |
| 0.92 | 35.8% | 0.957 | 2 |
| 0.94 | 19.5% | 0.960 | 1 |
| 0.96 | 8.1% | 1.000 | 0 |
| 0.98 | 0.8% | 1.000 | 0 |

There is no good row in that table. The first threshold that serves zero hard
negatives is 0.96, and it catches 8% of genuine duplicates. The reason is
visible in the raw scores:

**Different questions that score high**

| Similarity | Question A | Question B |
| ---: | --- | --- |
| 0.952 | How do I undo the last git commit? | How do I undo the last git merge? |
| 0.923 | Is this spam: ...free prize | Is this phishing: ...free prize |
| 0.920 | difference between INNER JOIN and LEFT JOIN? | difference between LEFT JOIN and RIGHT JOIN? |

**Same question that scores low**

| Similarity | Question A | Question B |
| ---: | --- | --- |
| 0.550 | What is retrieval augmented generation? | Explain RAG |
| 0.579 | What is CORS? | Explain cross origin resource sharing |
| 0.577 | Explain the global interpreter lock | what does the python GIL do |

An acronym and its expansion score 0.55. One word flipped from commit to merge
scores 0.95. The distributions overlap, so a global threshold cannot split
them. This is the single most important thing measuring produced.

## Finding 2: thresholds are not transferable between models

Six models, each scored at the highest threshold that served **zero** hard
negatives, and the recall available there:

| Model | Dim | Embed ms | Mean duplicate score | Worst hard negative | Separation | Safe threshold | Recall there |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 5.7 | 0.795 | 0.886 | -0.091 | **0.89** | **35.0%** |
| `thenlper/gte-base` | 768 | 20.6 | 0.930 | 0.955 | -0.026 | 0.96 | 26.0% |
| `jinaai/jina-embeddings-v2-small-en` | 512 | 1.9 | 0.908 | 0.959 | -0.051 | 0.96 | 16.3% |
| `BAAI/bge-base-en-v1.5` | 768 | 8.9 | 0.858 | 0.937 | -0.080 | 0.94 | 11.4% |
| `snowflake/snowflake-arctic-embed-s` | 384 | 3.2 | 0.925 | 0.972 | -0.047 | 0.98 | 11.4% |
| `BAAI/bge-small-en-v1.5` | 384 | 3.8 | 0.871 | 0.952 | -0.081 | 0.96 | 8.1% |

Two things fall out.

**The safe threshold moves by 0.09 between models.** A threshold quoted without
naming the model it was measured on is meaningless. CacheLLM ships this table
as `CALIBRATED_THRESHOLDS` and picks the right value automatically, then warns
at startup if you configure a model it has never measured.

**Separation is negative for every model.** Separation is the mean duplicate
score minus the worst hard-negative score. Negative means the distributions
overlap. Not one of these models can cleanly separate a paraphrase from a
one-word-flipped opposite. Bigger and slower did not fix it: `gte-base` is
768-dimensional and four times slower than MiniLM, and still overlaps.

MiniLM won on the metric that matters, gives 4x the safe recall of bge-small,
and downloads 90 MB instead of 130 MB. It is the default.

## Finding 3: a lexical guard does not rescue it

The obvious next idea: after a semantic match, require the two prompts to share
enough content words, on the theory that "commit" and "merge" would be caught.
Measured, it fails:

| Signal | Duplicates (mean) | Hard negatives (mean) | Hard negatives (max) |
| --- | ---: | ---: | ---: |
| Cosine | 0.795 | 0.653 | 0.886 |
| Token Jaccard | 0.450 | **0.422** | 0.714 |

Hard negatives have *higher* token overlap than genuine paraphrases, because
they are identical except for one decisive word. Meanwhile real paraphrases
("What is CORS?" and "Explain cross origin resource sharing") share almost no
tokens at all. A minimum-overlap guard rejects the good matches and keeps the
dangerous ones. It was measured and dropped rather than shipped.

What would actually work is a cheap verification step: send a semantic-tier
candidate and the incoming prompt to a small model and ask whether they are the
same question. That is on the roadmap, not in the code, and it is not claimed
anywhere in this README.

## Finding 4: my own benchmark was flattering itself

The first load test reported a 79.1% hit rate. It also showed 37 of 368
supposedly never-before-seen questions hitting the cache, which should be
impossible if they are genuinely new.

They were not new. The long-tail generator emitted five phrasings of every
(tool, task) pair, so "How do I configure Redis for rate limiting?" and "What
is the best practice for rate limiting in Redis?" were both in the "unseen"
pool. 19% of that pool matched another member above threshold.

Fixed by emitting exactly one phrasing per pair. Cross-matching within the tail
went to 0%, the stated ceiling became true, and the headline hit rate dropped
from 79.1% to 77.2%. The lower number is the honest one.

The lesson generalises: when a benchmark surprises you in your favour, the bug
is usually in the benchmark.

## Load test

2,000 requests, concurrency 8, against a warm proxy on an Apple M4.

Workload: 483 unique prompts, 4.1x repeat factor, deliberately skewed so a few
questions dominate, with an 18% long tail of genuinely new questions.

| Metric | Value |
| --- | ---: |
| Hit rate | **77.2%** |
| Workload ceiling | 79.6% |
| Exact-tier hits | 1,347 |
| Semantic-tier hits | 198 |
| Errors | 0 |
| Throughput | 55.6 req/s |

| Request kind | Count | Hit rate | |
| --- | ---: | ---: | --- |
| Exact repeat | 584 | 99.7% | as designed |
| Reworded repeat | 1,008 | 95.5% | the semantic tier earning its place |
| First sight | 40 | 0.0% | correct, nothing to hit |
| Genuinely new | 368 | **0.0%** | **zero false positives** |

| Latency (ms) | p50 | p95 | p99 |
| --- | ---: | ---: | ---: |
| Cache hit | 4.1 | 8.3 | 15.3 |
| Cache miss | 613.2 | 630.7 | 644.3 |

p95 speedup: **75.6x**. Modelled cost reduction: **73.9%**. Hit rate warmed
from 49% in the first 100 requests to 87% in the last 100.

**Read the latency honestly.** Cache-miss latency here is a 600 ms simulated
upstream, because AWS credentials were not available on the machine that
produced this run. The hit-side numbers are real measurements of real work:
embedding, vector search, Redis round trip, serialisation. The miss side is a
stand-in, and the speedup ratio inherits that. To produce the number for your
own deployment, point it at a real provider:

```bash
CACHELLM_DEFAULT_PROVIDER=bedrock make serve
uv run python -m bench.replay --model bedrock/us.amazon.nova-micro-v1:0 --reset
```

## What this means for a deployment

1. **The exact tier does the heavy lifting.** 87% of hits came from exact
   matching, which is free, sub-millisecond and cannot be semantically wrong.
   Any semantic cache without an exact tier in front of it is leaving the
   safest wins on the table.
2. **The semantic tier is a bonus, not the foundation.** It added 198 hits,
   about 10 points of hit rate, at a real risk that has to be measured.
3. **Start in shadow mode.** Log what the cache would have served for a week,
   read the near-miss log, then switch it on.
4. **Re-run the sweep on your own traffic.** These numbers describe this
   corpus. Yours will differ, and `/admin/threshold-sweep` takes your own
   labelled pairs.
