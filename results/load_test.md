### Load test

2000 requests, concurrency 8, model `fake/echo`, 35.98s wall (55.6 req/s).

| Metric | Value |
| --- | ---: |
| Hit rate (all requests) | 77.2% |
| Hit rate (cacheable only) | 77.2% |
| Workload ceiling | 79.6% |
| Exact-tier hits | 1347 |
| Semantic-tier hits | 198 |
| Semantic share of hits | 12.8% |
| Errors | 0 |

| Latency (ms) | p50 | p95 | p99 |
| --- | ---: | ---: | ---: |
| Cache hit | 4.1 | 8.34 | 15.31 |
| Cache miss | 613.23 | 630.72 | 644.27 |
| p95 speedup | 75.6x | | |

| Modelled cost | USD |
| --- | ---: |
| Spent upstream | 0.006417 |
| Saved by cache | 0.018198 |
| Reduction | 73.9% |