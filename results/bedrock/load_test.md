### Load test

2000 requests, concurrency 8, model `bedrock/us.amazon.nova-micro-v1:0`, 47.73s wall (41.9 req/s).

| Metric | Value |
| --- | ---: |
| Hit rate (all requests) | 77.0% |
| Hit rate (cacheable only) | 77.0% |
| Workload ceiling | 79.6% |
| Exact-tier hits | 1265 |
| Semantic-tier hits | 275 |
| Semantic share of hits | 17.9% |
| Errors | 0 |

| Latency (ms) | p50 | p95 | p99 |
| --- | ---: | ---: | ---: |
| Cache hit | 2.62 | 5.72 | 10.17 |
| Cache miss | 797.02 | 1022.83 | 1351.76 |
| p95 speedup | 178.8x | | |

| Modelled cost | USD |
| --- | ---: |
| Spent upstream | 0.003796 |
| Saved by cache | 0.013492 |
| Reduction | 78.0% |