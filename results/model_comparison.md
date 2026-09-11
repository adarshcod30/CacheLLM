| Model | Dim | Embed ms p50 | Embed ms p95 | Mean dup score | Worst hard negative | Separation | Safe threshold | Recall there |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 5.46 | 5.98 | 0.795 | 0.886 | -0.091 | 0.89 | 35.0% |
| `thenlper/gte-base` | 768 | 20.72 | 23.68 | 0.930 | 0.955 | -0.026 | 0.96 | 26.0% |
| `jinaai/jina-embeddings-v2-small-en` | 512 | 1.7 | 1.97 | 0.908 | 0.959 | -0.051 | 0.96 | 16.3% |
| `BAAI/bge-base-en-v1.5` | 768 | 7.5 | 10.03 | 0.858 | 0.937 | -0.080 | 0.94 | 11.4% |
| `snowflake/snowflake-arctic-embed-s` | 384 | 2.69 | 3.15 | 0.925 | 0.972 | -0.047 | 0.98 | 11.4% |
| `BAAI/bge-small-en-v1.5` | 384 | 3.0 | 3.99 | 0.871 | 0.952 | -0.081 | 0.96 | 8.1% |