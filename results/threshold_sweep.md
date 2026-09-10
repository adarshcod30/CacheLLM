Embedding model: `BAAI/bge-small-en-v1.5`  
Pairs: 193 (123 duplicates, 70 non-duplicates)

| Threshold | Recall | Precision | F1 | False hits | Hard-negative hits |
| --------: | -----: | --------: | -: | ---------: | -----------------: |
| 0.70 | 93.5% | 0.799 | 0.861 | 29 | 29 (82.9%) |
| 0.72 | 92.7% | 0.808 | 0.864 | 27 | 27 (77.1%) |
| 0.74 | 90.2% | 0.816 | 0.857 | 25 | 25 (71.4%) |
| 0.76 | 87.8% | 0.824 | 0.850 | 23 | 23 (65.7%) |
| 0.78 | 87.0% | 0.843 | 0.856 | 20 | 20 (57.1%) |
| 0.80 | 86.2% | 0.841 | 0.851 | 20 | 20 (57.1%) |
| 0.82 | 81.3% | 0.862 | 0.837 | 16 | 16 (45.7%) |
| 0.84 | 75.6% | 0.869 | 0.809 | 14 | 14 (40.0%) |
| 0.86 | 68.3% | 0.903 | 0.778 | 9 | 9 (25.7%) |
| 0.88 | 56.9% | 0.897 | 0.697 | 8 | 8 (22.9%) |
| 0.90 | 48.0% | 0.952 | 0.638 | 3 | 3 (8.6%) |
| 0.92 | 35.8% | 0.957 | 0.521 | 2 | 2 (5.7%) |
| 0.94 | 19.5% | 0.960 | 0.324 | 1 | 1 (2.9%) |
| 0.96 | 8.1% | 1.000 | 0.150 | 0 | 0 (0.0%) |
| 0.98 | 0.8% | 1.000 | 0.016 | 0 | 0 (0.0%) |
| 1.00 | 0.0% | 1.000 | 0.000 | 0 | 0 (0.0%) |

**Closest calls: different questions that look alike**

| Similarity | Question A | Question B |
| ---: | --- | --- |
| 0.952 | How do I undo the last git commit? | How do I undo the last git merge? |
| 0.923 | Is this spam: congratulations you have won a free prize | Is this phishing: congratulations you have won a free prize |
| 0.920 | What is the difference between INNER JOIN and LEFT JOIN? | What is the difference between LEFT JOIN and RIGHT JOIN? |
| 0.896 | How do I see logs for a Kubernetes pod? | How do I see events for a Kubernetes pod? |
| 0.894 | How do I enable logging in Django? | How do I disable logging in Django? |
| 0.893 | How do I find which process is using a port on Linux? | How do I find which process is using a file on Linux? |

**Hardest duplicates: same question, lowest score**

| Similarity | Question A | Question B |
| ---: | --- | --- |
| 0.550 | What is retrieval augmented generation? | Explain RAG |
| 0.572 | What is retrieval augmented generation? | what does RAG mean in AI |
| 0.577 | Explain the global interpreter lock | what does the python GIL do |
| 0.579 | What is CORS? | Explain cross origin resource sharing |
| 0.579 | What is the GIL in Python? | Explain the global interpreter lock |
| 0.609 | Explain cross origin resource sharing | what does cors do |