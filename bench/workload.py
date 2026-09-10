"""Build a request stream that behaves like real traffic.

Three properties matter, and a naive "send the same prompt 2000 times"
benchmark has none of them:

* **Popularity is skewed.** A handful of questions dominate; most are asked
  once. Real logs follow roughly a Zipf distribution, so the workload does too.
* **Repeats are mostly reworded, not identical.** That is the whole reason a
  semantic cache beats an exact one, so the mix is explicit and reportable.
* **There is a long tail of things never seen before.** Without it, hit rate
  converges to 100% and the number becomes meaningless.

Every knob is printed with the results, because a hit rate is only as honest as
the workload that produced it.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from eval.corpus import GROUPS

# Templates for the long tail: distinct, plausible, and never in the corpus.
_TOOLS = [
    "nginx",
    "Postgres",
    "Redis",
    "Kafka",
    "Airflow",
    "Terraform",
    "Kubernetes",
    "Grafana",
    "Spark",
    "RabbitMQ",
    "Elasticsearch",
    "Vault",
    "Consul",
    "Celery",
    "Prometheus",
    "Traefik",
    "MinIO",
    "ClickHouse",
    "DuckDB",
    "Nomad",
    "Cassandra",
    "Envoy",
    "Flink",
    "Temporal",
    "Pulsar",
    "Etcd",
]
_TASKS = [
    "TLS termination",
    "connection pooling",
    "rate limiting",
    "log rotation",
    "health checks",
    "blue-green deploys",
    "backup scheduling",
    "secret rotation",
    "autoscaling",
    "dead letter queues",
    "schema migrations",
    "read replicas",
    "circuit breaking",
    "request tracing",
    "canary releases",
    "quota enforcement",
    "cold start latency",
    "index rebuilding",
    "leader election",
    "audit logging",
]
_SHAPES = [
    "How do I configure {tool} for {task}?",
    "What is the best practice for {task} in {tool}?",
    "Why is {task} failing in my {tool} setup?",
    "Can you explain {task} in {tool}?",
    "What are the tradeoffs of {task} with {tool}?",
]


@dataclass
class WorkloadConfig:
    requests: int = 2000
    seed: int = 42
    #: Share of traffic that is a brand-new question nobody has asked before.
    unseen_ratio: float = 0.18
    #: Of the repeats, how many are word-for-word rather than reworded.
    exact_repeat_ratio: float = 0.40
    #: Higher means popularity is more concentrated on a few questions.
    zipf_a: float = 1.1


@dataclass
class WorkloadItem:
    prompt: str
    group_id: str | None
    kind: str  # "first_sight" | "exact_repeat" | "paraphrase" | "unseen"


def long_tail_prompts(count: int, rng: random.Random) -> list[str]:
    """Distinct one-off questions for the long tail.

    Exactly one phrasing per (tool, task) pair. An earlier version emitted all
    five shapes for every pair, which quietly made the "never seen before" pool
    full of paraphrases of itself: 19% of it matched another tail prompt above
    the threshold, so the workload's stated ceiling was wrong and the measured
    hit rate was flattered. Keeping one shape per pair is what makes "unseen"
    mean unseen.
    """
    pairs = [(tool, task) for tool in _TOOLS for task in _TASKS]
    rng.shuffle(pairs)
    prompts: list[str] = []
    for index, (tool, task) in enumerate(pairs[:count]):
        shape = _SHAPES[index % len(_SHAPES)]
        prompts.append(shape.format(tool=tool, task=task))
    return prompts


def build(config: WorkloadConfig) -> list[WorkloadItem]:
    rng = random.Random(config.seed)
    groups = list(GROUPS)
    rng.shuffle(groups)

    weights = [1.0 / ((rank + 1) ** config.zipf_a) for rank in range(len(groups))]
    total = sum(weights)
    weights = [w / total for w in weights]

    unseen_needed = int(config.requests * config.unseen_ratio) + 8
    tail = long_tail_prompts(unseen_needed, rng)
    tail_index = 0

    seen: set[str] = set()
    items: list[WorkloadItem] = []

    for _ in range(config.requests):
        if rng.random() < config.unseen_ratio and tail_index < len(tail):
            items.append(WorkloadItem(tail[tail_index], None, "unseen"))
            tail_index += 1
            continue

        group = rng.choices(groups, weights=weights, k=1)[0]
        forms = group.all_forms()
        if group.id not in seen:
            seen.add(group.id)
            items.append(WorkloadItem(group.canonical, group.id, "first_sight"))
            continue

        if rng.random() < config.exact_repeat_ratio:
            items.append(WorkloadItem(group.canonical, group.id, "exact_repeat"))
        else:
            items.append(WorkloadItem(rng.choice(forms), group.id, "paraphrase"))

    return items


def summarise(items: list[WorkloadItem], config: WorkloadConfig) -> dict[str, object]:
    kinds: dict[str, int] = {}
    for item in items:
        kinds[item.kind] = kinds.get(item.kind, 0) + 1
    unique = len({item.prompt for item in items})
    return {
        "config": asdict(config),
        "requests": len(items),
        "unique_prompts": unique,
        "repeat_factor": round(len(items) / unique, 2) if unique else 0,
        "mix": kinds,
        "theoretical_max_hit_rate": round(
            1 - (kinds.get("first_sight", 0) + kinds.get("unseen", 0)) / len(items), 4
        ),
    }
