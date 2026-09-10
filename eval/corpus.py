"""A hand-built evaluation corpus.

Two halves, and the second one is the point.

**Paraphrase groups** are questions a user could ask in several ways that all
deserve the same answer. They set the ceiling on hit rate.

**Hard negatives** are pairs that look almost identical to an embedding model
but must never share an answer: one word changed, opposite intent, different
entity, different time frame. Any cache can score well on paraphrases alone.
Reporting a hit rate without measuring these is how a semantic cache ends up
confidently wrong in production, so both numbers are always published together.

The corpus is deliberately small enough to read and argue with. For scale,
``eval/datasets.py`` pulls the 150k-pair Quora Question Pairs set.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuestionGroup:
    id: str
    category: str
    canonical: str
    paraphrases: tuple[str, ...]

    def all_forms(self) -> tuple[str, ...]:
        return (self.canonical, *self.paraphrases)


GROUPS: tuple[QuestionGroup, ...] = (
    # ---------------------------------------------------------- programming
    QuestionGroup(
        "py-what",
        "factual",
        "What is Python?",
        ("Explain Python to me", "Can you tell me what Python is", "what's python"),
    ),
    QuestionGroup(
        "py-list-reverse",
        "factual",
        "How do I reverse a list in Python?",
        (
            "What is the way to reverse a Python list?",
            "reverse a list python",
            "How can I flip the order of a list in Python?",
        ),
    ),
    QuestionGroup(
        "py-dict-merge",
        "factual",
        "How do I merge two dictionaries in Python?",
        ("combine two dicts in python", "What is the way to join two Python dicts?"),
    ),
    QuestionGroup(
        "py-venv",
        "factual",
        "How do I create a virtual environment in Python?",
        ("python virtual environment setup", "How to make a venv in Python?"),
    ),
    QuestionGroup(
        "py-gil",
        "factual",
        "What is the GIL in Python?",
        ("Explain the global interpreter lock", "what does the python GIL do"),
    ),
    QuestionGroup(
        "git-undo",
        "factual",
        "How do I undo the last git commit?",
        ("git undo last commit", "How can I revert my most recent commit in git?"),
    ),
    QuestionGroup(
        "git-branch",
        "factual",
        "How do I create a new branch in git?",
        ("git make a new branch", "What is the command to create a git branch?"),
    ),
    QuestionGroup(
        "docker-vs-vm",
        "factual",
        "What is the difference between Docker and a VM?",
        ("How do containers differ from virtual machines?", "docker vs virtual machine difference"),
    ),
    QuestionGroup(
        "sql-join",
        "factual",
        "What is the difference between INNER JOIN and LEFT JOIN?",
        ("Explain inner join versus left join in SQL", "sql inner join vs left join"),
    ),
    QuestionGroup(
        "rest-vs-graphql",
        "factual",
        "What is the difference between REST and GraphQL?",
        ("Compare REST APIs and GraphQL", "rest vs graphql"),
    ),
    # ------------------------------------------------------------- concepts
    QuestionGroup(
        "redis-what",
        "factual",
        "What is Redis used for?",
        ("Explain what Redis does", "what is redis", "Why would I use Redis?"),
    ),
    QuestionGroup(
        "vector-db",
        "factual",
        "What is a vector database?",
        ("Explain vector databases", "what does a vector db do"),
    ),
    QuestionGroup(
        "embedding-what",
        "factual",
        "What is a text embedding?",
        ("Explain what embeddings are", "what are text embeddings"),
    ),
    QuestionGroup(
        "rag-what",
        "factual",
        "What is retrieval augmented generation?",
        ("Explain RAG", "what does RAG mean in AI"),
    ),
    QuestionGroup(
        "llm-token",
        "factual",
        "What is a token in a language model?",
        ("Explain LLM tokens", "what counts as a token for an LLM"),
    ),
    QuestionGroup(
        "cosine-sim",
        "factual",
        "What is cosine similarity?",
        ("Explain cosine similarity", "how does cosine similarity work"),
    ),
    QuestionGroup(
        "http-idempotent",
        "factual",
        "What does idempotent mean in HTTP?",
        ("Explain HTTP idempotency", "which http methods are idempotent"),
    ),
    QuestionGroup(
        "acid",
        "factual",
        "What does ACID mean in databases?",
        ("Explain ACID properties", "what are acid guarantees in a database"),
    ),
    QuestionGroup(
        "cap-theorem",
        "factual",
        "What is the CAP theorem?",
        ("Explain the CAP theorem", "what does cap theorem say"),
    ),
    QuestionGroup(
        "big-o",
        "factual",
        "What is Big O notation?",
        ("Explain big o notation", "what does big o measure"),
    ),
    # ------------------------------------------------------------- general
    QuestionGroup(
        "capital-france",
        "factual",
        "What is the capital of France?",
        ("Which city is the capital of France?", "france capital city"),
    ),
    QuestionGroup(
        "capital-japan",
        "factual",
        "What is the capital of Japan?",
        ("Which city is the capital of Japan?", "japan capital city"),
    ),
    QuestionGroup(
        "photosynthesis",
        "factual",
        "What is photosynthesis?",
        ("Explain photosynthesis", "how does photosynthesis work"),
    ),
    QuestionGroup(
        "gravity",
        "factual",
        "What causes gravity?",
        ("Explain what gravity is", "why does gravity exist"),
    ),
    QuestionGroup(
        "water-boil",
        "factual",
        "At what temperature does water boil?",
        ("What is the boiling point of water?", "water boiling temperature"),
    ),
    QuestionGroup(
        "tallest-mountain",
        "factual",
        "What is the tallest mountain in the world?",
        ("Which mountain is the highest on earth?", "world's tallest mountain"),
    ),
    QuestionGroup(
        "speed-light",
        "factual",
        "What is the speed of light?",
        ("How fast does light travel?", "speed of light value"),
    ),
    QuestionGroup(
        "dna-what",
        "factual",
        "What is DNA?",
        ("Explain what DNA is", "what does dna stand for and do"),
    ),
    # ------------------------------------------------------ classification
    QuestionGroup(
        "sent-pos",
        "classification",
        "Classify the sentiment: I absolutely loved this product",
        (
            "What is the sentiment of: I absolutely loved this product",
            "sentiment of 'I absolutely loved this product'",
        ),
    ),
    QuestionGroup(
        "sent-neg",
        "classification",
        "Classify the sentiment: the delivery was late and the box was damaged",
        ("What is the sentiment of: the delivery was late and the box was damaged",),
    ),
    QuestionGroup(
        "spam-check",
        "classification",
        "Is this spam: congratulations you have won a free prize",
        ("Classify as spam or not: congratulations you have won a free prize",),
    ),
    QuestionGroup(
        "lang-detect",
        "classification",
        "What language is this: bonjour tout le monde",
        ("Detect the language of: bonjour tout le monde",),
    ),
    # ---------------------------------------------------------- how-to / ops
    QuestionGroup(
        "k8s-pod-logs",
        "factual",
        "How do I see logs for a Kubernetes pod?",
        ("kubectl get pod logs", "What is the command to view pod logs?"),
    ),
    QuestionGroup(
        "linux-port",
        "factual",
        "How do I find which process is using a port on Linux?",
        ("what is using port 8080 linux", "How to check the process on a given port?"),
    ),
    QuestionGroup(
        "ssh-key",
        "factual",
        "How do I generate an SSH key?",
        ("create ssh key pair", "What command makes a new SSH key?"),
    ),
    QuestionGroup(
        "npm-vs-yarn",
        "factual",
        "What is the difference between npm and yarn?",
        ("Compare npm and yarn", "npm vs yarn"),
    ),
    QuestionGroup(
        "http-status-429",
        "factual",
        "What does HTTP 429 mean?",
        ("Explain status code 429", "what is a 429 error"),
    ),
    QuestionGroup(
        "cors-what",
        "factual",
        "What is CORS?",
        ("Explain cross origin resource sharing", "what does cors do"),
    ),
    QuestionGroup(
        "jwt-what",
        "factual",
        "What is a JWT?",
        ("Explain JSON web tokens", "what is a json web token"),
    ),
    QuestionGroup(
        "index-db",
        "factual",
        "Why do database indexes make queries faster?",
        ("Explain how a database index speeds up a query", "what does a db index actually do"),
    ),
)

# Pairs that are lexically close and semantically different. A cache that hits
# on any of these is broken, however good its headline hit rate looks.
HARD_NEGATIVES: tuple[tuple[str, str], ...] = (
    ("What is the capital of France?", "What is the capital of Finland?"),
    ("What is the capital of Japan?", "What is the capital of Jordan?"),
    ("How do I reverse a list in Python?", "How do I reverse a string in Python?"),
    ("How do I merge two dictionaries in Python?", "How do I merge two lists in Python?"),
    ("How do I enable logging in Django?", "How do I disable logging in Django?"),
    ("How do I start a Docker container?", "How do I stop a Docker container?"),
    ("How do I undo the last git commit?", "How do I undo the last git merge?"),
    ("How do I create a new branch in git?", "How do I delete a branch in git?"),
    (
        "What is the difference between INNER JOIN and LEFT JOIN?",
        "What is the difference between LEFT JOIN and RIGHT JOIN?",
    ),
    ("What is the tallest mountain in the world?", "What is the longest river in the world?"),
    ("At what temperature does water boil?", "At what temperature does water freeze?"),
    ("What is the speed of light?", "What is the speed of sound?"),
    ("What is a vector database?", "What is a graph database?"),
    ("What is a text embedding?", "What is a text encoding?"),
    ("What is Redis used for?", "What is Kafka used for?"),
    ("What is the GIL in Python?", "What is the GC in Python?"),
    (
        "Classify the sentiment: I absolutely loved this product",
        "Classify the sentiment: I absolutely hated this product",
    ),
    (
        "Is this spam: congratulations you have won a free prize",
        "Is this phishing: congratulations you have won a free prize",
    ),
    ("What does HTTP 429 mean?", "What does HTTP 422 mean?"),
    ("What is the difference between npm and yarn?", "What is the difference between npm and npx?"),
    ("How do I generate an SSH key?", "How do I revoke an SSH key?"),
    ("What is retrieval augmented generation?", "What is retrieval augmented fine tuning?"),
    ("Why do database indexes make queries faster?", "Why do database indexes make writes slower?"),
    (
        "What is the difference between Docker and a VM?",
        "What is the difference between Docker and Podman?",
    ),
    ("How do I see logs for a Kubernetes pod?", "How do I see events for a Kubernetes pod?"),
    ("What is photosynthesis?", "What is photorespiration?"),
    ("What is CORS?", "What is CSRF?"),
    ("What is a JWT?", "What is a JWK?"),
    ("What causes gravity?", "What causes magnetism?"),
    ("What is Big O notation?", "What is Big Omega notation?"),
    ("What is cosine similarity?", "What is cosine distance?"),
    ("What language is this: bonjour tout le monde", "What language is this: buenos dias a todos"),
    (
        "How do I find which process is using a port on Linux?",
        "How do I find which process is using a file on Linux?",
    ),
    ("What does ACID mean in databases?", "What does BASE mean in databases?"),
    ("What is the CAP theorem?", "What is the PACELC theorem?"),
)


def paraphrase_pairs() -> list[tuple[str, str, bool]]:
    """Every within-group pair, labelled as a duplicate."""
    pairs: list[tuple[str, str, bool]] = []
    for group in GROUPS:
        forms = group.all_forms()
        for i in range(len(forms)):
            for j in range(i + 1, len(forms)):
                pairs.append((forms[i], forms[j], True))
    return pairs


def negative_pairs(include_random: bool = True, seed: int = 7) -> list[tuple[str, str, bool]]:
    """Hard negatives, plus easy cross-topic negatives for a realistic baseline."""
    pairs: list[tuple[str, str, bool]] = [(a, b, False) for a, b in HARD_NEGATIVES]
    if include_random:
        import random

        rng = random.Random(seed)
        canonicals = [g.canonical for g in GROUPS]
        for _ in range(len(HARD_NEGATIVES)):
            a, b = rng.sample(canonicals, 2)
            pairs.append((a, b, False))
    return pairs


def labelled_pairs(include_random_negatives: bool = True) -> list[dict[str, object]]:
    """The full labelled set consumed by the threshold sweep.

    ``kind`` matters as much as ``duplicate``: a false positive against a hard
    negative ("France" vs "Finland") is a real product failure, while one
    against a random cross-topic pair mostly says the embedding model is
    broken. Averaging them together hides the number you care about.
    """
    rows: list[dict[str, object]] = [
        {"a": a, "b": b, "duplicate": True, "kind": "paraphrase"} for a, b, _ in paraphrase_pairs()
    ]
    rows += [
        {"a": a, "b": b, "duplicate": False, "kind": "hard_negative"} for a, b in HARD_NEGATIVES
    ]
    if include_random_negatives:
        hard = {(a, b) for a, b in HARD_NEGATIVES}
        for a, b, _ in negative_pairs(True):
            if (a, b) not in hard:
                rows.append({"a": a, "b": b, "duplicate": False, "kind": "random_negative"})
    return rows


def all_prompts() -> list[str]:
    prompts: list[str] = []
    for group in GROUPS:
        prompts.extend(group.all_forms())
    return prompts


def stats() -> dict[str, int]:
    pairs = labelled_pairs()
    return {
        "groups": len(GROUPS),
        "prompts": len(all_prompts()),
        "duplicate_pairs": sum(1 for p in pairs if p["duplicate"]),
        "non_duplicate_pairs": sum(1 for p in pairs if not p["duplicate"]),
        "hard_negatives": len(HARD_NEGATIVES),
    }
