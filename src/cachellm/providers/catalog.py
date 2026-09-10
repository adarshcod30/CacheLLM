"""What model names mean, and where they are served.

Routing has only three destinations: AWS Bedrock, anything that speaks the
OpenAI protocol, and the test double. Groq, Together, OpenRouter, Ollama, vLLM
and OpenAI itself all share one adapter, so the router never has to tell them
apart. It only has to answer one question: **is this a Bedrock model id, an
OpenAI model id, or something served by whatever OpenAI-compatible endpoint the
operator configured?**

Two kinds of knowledge live here.

`BEDROCK_VENDORS` and `OPENAI_FAMILIES` are *routing rules*. They match on the
naming conventions each platform uses, not on individual model names, so they
keep working as new models ship. Bedrock ids are always `vendor.model`, and
OpenAI's are always a small set of family prefixes.

`HOSTS` is a *quick reference* for humans: the base URL each hosting provider
uses and a few real model ids you can paste. Model lineups change constantly,
so treat those ids as examples rather than a live catalogue. Nothing in the
router depends on them; they exist for `GET /v1/models` and the docs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------- routing

#: Bedrock model ids are `vendor.model`, optionally with a cross-region prefix
#: (`us.`, `eu.`, `apac.`). Matching the vendor segment is stable across new
#: model releases in a way that listing individual ids never is.
BEDROCK_VENDORS: tuple[str, ...] = (
    "ai21.",
    "amazon.",
    "anthropic.",
    "cohere.",
    "deepseek.",
    "google.",
    "luma.",
    "meta.",
    "minimax.",
    "mistral.",
    "moonshot",
    "nvidia.",
    "openai.",
    "qwen.",
    "stability.",
    "writer.",
    "zai.",
)

#: Cross-region inference profiles put a region in front of the vendor.
BEDROCK_REGION_PREFIXES: tuple[str, ...] = ("us.", "eu.", "apac.", "ap.")

#: OpenAI's own families. Anything here is OpenAI no matter what the configured
#: default provider is, because no other vendor ships a model called `gpt-4o`.
OPENAI_FAMILIES: tuple[str, ...] = (
    "gpt-",
    "gpt3",
    "gpt4",
    "o1",
    "o3",
    "o4",
    "chatgpt-",
    "text-embedding-",
    "text-moderation-",
    "davinci",
    "babbage",
    "whisper-",
    "dall-e",
    "tts-",
)

#: Prefixes this proxy owns. Only these are stripped before forwarding, so a
#: model id that legitimately contains a slash survives intact. OpenRouter and
#: Together both use `vendor/model` ids, and stripping their vendor segment
#: makes the upstream reject the request as an unknown model.
ROUTING_PREFIXES: tuple[str, ...] = ("bedrock", "openai", "fake")


# ----------------------------------------------------------------- quick guide


@dataclass(frozen=True)
class Host:
    """One place models are served from."""

    name: str
    provider: str  # which adapter handles it
    base_url: str | None  # what to put in CACHELLM_OPENAI_BASE_URL
    extra: str | None  # pip extra required, if any
    examples: tuple[str, ...] = field(default_factory=tuple)
    note: str = ""


#: A human quick reference. Model ids are examples, current as of writing, and
#: are not used for routing. Anything not listed still works: it falls through
#: to the configured OpenAI-compatible endpoint, which is the right answer for
#: every host in this table except Bedrock.
HOSTS: tuple[Host, ...] = (
    Host(
        "AWS Bedrock",
        "bedrock",
        None,
        "aws",
        (
            "bedrock/us.amazon.nova-micro-v1:0",
            "bedrock/us.amazon.nova-lite-v1:0",
            "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
            "bedrock/meta.llama3-70b-instruct-v1:0",
            "bedrock/mistral.mixtral-8x7b-instruct-v0:1",
        ),
        "Recognised by the vendor prefix, so the `bedrock/` prefix is optional. "
        "Uses your AWS credentials; no vendor API key needed.",
    ),
    Host(
        "OpenAI",
        "openai",
        "https://api.openai.com/v1",
        None,
        ("gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "o3-mini", "text-embedding-3-small"),
        "Recognised by family prefix, so these route to OpenAI whatever the default is.",
    ),
    Host(
        "Groq",
        "openai",
        "https://api.groq.com/openai/v1",
        None,
        ("llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"),
        "OpenAI-compatible. Very fast; useful free tier.",
    ),
    Host(
        "Google Gemini",
        "openai",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        None,
        ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"),
        "Google ships an OpenAI-compatible endpoint, so no separate adapter is needed.",
    ),
    Host(
        "Anthropic",
        "openai",
        "https://api.anthropic.com/v1",
        None,
        ("claude-sonnet-4-5", "claude-haiku-4-5", "claude-3-5-haiku-latest"),
        "Anthropic ships an OpenAI-compatible endpoint. Claude is also reachable "
        "through Bedrock without a separate key.",
    ),
    Host(
        "Ollama, local",
        "openai",
        "http://localhost:11434/v1",
        None,
        ("llama3.2", "qwen2.5-coder:7b", "mistral", "phi4", "gemma3"),
        "Ollama tags use `name:tag`. Any API key value works; it is ignored.",
    ),
    Host(
        "OpenRouter",
        "openai",
        "https://openrouter.ai/api/v1",
        None,
        (
            "anthropic/claude-3.5-sonnet",
            "meta-llama/llama-3.1-8b-instruct",
            "google/gemini-2.0-flash-exp",
        ),
        "Model ids contain a slash. The proxy forwards them whole; only its own "
        "`bedrock/`, `openai/` and `fake/` prefixes are stripped.",
    ),
    Host(
        "Together",
        "openai",
        "https://api.together.xyz/v1",
        None,
        ("meta-llama/Llama-3.3-70B-Instruct-Turbo", "mistralai/Mixtral-8x7B-Instruct-v0.1"),
        "Also uses slash-containing ids, forwarded whole.",
    ),
    Host(
        "vLLM or LM Studio, self-hosted",
        "openai",
        "http://localhost:8000/v1",
        None,
        ("whatever you served", "my-finetuned-model"),
        "Any OpenAI-compatible server. Unknown names fall through to here, which "
        "is why the default provider is the OpenAI-compatible adapter.",
    ),
    Host(
        "Built-in test double",
        "fake",
        None,
        None,
        ("fake/echo",),
        "Deterministic, free, offline. Used by the tests and the reproducible benchmark.",
    ),
)


def looks_like_bedrock(model: str) -> bool:
    """True when the id follows Bedrock's `vendor.model` convention."""
    candidate = model.split("/", 1)[1] if model.startswith("bedrock/") else model
    for region in BEDROCK_REGION_PREFIXES:
        if candidate.startswith(region):
            candidate = candidate[len(region) :]
            break
    return candidate.startswith(BEDROCK_VENDORS)


def looks_like_openai(model: str) -> bool:
    """True when the id is one of OpenAI's own model families."""
    return model.lower().startswith(OPENAI_FAMILIES)


def strip_routing_prefix(model: str) -> str:
    """Remove only a prefix this proxy owns, leaving vendor ids intact.

    `openai/gpt-4o-mini` becomes `gpt-4o-mini`, but
    `meta-llama/Llama-3.3-70B-Instruct-Turbo` is forwarded unchanged, because
    that slash is part of the model's real name on Together and OpenRouter.
    """
    head, sep, tail = model.partition("/")
    if sep and head.lower() in ROUTING_PREFIXES:
        return tail
    return model


def suggested_models() -> list[str]:
    """Flat list for `GET /v1/models`."""
    seen: list[str] = []
    for host in HOSTS:
        for example in host.examples:
            if example not in seen:
                seen.append(example)
    return seen
