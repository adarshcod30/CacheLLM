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
    "moonshot.",
    "moonshotai.",
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
#:
#: A model id starting with one of these always routes to that adapter. What
#: reaches the upstream is the adapter's call: `openai/` is removed only when the
#: upstream is OpenAI's own API, because Groq, OpenRouter and Together name
#: OpenAI's models `openai/gpt-oss-120b` and would reject the bare id.
ROUTING_PREFIXES: tuple[str, ...] = ("bedrock", "openai", "fake")


# ----------------------------------------------------------------- quick guide


@dataclass(frozen=True)
class Host:
    """One place models are served from."""

    key: str  # short id, e.g. "groq"
    name: str  # what a person calls it
    provider: str  # which adapter handles it
    base_url: str | None  # what to put in CACHELLM_OPENAI_BASE_URL
    env_key: str | None  # where its API key conventionally lives
    extra: str | None = None  # pip extra required, if any
    examples: tuple[str, ...] = field(default_factory=tuple)
    note: str = ""

    @property
    def needs_key(self) -> bool:
        """True when this host expects a vendor API key from the environment."""
        return self.env_key is not None


#: Every host CacheLLM can sit in front of. Base URLs were checked live; model
#: ids are examples current at the time of writing and are not used for routing.
#: Anything not listed still works, because an unrecognised name falls through
#: to whichever OpenAI-compatible endpoint the operator configured.
HOSTS: tuple[Host, ...] = (
    # ------------------------------------------------ needs no vendor API key
    Host(
        "bedrock",
        "AWS Bedrock",
        "bedrock",
        None,
        None,
        "aws",
        (
            "bedrock/us.amazon.nova-micro-v1:0",
            "bedrock/us.amazon.nova-lite-v1:0",
            "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
            "bedrock/meta.llama3-70b-instruct-v1:0",
        ),
        "Uses your existing AWS credentials, so there is no vendor key at all. "
        "Bedrock ids are recognised by their vendor prefix, so the `bedrock/` "
        "prefix is optional.",
    ),
    Host(
        "ollama",
        "Ollama, on this machine",
        "openai",
        "http://localhost:11434/v1",
        None,
        None,
        ("llama3.2", "qwen2.5-coder:7b", "mistral", "phi4", "gemma3"),
        "Runs locally and costs nothing. Tags use `name:tag`. Any API key value "
        "works because Ollama ignores it.",
    ),
    Host(
        "local",
        "vLLM, LM Studio, llama.cpp",
        "openai",
        "http://localhost:8000/v1",
        None,
        None,
        ("whatever you served",),
        "Any OpenAI-compatible server you run yourself.",
    ),
    # ------------------------------------------------------- key in the usual env var
    Host(
        "openai",
        "OpenAI",
        "openai",
        "https://api.openai.com/v1",
        "OPENAI_API_KEY",
        None,
        ("gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "o3-mini", "text-embedding-3-small"),
        "Recognised by family prefix, so these route here whatever the default is.",
    ),
    Host(
        "anthropic",
        "Anthropic, Claude",
        "openai",
        "https://api.anthropic.com/v1",
        "ANTHROPIC_API_KEY",
        None,
        ("claude-sonnet-4-5", "claude-haiku-4-5", "claude-3-5-haiku-latest"),
        "Anthropic ships an OpenAI-compatible endpoint. Claude is also reachable "
        "through Bedrock with no vendor key.",
    ),
    Host(
        "gemini",
        "Google Gemini",
        "openai",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY",
        None,
        ("gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3.5-flash"),
        "Also reads GOOGLE_API_KEY. Generous free tier.",
    ),
    Host(
        "xai",
        "xAI, Grok",
        "openai",
        "https://api.x.ai/v1",
        "XAI_API_KEY",
        None,
        ("grok-4", "grok-3-mini", "grok-2-1212"),
        "",
    ),
    Host(
        "groq",
        "Groq",
        "openai",
        "https://api.groq.com/openai/v1",
        "GROQ_API_KEY",
        None,
        ("openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.6-27b"),
        "Very fast, useful free tier.",
    ),
    Host(
        "deepseek",
        "DeepSeek",
        "openai",
        "https://api.deepseek.com/v1",
        "DEEPSEEK_API_KEY",
        None,
        ("deepseek-chat", "deepseek-reasoner"),
        "",
    ),
    Host(
        "mistral",
        "Mistral AI",
        "openai",
        "https://api.mistral.ai/v1",
        "MISTRAL_API_KEY",
        None,
        ("mistral-large-latest", "mistral-small-latest", "codestral-latest"),
        "",
    ),
    Host(
        "openrouter",
        "OpenRouter",
        "openai",
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        None,
        (
            "anthropic/claude-3.5-sonnet",
            "meta-llama/llama-3.1-8b-instruct",
            "google/gemini-2.0-flash-exp",
        ),
        "One key reaches hundreds of models. Ids contain a slash and are forwarded whole.",
    ),
    Host(
        "together",
        "Together",
        "openai",
        "https://api.together.xyz/v1",
        "TOGETHER_API_KEY",
        None,
        ("meta-llama/Llama-3.3-70B-Instruct-Turbo", "mistralai/Mixtral-8x7B-Instruct-v0.1"),
        "Ids contain a slash and are forwarded whole.",
    ),
    Host(
        "fireworks",
        "Fireworks",
        "openai",
        "https://api.fireworks.ai/inference/v1",
        "FIREWORKS_API_KEY",
        None,
        ("accounts/fireworks/models/llama-v3p3-70b-instruct",),
        "",
    ),
    Host(
        "cerebras",
        "Cerebras",
        "openai",
        "https://api.cerebras.ai/v1",
        "CEREBRAS_API_KEY",
        None,
        ("llama3.1-8b", "llama-3.3-70b"),
        "",
    ),
    Host(
        "perplexity",
        "Perplexity",
        "openai",
        "https://api.perplexity.ai",
        "PERPLEXITY_API_KEY",
        None,
        ("sonar", "sonar-pro"),
        "Answers are search-grounded, so cache them with a short TTL.",
    ),
    Host(
        "moonshot",
        "Moonshot, Kimi",
        "openai",
        "https://api.moonshot.ai/v1",
        "MOONSHOT_API_KEY",
        None,
        ("kimi-k2-0905-preview", "moonshot-v1-8k"),
        "",
    ),
    # ---------------------------------------------------------------- built in
    Host(
        "fake",
        "Built-in test double",
        "fake",
        None,
        None,
        None,
        ("fake/echo",),
        "Deterministic, free, offline. Used by the tests and the reproducible "
        "benchmark, and by the quick start so you can see it work before "
        "spending anything.",
    ),
)

#: Extra environment variables worth checking for the same host.
ALSO_READS: dict[str, tuple[str, ...]] = {
    "gemini": ("GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"),
    "openai": ("OPENAI_KEY",),
}

HOSTS_BY_KEY: dict[str, Host] = {h.key: h for h in HOSTS}


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
