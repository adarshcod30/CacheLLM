"""Cache key derivation.

Two ideas do the heavy lifting here.

**Namespace.** Everything that changes what a *correct* answer looks like goes
into a namespace hash: provider, model, system prompt, temperature bucket,
max_tokens, response format. Two identical user questions asked under different
system prompts must never share an answer, and a model upgrade must not serve
stale text from the old model. Making that a tag on the index means
invalidation is one query, not a scan.

**Two levels.** The exact hash catches literal repeats in about a millisecond
without touching the embedding model. Only genuine near misses pay for a vector
search. In replayed traffic a large share of repeats are literal, so this tier
does real work.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata

from cachellm.models import ChatCompletionRequest

_WS = re.compile(r"\s+")
_TRAILING_PUNCT = re.compile(r"[\s\.\?!,;:]+$")

# Words that carry no meaning for retrieval. Stripping them raises hit rate by
# collapsing polite phrasing onto the same point in vector space. Off by default
# because it also erases some genuine distinctions.
FILLER_WORDS = frozenset(
    {
        "please",
        "could",
        "would",
        "can",
        "you",
        "kindly",
        "hey",
        "hi",
        "hello",
        "thanks",
        "thank",
        "sorry",
        "just",
        "actually",
        "basically",
        "really",
        "simply",
        "quick",
        "quickly",
        "tell",
        "me",
        "explain",
        "about",
        "i",
        "want",
        "to",
        "know",
        "wondering",
        "help",
    }
)


def normalise_text(text: str) -> str:
    """Unicode-normalise, lowercase, collapse whitespace, drop trailing punctuation."""
    out = unicodedata.normalize("NFKC", text or "").strip().lower()
    out = _WS.sub(" ", out)
    return _TRAILING_PUNCT.sub("", out)


_WORD = re.compile(r"[\w']+", re.UNICODE)


def strip_filler(text: str) -> str:
    """Drop filler words, punctuation included, keeping the meaningful tokens."""
    tokens = [t for t in _WORD.findall(normalise_text(text)) if t not in FILLER_WORDS]
    return " ".join(tokens) if tokens else normalise_text(text)


def embedding_text(raw: str, *, strip_fillers: bool) -> str:
    return strip_filler(raw) if strip_fillers else normalise_text(raw)


def temperature_bucket(temperature: float) -> str:
    """Bucket to one decimal place.

    0.0 and 0.05 produce effectively the same answer distribution and should
    share a cache entry; 0.2 and 0.9 should not.
    """
    return f"{round(float(temperature), 1):.1f}"


def namespace_for(request: ChatCompletionRequest, provider: str) -> str:
    """Stable 16-hex-char id for "requests that may share answers"."""
    response_format = request.response_format or {}
    payload = {
        "provider": provider,
        "model": request.model.strip(),
        "system": normalise_text(request.system_text()),
        "temperature": temperature_bucket(request.effective_temperature),
        "top_p": request.top_p,
        "max_tokens": request.effective_max_tokens,
        "response_format": response_format.get("type"),
        "stop": request.stop,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def exact_hash(text: str) -> str:
    return hashlib.sha256(normalise_text(text).encode("utf-8")).hexdigest()[:32]


def entry_id(namespace: str, exact: str) -> str:
    return f"{namespace}-{exact[:20]}"


def prompt_fingerprint(text: str) -> str:
    """Short, non-reversible id for a prompt, safe to put in logs and metrics."""
    return hashlib.blake2b(normalise_text(text).encode("utf-8"), digest_size=6).hexdigest()
