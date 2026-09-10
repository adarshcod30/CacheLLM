"""Decides whether a request may be cached, in which category, for how long.

This module is the honesty layer. A semantic cache that caches everything will
eventually serve yesterday's stock price or one user's order status to another
user. Each rule below exists because of a specific way that goes wrong.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from cachellm.models import ChatCompletionRequest
from cachellm.settings import Settings

# Anything that pins an answer to *now*. These still get cached, but with a
# short TTL, because "what is the weather in Jaipur" repeats a lot within an hour.
VOLATILE_PATTERNS = re.compile(
    r"\b(today|tonight|right now|currently|current|latest|breaking|this (week|month|year|morning)|"
    r"yesterday|tomorrow|as of|so far|up to date|live|news|stock price|share price|weather|"
    r"forecast|who won|score|trending|recent|nowadays|these days)\b",
    re.IGNORECASE,
)

# Open-ended generation: two different answers are both correct, so a cache hit
# that returns the same poem twice is a product bug, not a saving.
CREATIVE_PATTERNS = re.compile(
    r"\b(write|compose|draft|generate|create|invent|imagine|brainstorm|come up with)\b.{0,40}"
    r"\b(poem|story|song|lyrics|joke|essay|article|blog|caption|tagline|slogan|name|names|ideas?|"
    r"script|email|letter|post)\b",
    re.IGNORECASE,
)

# Constrained answer space, so near-duplicates are safe at a lower threshold.
CLASSIFICATION_PATTERNS = re.compile(
    r"\b(classify|categorise|categorize|label|sentiment|is this|does this|yes or no|true or false|"
    r"which category|tag this|detect|extract|rate this|score this)\b",
    re.IGNORECASE,
)

# Personal data. If any of this is present the answer is about one person and
# must never be served to another, so we do not store it at all.
PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")),
    ("long_digits", re.compile(r"\b\d{9,}\b")),  # phone, Aadhaar, account, order id
    ("card", re.compile(r"\b(?:\d[ -]*?){13,16}\b")),
    ("api_key", re.compile(r"\b(sk|pk|ghp|gho|AKIA|ASIA)[-_A-Za-z0-9]{12,}\b")),
    (
        "possessive",
        re.compile(
            r"\bmy (order|account|booking|policy|invoice|ticket|card|salary|"
            r"password|address|phone|otp|pan|aadhaar|ssn)\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class PolicyDecision:
    cacheable: bool
    category: str
    threshold: float
    ttl: int
    reason: str = ""

    @property
    def bypass_reason(self) -> str:
        return "" if self.cacheable else self.reason


def _classify(text: str, request: ChatCompletionRequest) -> str:
    if VOLATILE_PATTERNS.search(text):
        return "volatile"
    if CREATIVE_PATTERNS.search(text):
        return "creative"
    max_tokens = request.effective_max_tokens
    if CLASSIFICATION_PATTERNS.search(text) or (max_tokens is not None and max_tokens <= 32):
        return "classification"
    if request.user_turn_count() > 1:
        return "conversational"
    return "factual"


def detect_pii(text: str) -> str | None:
    for label, pattern in PII_PATTERNS:
        if pattern.search(text):
            return label
    return None


def decide(
    request: ChatCompletionRequest,
    settings: Settings,
    *,
    cache_control: str = "",
) -> PolicyDecision:
    """Return the caching decision for one request."""
    text = request.last_user_text()
    category = _classify(text, request)
    threshold = settings.threshold_for(category)
    ttl = settings.ttl_for(category)

    def no(reason: str) -> PolicyDecision:
        return PolicyDecision(False, category, threshold, ttl, reason)

    control = (cache_control or "").lower()
    if "no-store" in control or "no-cache" in control:
        return no("client_requested_bypass")
    if not settings.enabled:
        return no("cache_disabled")
    if not text:
        return no("no_user_message")
    if len(text) > settings.max_prompt_chars:
        return no("prompt_too_long")
    if request.effective_temperature > settings.max_cacheable_temperature:
        return no("temperature_too_high")
    if (request.n or 1) > 1:
        return no("multiple_completions_requested")
    if request.tools and not settings.cache_tool_calls:
        return no("tool_calls")
    if any(m.has_non_text_parts() for m in request.messages):
        return no("non_text_content")
    fmt = (request.response_format or {}).get("type")
    if fmt in ("json_object", "json_schema") and not settings.cache_json_mode:
        return no("json_mode")
    if request.user_turn_count() > 1 and not settings.cache_multi_turn:
        return no("multi_turn_conversation")
    if settings.pii_guard:
        found = detect_pii(text)
        if found:
            return no(f"pii:{found}")

    return PolicyDecision(True, category, threshold, ttl, "")
