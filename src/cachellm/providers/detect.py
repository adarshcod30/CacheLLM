"""Work out what this machine can already talk to.

The point is that `cachellm serve` should do something useful on a machine you
have not configured. Most people already have a key in their environment, or
Ollama running, or AWS credentials, and asking them to restate that in a config
file is friction for no gain.

Detection never guesses silently. Whatever it picks is logged with the reason
and the one variable that overrides it.
"""

from __future__ import annotations

import contextlib
import os
import socket
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from cachellm.providers.catalog import ALSO_READS, HOSTS_BY_KEY, Host

#: Order of preference when several are available. An explicit API key is a
#: stronger statement of intent than a server that merely happens to be running,
#: so keyed hosts come first; Ollama and Bedrock follow because they are free or
#: already paid for, and the test double is the last resort.
PREFERENCE: tuple[str, ...] = (
    "openai",
    "anthropic",
    "gemini",
    "xai",
    "groq",
    "deepseek",
    "mistral",
    "openrouter",
    "together",
    "fireworks",
    "cerebras",
    "perplexity",
    "moonshot",
    "ollama",
    "bedrock",
    "fake",
)


@dataclass(frozen=True)
class Detected:
    host: Host
    reason: str
    api_key: str | None = None

    @property
    def key(self) -> str:
        return self.host.key


def env_key_for(host: Host) -> tuple[str, str] | None:
    """Return (variable name, value) for the first populated key variable."""
    candidates = [host.env_key, *ALSO_READS.get(host.key, ())] if host.env_key else []
    for name in candidates:
        if name and (value := os.environ.get(name, "").strip()):
            return name, value
    return None


def port_open(url: str, timeout: float = 0.35) -> bool:
    """Cheap liveness check for a local server, without an HTTP round trip."""
    parsed = urlparse(url)
    host, port = parsed.hostname, parsed.port
    if not host or not port:
        return False
    with contextlib.suppress(OSError), socket.create_connection((host, port), timeout=timeout):
        return True
    return False


def aws_credentials_available() -> bool:
    """True when boto3 is installed and can resolve credentials from anywhere."""
    try:
        import boto3
    except ImportError:
        return False
    with contextlib.suppress(Exception):
        return boto3.Session().get_credentials() is not None
    return False


def available(include_fake: bool = True) -> list[Detected]:
    """Everything this machine could route to right now, best first."""
    found: list[Detected] = []
    for key in PREFERENCE:
        host = HOSTS_BY_KEY.get(key)
        if host is None:
            continue
        if key == "fake":
            if include_fake:
                found.append(Detected(host, "always available, costs nothing"))
            continue
        if key == "ollama":
            if host.base_url and port_open(host.base_url):
                found.append(Detected(host, "running on this machine"))
            continue
        if key == "bedrock":
            if aws_credentials_available():
                found.append(Detected(host, "AWS credentials resolved"))
            continue
        if (pair := env_key_for(host)) is not None:
            name, value = pair
            found.append(Detected(host, f"{name} is set", value))
    return found


def choose(explicit_base_url: str = "", explicit_key: str = "") -> Detected | None:
    """Pick one host, or None when the operator has already configured it.

    A configured base URL means the decision is already made, so detection
    steps aside rather than overriding it.
    """
    if explicit_base_url.strip():
        return None
    options = available()
    return options[0] if options else None


def apply(settings: Any) -> str | None:
    """Configure `settings` from what this machine already has.

    Runs only when the operator has said nothing. Any of
    CACHELLM_DEFAULT_PROVIDER, CACHELLM_OPENAI_BASE_URL or
    CACHELLM_OPENAI_API_KEY being set means the decision is already made, and
    detection must not second-guess it.

    "Set" means set anywhere settings are read from, not just exported. A base
    URL written in a .env file used to be replaced by whichever vendor key
    happened to be in the shell, because only the process environment was
    checked. Settings already records which fields came from any source.

    Returns a one-line description of what it configured, or None.
    """
    fields = ("default_provider", "openai_base_url", "openai_api_key")
    exported = any(os.environ.get(f"CACHELLM_{name.upper()}", "").strip() for name in fields)
    # An empty `CACHELLM_OPENAI_API_KEY=` line is a placeholder, not a decision.
    written = set(getattr(settings, "model_fields_set", ()))
    configured = any(
        name in written and str(getattr(settings, name, "") or "").strip() for name in fields
    )
    if exported or configured:
        return None

    options = available(include_fake=False)
    if not options:
        return None

    chosen = options[0]
    settings.default_provider = chosen.host.provider
    if chosen.host.base_url:
        settings.openai_base_url = chosen.host.base_url
    if chosen.api_key:
        settings.openai_api_key = chosen.api_key
    return describe(chosen)


def describe(detected: Detected) -> str:
    """One line explaining a choice, for the startup log."""
    host = detected.host
    where = f" at {host.base_url}" if host.base_url else ""
    return f"{host.name}{where} ({detected.reason})"
