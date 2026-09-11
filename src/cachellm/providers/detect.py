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
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from cachellm.providers.catalog import (
    ALSO_READS,
    HOSTS_BY_KEY,
    LOCAL_HOSTS,
    Host,
    host_for_base_url,
)

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


_EXPLICIT_FIELDS = ("default_provider", "openai_base_url", "openai_api_key")


def configured(settings: Any) -> bool:
    """True when the operator has already said where requests go.

    Any of CACHELLM_DEFAULT_PROVIDER, CACHELLM_OPENAI_BASE_URL or
    CACHELLM_OPENAI_API_KEY counts, whether exported or written in a .env file.
    A base URL written in .env used to lose to whichever vendor key happened to
    be exported, because only the process environment was checked. An empty
    `CACHELLM_OPENAI_API_KEY=` line is a placeholder, not a decision.
    """
    exported = any(
        os.environ.get(f"CACHELLM_{name.upper()}", "").strip() for name in _EXPLICIT_FIELDS
    )
    written = set(getattr(settings, "model_fields_set", ()))
    return exported or any(
        name in written and str(getattr(settings, name, "") or "").strip()
        for name in _EXPLICIT_FIELDS
    )


# ---------------------------------------------------------------- route plan


@dataclass(frozen=True)
class Route:
    """One upstream the proxy can send a request to."""

    key: str  # catalog host key, "custom", "bedrock" or "fake"
    name: str
    adapter: str  # "openai", "bedrock" or "fake"
    base_url: str | None = None
    api_key: str | None = field(default=None, repr=False)
    reason: str = ""
    local: bool = False

    @property
    def keyless(self) -> bool:
        """A catalogued host that expects a key and has none, so cannot answer."""
        host = HOSTS_BY_KEY.get(self.key)
        return bool(host and host.needs_key and not self.api_key)


@dataclass(frozen=True)
class RoutePlan:
    """Every upstream in play, and which one gets names nobody else claims."""

    routes: dict[str, Route]
    default: str
    #: The hosts the operator actually has, default first. Bedrock and the test
    #: double are always routable by name, but only listed here when chosen.
    enabled: tuple[str, ...]
    #: "configured", "detected" or "fallback".
    source: str

    def openai_routes(self) -> list[Route]:
        return [self.routes[k] for k in self.enabled if self.routes[k].adapter == "openai"]


_BUILT_IN = {
    "bedrock": Route("bedrock", "AWS Bedrock", "bedrock", reason="Bedrock ids route here by name"),
    "fake": Route("fake", "Test double", "fake", reason="always available"),
}


def _is_local(url: str | None) -> bool:
    return bool(url) and (urlparse(url).hostname or "") in {"localhost", "127.0.0.1", "::1"}


def _wanted(settings: Any) -> list[str]:
    raw = str(getattr(settings, "hosts", "") or "")
    keys = [k.strip().lower() for k in raw.split(",") if k.strip()]
    return list(dict.fromkeys(k for k in keys if k in HOSTS_BY_KEY and k != "fake"))


def _route_from(detected: Detected) -> Route:
    host = detected.host
    if host.provider == "bedrock":
        return Route("bedrock", host.name, "bedrock", reason=detected.reason)
    return Route(
        host.key,
        host.name,
        host.provider,
        host.base_url,
        detected.api_key,
        detected.reason,
        local=host.key in LOCAL_HOSTS,
    )


def _named_route(key: str) -> Route | None:
    """A route for a host asked for by name in CACHELLM_HOSTS, if it can answer."""
    host = HOSTS_BY_KEY[key]
    if host.provider == "bedrock":
        return Route("bedrock", host.name, "bedrock", reason="named in CACHELLM_HOSTS")
    pair = env_key_for(host)
    if host.needs_key and pair is None:
        return None
    reason = "named in CACHELLM_HOSTS" + (f", {pair[0]} is set" if pair else "")
    return Route(
        key,
        host.name,
        "openai",
        host.base_url,
        pair[1] if pair else None,
        reason,
        local=key in LOCAL_HOSTS,
    )


def plan(settings: Any) -> RoutePlan:
    """Work out every upstream this proxy can use, and the default among them.

    Nothing configured: every host this machine has a key for, plus Ollama when
    it is running and Bedrock when AWS credentials resolve, best first. The
    first is the default. CACHELLM_HOSTS narrows and reorders that list.

    Configured: the endpoint the operator set, exactly as before, plus only the
    hosts they name in CACHELLM_HOSTS. Someone pointing the proxy at their own
    gateway wants everything to go there, not to whichever vendor key happens
    to be exported.
    """
    wanted = _wanted(settings)
    if configured(settings):
        base = str(settings.openai_base_url)
        host = host_for_base_url(base)
        pair = env_key_for(host) if host else None
        api_key = settings.openai_api_key or (pair[1] if pair else "")
        explicit_url = "openai_base_url" in set(getattr(settings, "model_fields_set", ())) or bool(
            os.environ.get("CACHELLM_OPENAI_BASE_URL", "").strip()
        )
        first = Route(
            host.key if host else "custom",
            host.name if host else "Your endpoint",
            "openai",
            base,
            api_key,
            "set in CACHELLM_OPENAI_BASE_URL" if explicit_url else "the default endpoint",
            local=(host is not None and host.key in LOCAL_HOSTS) or _is_local(base),
        )
        ordered = [first]
        for key in wanted:
            if key != first.key and (route := _named_route(key)) is not None:
                ordered.append(route)
        default = {"bedrock": "bedrock", "fake": "fake"}.get(
            str(settings.default_provider), first.key
        )
        source = "configured"
    else:
        found = available(include_fake=False)
        if wanted:
            by_key = {d.host.key: d for d in found}
            ordered = []
            for key in wanted:
                if key in by_key:
                    ordered.append(_route_from(by_key[key]))
                elif key in LOCAL_HOSTS and (route := _named_route(key)) is not None:
                    ordered.append(route)  # named but not running yet; it may start later
        else:
            ordered = [_route_from(d) for d in found]
        if ordered:
            default, source = ordered[0].key, "detected"
        else:
            ordered = [
                Route(
                    "openai",
                    "OpenAI",
                    "openai",
                    str(settings.openai_base_url),
                    settings.openai_api_key,
                    "nothing detected",
                )
            ]
            default, source = "openai", "fallback"

    routes = {route.key: route for route in ordered}
    for key, route in _BUILT_IN.items():
        routes.setdefault(key, route)
    enabled = tuple(dict.fromkeys([default, *(r.key for r in ordered)]))
    return RoutePlan(routes=routes, default=default, enabled=enabled, source=source)


def apply_plan(settings: Any, route_plan: RoutePlan) -> None:
    """Mirror the default route into settings, for anything that reads them."""
    route = route_plan.routes[route_plan.default]
    if route.adapter == "openai":
        settings.default_provider = "openai"
        settings.openai_base_url = route.base_url or settings.openai_base_url
        settings.openai_api_key = route.api_key or ""
    else:
        settings.default_provider = route.adapter


def apply(settings: Any) -> str | None:
    """Configure `settings` from what this machine already has.

    Runs only when the operator has said nothing (see `configured`). Returns a
    one-line description of the default it chose, or None.
    """
    route_plan = plan(settings)
    if route_plan.source != "detected":
        return None
    apply_plan(settings, route_plan)
    return describe_route(route_plan.routes[route_plan.default])


def describe_route(route: Route) -> str:
    """One line explaining a route, for the startup log."""
    where = f" at {route.base_url}" if route.base_url else ""
    return f"{route.name}{where} ({route.reason})"


def describe(detected: Detected) -> str:
    """One line explaining a choice, for the startup log."""
    host = detected.host
    where = f" at {host.base_url}" if host.base_url else ""
    return f"{host.name}{where} ({detected.reason})"
