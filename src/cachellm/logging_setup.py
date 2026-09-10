"""structlog configuration.

Prompt and completion text is treated as sensitive by default. A proxy sees
every question every user asks, so logging it wholesale is a privacy incident
waiting to happen. Set ``CACHELLM_LOG_PROMPTS=true`` only in local development.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from cachellm.settings import Settings

SENSITIVE_KEYS = {"prompt", "matched_prompt", "response_text", "messages", "content", "answer"}


def _redact_prompts(_logger: Any, _name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    for key in list(event_dict):
        if key in SENSITIVE_KEYS:
            value = event_dict[key]
            if isinstance(value, str):
                event_dict[key] = f"<redacted {len(value)} chars>"
            else:
                event_dict[key] = "<redacted>"
    return event_dict


def configure_logging(settings: Settings) -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    if not settings.log_prompts:
        processors.append(_redact_prompts)
    processors.append(structlog.processors.StackInfoRenderer())
    processors.append(structlog.processors.format_exc_info)
    processors.append(
        structlog.processors.JSONRenderer()
        if settings.log_json
        else structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
