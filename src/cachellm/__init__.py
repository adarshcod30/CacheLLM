"""CacheLLM: a drop-in semantic cache for OpenAI-compatible LLM APIs."""

from __future__ import annotations

__version__ = "0.2.1"

__all__ = ["__version__", "main"]


def main() -> None:  # pragma: no cover - thin console-script shim
    from cachellm.cli import app

    app()
