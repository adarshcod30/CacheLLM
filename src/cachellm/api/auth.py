"""Client authentication.

The proxy holds provider credentials, so an unauthenticated proxy on a public
address is an open wallet. Keys are compared in constant time and never logged.
Auth is off only when no keys are configured, which is the local-dev case and
is reported loudly at startup.
"""

from __future__ import annotations

import hmac

from fastapi import Request

from cachellm.errors import AuthError


def extract_key(request: Request) -> str:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return request.headers.get("api-key", "").strip()


def verify(request: Request, keys: set[str]) -> None:
    if not keys:
        return
    presented = extract_key(request)
    if not presented:
        raise AuthError("Missing API key. Pass it as `Authorization: Bearer <key>`.")
    for known in keys:
        if hmac.compare_digest(presented, known):
            return
    raise AuthError()
