"""API-key authentication and caller identity.

When `API_KEYS` is configured, requests must present a matching key in the
`X-API-Key` header; the caller is then identified by a non-reversible hash of
that key, which is used to scope conversation history. When no keys are
configured, authentication is disabled and every caller shares the "public"
owner, so local development and the tests need no setup.
"""

import hashlib

from fastapi import Request

from app.core.config import get_settings
from app.core.exceptions import AuthenticationError

PUBLIC_OWNER = "public"


def get_principal(request: Request) -> str:
    """Authenticate the request and return the owner id to scope data by."""
    api_keys = get_settings().api_key_set
    if not api_keys:
        return PUBLIC_OWNER

    presented = request.headers.get("x-api-key")
    if presented is None or presented not in api_keys:
        raise AuthenticationError("A valid X-API-Key header is required.")
    return "key:" + hashlib.sha256(presented.encode()).hexdigest()[:16]
