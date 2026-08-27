from __future__ import annotations

import httpx
import requests

TRANSIENT_HTTP_STATUSES = frozenset({408, 409, 425, 429})


def is_transient_http_error(error: Exception) -> bool:
    if isinstance(error, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(error, httpx.HTTPError):
        response = getattr(error, "response", None)
        return response is not None and is_transient_status(response.status_code)
    if isinstance(error, (requests.Timeout, requests.ConnectionError)):
        return True
    if isinstance(error, requests.HTTPError):
        return error.response is not None and is_transient_status(error.response.status_code)
    return False


def is_transient_status(status_code: int) -> bool:
    return status_code in TRANSIENT_HTTP_STATUSES or status_code >= 500
