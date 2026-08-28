import httpx
import pytest

from librivox_mirror.network import is_transient_http_error


@pytest.mark.parametrize("status_code", [408, 409, 425, 429, 500, 503])
def test_retryable_http_statuses(status_code: int) -> None:
    request = httpx.Request("POST", "https://example.test")
    response = httpx.Response(status_code, request=request)
    error = httpx.HTTPStatusError("failed", request=request, response=response)

    assert is_transient_http_error(error)


@pytest.mark.parametrize("status_code", [400, 401, 403, 404])
def test_non_retryable_http_statuses(status_code: int) -> None:
    request = httpx.Request("POST", "https://example.test")
    response = httpx.Response(status_code, request=request)
    error = httpx.HTTPStatusError("failed", request=request, response=response)

    assert not is_transient_http_error(error)


def test_transport_failures_are_transient() -> None:
    request = httpx.Request("GET", "https://example.test")

    assert is_transient_http_error(httpx.ReadTimeout("timed out", request=request))
    assert not is_transient_http_error(ValueError("invalid payload"))
