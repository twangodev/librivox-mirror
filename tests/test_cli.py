import json

import httpx
import respx
from typer.testing import CliRunner

from librivox_mirror.catalog import CATALOG_URL
from librivox_mirror.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "0.1.0"


def test_help_names_the_command() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "librivox-mirror" in result.stdout
    for command in ("plan", "mirror", "backfill", "sync", "reconcile", "verify"):
        assert command in result.stdout
    assert "\x1b[" not in result.stdout


def test_json_version_keeps_stdout_machine_readable() -> None:
    result = runner.invoke(app, ["--json", "version"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {"version": "0.1.0"}


@respx.mock
def test_plan_json_is_bounded_and_preserves_clean_stdout() -> None:
    respx.get(CATALOG_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "books": [
                    {
                        "id": "47",
                        "title": "A Test Book",
                        "url_librivox": "https://librivox.org/a-test-book/",
                        "url_iarchive": "https://archive.org/details/a_test_book",
                        "sections": [],
                    }
                ]
            },
        )
    )

    result = runner.invoke(app, ["--json", "plan", "--since", "0", "--max-books", "1"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["count"] == 1
    assert payload["books"][0]["book_id"] == 47


def test_invalid_range_is_a_usage_error() -> None:
    result = runner.invoke(
        app,
        ["backfill", "--start-id", "20", "--end-id", "10", "--repo", "owner/repo"],
    )

    assert result.exit_code == 2
    assert "start-id" in result.output
