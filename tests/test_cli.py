import json
from typing import cast

import httpx
import respx
from click import unstyle
from typer import Context
from typer.testing import CliRunner

from librivox_mirror.catalog import CATALOG_URL
from librivox_mirror.cli import app, publish_batches
from librivox_mirror.models import Book, BookStatus, SyncState
from librivox_mirror.state import RunLock, StateStore
from librivox_mirror.workflow import BookOutcome, MirrorRunner

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert unstyle(result.stdout).strip() == "0.0.0"


def test_help_names_the_command() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    output = unstyle(result.stdout)
    assert "librivox-mirror" in output
    for command in ("plan", "mirror", "backfill", "sync", "reconcile", "status", "verify"):
        assert command in output


def test_json_version_keeps_stdout_machine_readable() -> None:
    result = runner.invoke(app, ["--json", "version"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {"version": "0.0.0"}


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
    assert "start-id" in unstyle(result.output)


def test_status_reports_persistent_progress_as_json(book, tmp_path) -> None:
    state_path = tmp_path / "state.sqlite3"
    with StateStore(state_path) as state:
        state.discover(book)
        state.begin_attempt(book.id)
        state.transition(book.id, BookStatus.RESOLVED, archive_identifier="a_test_book")
        state.record_failure(book.id, RuntimeError("connection lost"))

    result = runner.invoke(app, ["--json", "status", "--state", str(state_path)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["integrity"] == "ok"
    assert payload["counts"]["resolved"] == 1
    assert payload["recent_problems"][0]["attempts"] == 1
    assert "connection lost" in payload["recent_problems"][0]["error"]


def test_status_does_not_create_a_missing_database(tmp_path) -> None:
    state_path = tmp_path / "missing.sqlite3"

    result = runner.invoke(app, ["--json", "status", "--state", str(state_path)])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["exists"] is False
    assert not state_path.exists()


def test_status_reports_the_active_run(book, tmp_path) -> None:
    state_path = tmp_path / "state.sqlite3"
    with StateStore(state_path) as state:
        state.discover(book)

    with RunLock(state_path):
        result = runner.invoke(app, ["--json", "status", "--state", str(state_path)])

    assert result.exit_code == 0
    active_run = json.loads(result.stdout)["active_run"]
    assert active_run["pid"] > 0
    assert active_run["started_at"]


class RecordingRunner:
    def __init__(self) -> None:
        self.states: list[SyncState] = []

    def publish(self, outcomes, sync_state, *, commit_message):
        self.states.append(sync_state)
        return None


def test_failed_batches_do_not_advance_the_catalog_watermark(book: Book, monkeypatch) -> None:
    books = [book, book.model_copy(update={"id": 48})]
    runner = RecordingRunner()

    def prepare(context, mirror_runner, batch):
        return [
            BookOutcome(book=item, error="SourceUnavailableError: unavailable")
            if item.id == book.id
            else BookOutcome(book=item, skipped=True)
            for item in batch
        ]

    monkeypatch.setattr("librivox_mirror.cli.prepare_books", prepare)

    publish_batches(
        cast(Context, None),
        cast(MirrorRunner, runner),
        books,
        SyncState(catalog_watermark=10),
        commit_size=1,
        final_catalog_watermark=20,
    )

    assert runner.states[-1].catalog_watermark == 10
