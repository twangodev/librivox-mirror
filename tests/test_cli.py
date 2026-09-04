import io
import json
from threading import Event
from typing import cast

import httpx
import respx
from click import unstyle
from rich.console import Console
from typer import Context
from typer.main import get_command
from typer.testing import CliRunner

from librivox_mirror.catalog import CATALOG_URL
from librivox_mirror.cli import AppSettings, app, publish_batches
from librivox_mirror.hub import PublishResult
from librivox_mirror.models import Book, BookStatus, SyncState
from librivox_mirror.state import RunLock, StateStore
from librivox_mirror.sync import select_pending_batch
from librivox_mirror.workflow import BookOutcome, MirrorRunner

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert unstyle(result.stdout).strip() == "0.0.0"


def test_cli_disables_huggingface_progress_renderers(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        "librivox_mirror.cli.disable_huggingface_progress_bars",
        lambda: calls.append(None),
    )

    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert calls == [None]


def test_help_names_the_command() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    output = unstyle(result.stdout)
    assert "librivox-mirror" in output
    for command in (
        "plan",
        "mirror",
        "backfill",
        "repair",
        "sync",
        "reconcile",
        "status",
        "verify",
    ):
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


def test_backfill_concurrency_has_no_arbitrary_upper_limits() -> None:
    result = runner.invoke(
        app,
        [
            "backfill",
            "--start-id",
            "20",
            "--end-id",
            "10",
            "--repo",
            "owner/repo",
            "--download-jobs",
            "128",
            "--book-jobs",
            "64",
            "--upload-jobs",
            "128",
        ],
    )

    assert result.exit_code == 2
    assert "start-id" in unstyle(result.output)
    assert "not in the range" not in unstyle(result.output)


def test_backfill_keeps_jobs_as_a_download_jobs_alias() -> None:
    result = runner.invoke(
        app,
        [
            "backfill",
            "--start-id",
            "20",
            "--end-id",
            "10",
            "--repo",
            "owner/repo",
            "--jobs",
            "128",
        ],
    )

    assert result.exit_code == 2
    assert "start-id" in unstyle(result.output)
    assert "not in the range" not in unstyle(result.output)


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


class OverlapRunner(RecordingRunner):
    def __init__(self, second_book_id: int) -> None:
        super().__init__()
        self.second_book_id = second_book_id
        self.second_book_prepared = Event()
        self.publications = 0

    def prepare_book_resiliently(self, book, *, progress):
        if book.id == self.second_book_id:
            self.second_book_prepared.set()
        return BookOutcome(book=book, skipped=True)

    def publish(self, outcomes, sync_state, *, commit_message):
        self.publications += 1
        if self.publications == 1:
            assert self.second_book_prepared.wait(timeout=1)
        return super().publish(outcomes, sync_state, commit_message=commit_message)


class CurrentBookPublisher:
    def __init__(self, current_ids: set[int]) -> None:
        self.current_ids = current_ids

    def has_current_book(self, book: Book) -> bool:
        return book.id in self.current_ids


def test_pending_catalog_batch_stops_at_the_ci_limit(book: Book) -> None:
    books = [
        book,
        book,
        book.model_copy(update={"id": 48}),
        book.model_copy(update={"id": 49}),
        book.model_copy(update={"id": 50}),
    ]

    batch = select_pending_batch(
        books,
        CurrentBookPublisher({book.id, 49}),
        max_books=2,
    )

    assert [book.id for book in batch.books] == [48, 50]
    assert batch.reached_catalog_end is False
    assert batch.all_pending_selected is False
    assert batch.scanned_count == 4


def test_pending_catalog_batch_reports_a_complete_scan(book: Book) -> None:
    books = [book, book.model_copy(update={"id": 48})]

    batch = select_pending_batch(
        books,
        CurrentBookPublisher({book.id}),
        max_books=2,
    )

    assert [book.id for book in batch.books] == [48]
    assert batch.reached_catalog_end is True
    assert batch.all_pending_selected is True
    assert batch.scanned_count == 2


def test_preparation_overlaps_publication(book: Book) -> None:
    books = [book, book.model_copy(update={"id": book.id + 1})]
    runner = OverlapRunner(books[1].id)
    output = Console(file=io.StringIO(), force_terminal=False)
    context = Context(get_command(app))
    context.obj = AppSettings(
        json_output=False,
        verbose=False,
        output=output,
        errors=output,
    )

    publish_batches(
        context,
        cast(MirrorRunner, runner),
        books,
        SyncState(),
        commit_size=1,
        book_jobs=1,
    )

    assert runner.publications == 2


def test_failed_batches_do_not_advance_the_catalog_watermark(book: Book, monkeypatch) -> None:
    books = [book, book.model_copy(update={"id": 48})]
    runner = RecordingRunner()

    def prepare(context, mirror_runner, batch, *, workers, capacity):
        for item in batch:
            yield (
                BookOutcome(book=item, error="SourceUnavailableError: unavailable")
                if item.id == book.id
                else BookOutcome(book=item, skipped=True)
            )

    monkeypatch.setattr("librivox_mirror.cli.prepare_books_stream", prepare)

    publish_batches(
        cast(Context, None),
        cast(MirrorRunner, runner),
        books,
        SyncState(
            catalog_watermark=10,
            catalog_scan_started_at=15,
            catalog_scan_after_book_id=40,
        ),
        commit_size=1,
        final_catalog_watermark=20,
        catalog_scan_started_at=15,
    )

    assert runner.states[-1].catalog_watermark == 10
    assert runner.states[-1].catalog_scan_after_book_id == 40


def test_catalog_catchup_checkpoints_each_published_batch(book: Book, monkeypatch) -> None:
    books = [book, book.model_copy(update={"id": 48})]
    runner = RecordingRunner()

    def prepare(context, mirror_runner, batch, *, workers, capacity):
        for item in batch:
            yield BookOutcome(book=item, skipped=True)

    monkeypatch.setattr("librivox_mirror.cli.prepare_books_stream", prepare)

    publish_batches(
        cast(Context, None),
        cast(MirrorRunner, runner),
        books,
        SyncState(),
        commit_size=1,
        catalog_scan_started_at=100,
    )

    assert [state.catalog_scan_after_book_id for state in runner.states] == [book.id, 48]
    assert all(state.catalog_scan_started_at == 100 for state in runner.states)


def test_catalog_catchup_clears_cursor_when_the_scan_completes(book: Book, monkeypatch) -> None:
    books = [book, book.model_copy(update={"id": 48})]
    runner = RecordingRunner()

    def prepare(context, mirror_runner, batch, *, workers, capacity):
        for item in batch:
            yield BookOutcome(book=item, skipped=True)

    monkeypatch.setattr("librivox_mirror.cli.prepare_books_stream", prepare)

    publish_batches(
        cast(Context, None),
        cast(MirrorRunner, runner),
        books,
        SyncState(),
        commit_size=1,
        final_catalog_watermark=100,
        catalog_scan_started_at=100,
    )

    assert runner.states[-1].catalog_watermark == 100
    assert runner.states[-1].catalog_scan_started_at is None
    assert runner.states[-1].catalog_scan_after_book_id is None


def test_sync_resumes_catchup_and_checkpoints_an_empty_tail(book: Book, monkeypatch) -> None:
    published_states = []
    catalog_calls = []

    class FakePublisher:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def load_sync_state(self) -> SyncState:
            return SyncState(
                catalog_scan_started_at=100,
                catalog_scan_after_book_id=book.id,
            )

        def has_current_book(self, candidate: Book) -> bool:
            return True

        def publish_sync_state(self, state: SyncState, *, commit_message: str) -> PublishResult:
            published_states.append(state)
            return PublishResult(revision="state-revision", state=state)

    class FakeCatalog:
        def __init__(self, *, user_agent: str, retry_forever: bool) -> None:
            catalog_calls.append(("init", retry_forever))

        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def iter_books(self, *, since: int, start_id: int | None = None):
            catalog_calls.append(("iter", since, start_id))
            yield book.model_copy(update={"id": book.id + 1})

    monkeypatch.setattr("librivox_mirror.cli.HubPublisher", FakePublisher)
    monkeypatch.setattr("librivox_mirror.cli.LibriVoxCatalog", FakeCatalog)
    monkeypatch.setattr("librivox_mirror.cli.time.time", lambda: 200)

    result = runner.invoke(app, ["--json", "sync", "--repo", "owner/repo"])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["window_fully_processed"] is True
    assert catalog_calls == [("init", True), ("iter", 0, book.id + 1)]
    assert published_states[0].catalog_watermark == 100
    assert published_states[0].catalog_scan_started_at is None
    assert published_states[0].catalog_scan_after_book_id is None
