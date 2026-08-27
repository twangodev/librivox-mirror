import sqlite3

from librivox_mirror.models import Book, BookStatus, QuarantineCode, QuarantineRecord
from librivox_mirror.state import StateStore


def test_state_persists_progress_across_processes(book: Book, tmp_path) -> None:
    path = tmp_path / "state.sqlite3"
    artifact = tmp_path / "book.tar"
    with StateStore(path) as state:
        assert state.discover(book).status == BookStatus.DISCOVERED
        state.transition(book.id, BookStatus.RESOLVED, archive_identifier="a_test_book")
        state.transition(
            book.id,
            BookStatus.PACKED,
            artifact_path=artifact,
            artifact_sha256="abc",
        )

    with StateStore(path) as state:
        checkpoint = state.get(book.id)

    assert checkpoint is not None
    assert checkpoint.status == BookStatus.PACKED
    assert checkpoint.artifact_path == artifact
    assert checkpoint.artifact_sha256 == "abc"


def test_changed_source_resets_progress(book: Book, tmp_path) -> None:
    with StateStore(tmp_path / "state.sqlite3") as state:
        state.discover(book)
        state.transition(book.id, BookStatus.PUBLISHED, published_revision="revision")

        checkpoint = state.discover(book.model_copy(update={"title": "Changed"}))

    assert checkpoint.status == BookStatus.DISCOVERED
    assert checkpoint.published_revision is None


def test_quarantine_is_durable(book: Book, tmp_path) -> None:
    record = QuarantineRecord(
        book_id=book.id,
        title=book.title,
        code=QuarantineCode.ORIGINAL_FILE_MISSING,
        detail="missing",
        source_fingerprint=book.source_fingerprint,
    )
    with StateStore(tmp_path / "state.sqlite3") as state:
        checkpoint = state.quarantine(record)

    assert checkpoint.status == BookStatus.QUARANTINED
    assert checkpoint.error_code == QuarantineCode.ORIGINAL_FILE_MISSING


def test_restart_clears_progress_for_safe_replay(book: Book, tmp_path) -> None:
    with StateStore(tmp_path / "state.sqlite3") as state:
        state.discover(book)
        state.transition(book.id, BookStatus.RESOLVED, archive_identifier="a_test_book")

        checkpoint = state.restart(book.id)

    assert checkpoint.status == BookStatus.DISCOVERED
    assert checkpoint.archive_identifier is None
    assert checkpoint.artifact_path is None


def test_attempt_failure_is_persisted(book: Book, tmp_path) -> None:
    with StateStore(tmp_path / "state.sqlite3") as state:
        state.discover(book)
        state.begin_attempt(book.id)

        checkpoint = state.record_failure(book.id, RuntimeError("connection lost"))

    assert checkpoint.attempt_count == 1
    assert checkpoint.last_started_at is not None
    assert checkpoint.last_error == "RuntimeError: connection lost"


def test_existing_state_schema_is_migrated(book: Book, tmp_path) -> None:
    path = tmp_path / "state.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE books (
                book_id INTEGER PRIMARY KEY,
                source_fingerprint TEXT NOT NULL,
                status TEXT NOT NULL,
                archive_identifier TEXT,
                artifact_path TEXT,
                artifact_sha256 TEXT,
                error_code TEXT,
                error_detail TEXT,
                published_revision TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO books (book_id, source_fingerprint, status, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (book.id, book.source_fingerprint, BookStatus.DISCOVERED, "2026-01-01T00:00:00+00:00"),
        )

    with StateStore(path) as state:
        checkpoint = state.get(book.id)

    assert checkpoint is not None
    assert checkpoint.attempt_count == 0
    assert checkpoint.last_error is None


def test_state_uses_full_synchronous_wal_and_checkpoints_on_close(book: Book, tmp_path) -> None:
    path = tmp_path / "state.sqlite3"
    with StateStore(path) as state:
        state.discover(book)
        synchronous = state._connection.execute("PRAGMA synchronous").fetchone()[0]
        journal_mode = state._connection.execute("PRAGMA journal_mode").fetchone()[0]

    assert synchronous == 2
    assert journal_mode == "wal"
    assert not path.with_name(f"{path.name}-wal").exists()
