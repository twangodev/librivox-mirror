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
