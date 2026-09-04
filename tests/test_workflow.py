import hashlib
from typing import cast

import httpx
import pytest

from librivox_mirror.archive import (
    InternetArchiveClient,
    QuarantinedBookError,
    resolve_original_files,
)
from librivox_mirror.artifact import ArtifactBuildError
from librivox_mirror.capacity import StagingCapacity
from librivox_mirror.catalog import LibriVoxCatalog
from librivox_mirror.models import (
    Book,
    BookStatus,
    DownloadedSection,
    QuarantineCode,
    QuarantineRecord,
    SyncState,
)
from librivox_mirror.state import StateStore
from librivox_mirror.workflow import MirrorRunner


class PreparedArchive:
    def __init__(self, book: Book, tmp_path) -> None:
        self.content = b"".join(b"\xff\xfb\x90\x64" + bytes(413) for _ in range(20))
        self.resolved = resolve_original_files(
            book,
            "a_test_book",
            [
                {
                    "name": "chapter_64kb.mp3",
                    "source": "derivative",
                    "format": "64Kbps MP3",
                    "original": "chapter.mp3",
                },
                {
                    "name": "chapter.mp3",
                    "source": "original",
                    "format": "VBR MP3",
                    "size": str(len(self.content)),
                },
            ],
        )
        self.tmp_path = tmp_path

    def resolve_book(self, book):
        return self.resolved

    def download_book(self, resolved, destination, *, progress=None):
        if progress is not None:
            progress(len(self.content), len(self.content))
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / "section.mp3"
        path.write_bytes(self.content)
        return (
            DownloadedSection(
                resolved=resolved.sections[0],
                path=path,
                sha256=hashlib.sha256(self.content).hexdigest(),
            ),
        )


class QuarantineArchive:
    def __init__(self, book: Book) -> None:
        self.record = QuarantineRecord(
            book_id=book.id,
            title=book.title,
            code=QuarantineCode.ORIGINAL_FILE_MISSING,
            detail="missing",
            source_fingerprint=book.source_fingerprint,
        )

    def resolve_book(self, book):
        raise QuarantinedBookError(self.record)


class FailingArchive:
    def resolve_book(self, book):
        raise RuntimeError("connection lost")


class UnavailableArchive:
    def resolve_book(self, book):
        from librivox_mirror.archive import SourceUnavailableError

        raise SourceUnavailableError("archive edge unavailable")


class FakePublisher:
    def __init__(self, *, current: bool = False) -> None:
        self.current = current
        self.published = []

    def has_current_book(self, book: Book) -> bool:
        return self.current

    def publish(self, artifacts, quarantines, sync_state, *, commit_message):
        from librivox_mirror.hub import PublishResult

        self.published.extend(artifacts)
        return PublishResult(revision="revision", state=sync_state)

    def invalidate_cache(self) -> None:
        pass

    def current_revision(self) -> str:
        return "revision"

    def load_sync_state(self) -> SyncState:
        return SyncState()


class QuarantinedPublisher(FakePublisher):
    def has_current_book(self, book: Book, *, include_quarantined: bool = True) -> bool:
        return include_quarantined


class TransientPublisher(FakePublisher):
    def __init__(self) -> None:
        super().__init__()
        self.attempts = 0

    def publish(self, artifacts, quarantines, sync_state, *, commit_message):
        self.attempts += 1
        if self.attempts == 1:
            request = httpx.Request("POST", "https://huggingface.co/api/datasets/commit")
            raise httpx.ReadTimeout("timed out", request=request)
        return super().publish(
            artifacts,
            quarantines,
            sync_state,
            commit_message=commit_message,
        )


class AmbiguousPublisher(FakePublisher):
    def __init__(self) -> None:
        super().__init__()
        self.attempts = 0

    def publish(self, artifacts, quarantines, sync_state, *, commit_message):
        self.attempts += 1
        self.current = True
        request = httpx.Request("POST", "https://huggingface.co/api/datasets/commit")
        raise httpx.ReadTimeout("response lost", request=request)

    def load_sync_state(self) -> SyncState:
        return SyncState(published_books=1, published_sections=1)

    def current_revision(self) -> str:
        return "committed-revision"


def make_runner(
    book: Book,
    archive,
    tmp_path,
    state: StateStore,
    *,
    publisher=None,
    staging_capacity=None,
    retry_quarantined=False,
) -> MirrorRunner:
    return MirrorRunner(
        catalog=cast(LibriVoxCatalog, None),
        archive=cast(InternetArchiveClient, archive),
        state=state,
        staging_directory=tmp_path / "staging",
        publisher=publisher,
        source_index=publisher,
        staging_capacity=staging_capacity,
        retry_quarantined=retry_quarantined,
    )


def test_prepare_book_advances_to_deterministic_artifact(book: Book, tmp_path) -> None:
    with StateStore(tmp_path / "state.sqlite3") as state:
        outcome = make_runner(book, PreparedArchive(book, tmp_path), tmp_path, state).prepare_book(
            book
        )
        checkpoint = state.get(book.id)

    assert outcome.artifact is not None
    assert outcome.artifact.path.exists()
    assert checkpoint is not None
    assert checkpoint.status == BookStatus.PACKED
    assert not (tmp_path / "staging/downloads/000047").exists()
    assert (tmp_path / "staging/manifests/000047.json").exists()


def test_prepare_book_quarantines_without_downloading(book: Book, tmp_path) -> None:
    with StateStore(tmp_path / "state.sqlite3") as state:
        outcome = make_runner(book, QuarantineArchive(book), tmp_path, state).prepare_book(book)
        checkpoint = state.get(book.id)

    assert outcome.quarantine is not None
    assert outcome.artifact is None
    assert checkpoint is not None
    assert checkpoint.status == BookStatus.QUARANTINED


def test_published_resume_cleans_staging_without_source_requests(book: Book, tmp_path) -> None:
    staging = tmp_path / "staging"
    download_directory = staging / "downloads/000047"
    download_directory.mkdir(parents=True)
    (download_directory / "section.mp3").write_bytes(b"staged")
    artifact = staging / "repository/data/000/000047.tar"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"staged")
    with StateStore(tmp_path / "state.sqlite3") as state:
        state.discover(book)
        state.transition(book.id, BookStatus.PACKED, artifact_path=artifact)
        state.transition(book.id, BookStatus.PUBLISHED)

        outcome = make_runner(book, None, tmp_path, state).prepare_book(book)

    assert outcome.skipped
    assert not download_directory.exists()
    assert not artifact.exists()


def test_packed_resume_uses_manifest_without_source_requests(book: Book, tmp_path) -> None:
    state_path = tmp_path / "state.sqlite3"
    with StateStore(state_path) as state:
        prepared = make_runner(book, PreparedArchive(book, tmp_path), tmp_path, state).prepare_book(
            book
        )

    with StateStore(state_path) as state:
        resumed = make_runner(book, None, tmp_path, state).prepare_book(book)

    assert resumed.artifact == prepared.artifact
    assert resumed.artifact is not None
    assert resumed.artifact.path.exists()


def test_invalid_packed_manifest_is_rebuilt(book: Book, tmp_path) -> None:
    state_path = tmp_path / "state.sqlite3"
    with StateStore(state_path) as state:
        first = make_runner(book, PreparedArchive(book, tmp_path), tmp_path, state).prepare_book(
            book
        )
        assert first.artifact is not None
        first.artifact.path.write_bytes(b"corrupt")

        rebuilt = make_runner(book, PreparedArchive(book, tmp_path), tmp_path, state).prepare_book(
            book
        )
        checkpoint = state.get(book.id)

    assert rebuilt.artifact is not None
    assert rebuilt.artifact.path.stat().st_size == rebuilt.artifact.size
    assert checkpoint is not None
    assert checkpoint.status == BookStatus.PACKED


def test_hub_is_authoritative_over_local_published_state(book: Book, tmp_path) -> None:
    publisher = FakePublisher(current=False)
    with StateStore(tmp_path / "state.sqlite3") as state:
        state.discover(book)
        state.transition(book.id, BookStatus.PUBLISHED, published_revision="old-revision")

        outcome = make_runner(
            book,
            PreparedArchive(book, tmp_path),
            tmp_path,
            state,
            publisher=publisher,
        ).prepare_book(book)

    assert outcome.artifact is not None


def test_current_hub_book_cleans_local_checkpoint_without_source_requests(
    book: Book, tmp_path
) -> None:
    artifact = tmp_path / "staging/repository/data/000/000047.tar"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"staged")
    with StateStore(tmp_path / "state.sqlite3") as state:
        state.discover(book)
        state.transition(book.id, BookStatus.PACKED, artifact_path=artifact)

        outcome = make_runner(
            book,
            None,
            tmp_path,
            state,
            publisher=FakePublisher(current=True),
        ).prepare_book(book)
        checkpoint = state.get(book.id)

    assert outcome.skipped
    assert checkpoint is not None
    assert checkpoint.status == BookStatus.PUBLISHED
    assert not artifact.exists()


def test_retry_quarantined_reprocesses_a_current_quarantine(book: Book, tmp_path) -> None:
    publisher = QuarantinedPublisher()
    with StateStore(tmp_path / "state.sqlite3") as state:
        outcome = make_runner(
            book,
            PreparedArchive(book, tmp_path),
            tmp_path,
            state,
            publisher=publisher,
            retry_quarantined=True,
        ).prepare_book(book)

    assert outcome.artifact is not None
    assert not outcome.skipped


def test_publish_removes_packed_checkpoint_files(book: Book, tmp_path) -> None:
    publisher = FakePublisher()
    with StateStore(tmp_path / "state.sqlite3") as state:
        runner = make_runner(
            book,
            PreparedArchive(book, tmp_path),
            tmp_path,
            state,
            publisher=publisher,
        )
        outcome = runner.prepare_book(book)
        assert outcome.artifact is not None
        manifest = tmp_path / "staging/manifests/000047.json"

        result = runner.publish([outcome], SyncState(), commit_message="test")

    assert result is not None
    assert not outcome.artifact.path.exists()
    assert not manifest.exists()


def test_publish_releases_packed_staging_capacity(book: Book, tmp_path) -> None:
    publisher = FakePublisher()
    capacity = StagingCapacity(
        tmp_path / "staging",
        max_bytes=1024**2,
        minimum_free_bytes=0,
    )
    with StateStore(tmp_path / "state.sqlite3") as state:
        runner = make_runner(
            book,
            PreparedArchive(book, tmp_path),
            tmp_path,
            state,
            publisher=publisher,
            staging_capacity=capacity,
        )
        outcome = runner.prepare_book(book)
        assert outcome.artifact is not None
        assert capacity.reserved_bytes == outcome.artifact.size

        runner.publish([outcome], SyncState(), commit_message="test")

    assert capacity.reserved_bytes == 0


def test_prepare_failure_is_visible_in_persistent_state(book: Book, tmp_path) -> None:
    with StateStore(tmp_path / "state.sqlite3") as state:
        with pytest.raises(RuntimeError, match="connection lost"):
            make_runner(book, FailingArchive(), tmp_path, state).prepare_book(book)
        checkpoint = state.get(book.id)

    assert checkpoint is not None
    assert checkpoint.attempt_count == 1
    assert checkpoint.last_error == "RuntimeError: connection lost"


def test_resilient_preparation_defers_source_failures(book: Book, tmp_path) -> None:
    with StateStore(tmp_path / "state.sqlite3") as state:
        outcome = make_runner(book, UnavailableArchive(), tmp_path, state).prepare_book_resiliently(
            book
        )
        checkpoint = state.get(book.id)

    assert outcome.error == "SourceUnavailableError: archive edge unavailable"
    assert checkpoint is not None
    assert checkpoint.last_error == outcome.error


def test_resilient_preparation_retries_artifact_build_failures(
    book: Book, tmp_path, monkeypatch
) -> None:
    from librivox_mirror.artifact import build_artifact

    attempts = 0

    def build_with_transient_failure(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ArtifactBuildError("unexpected end of data")
        return build_artifact(*args, **kwargs)

    monkeypatch.setattr("librivox_mirror.workflow.build_artifact", build_with_transient_failure)
    with StateStore(tmp_path / "state.sqlite3") as state:
        outcome = make_runner(
            book, PreparedArchive(book, tmp_path), tmp_path, state
        ).prepare_book_resiliently(book)
        checkpoint = state.get(book.id)

    assert outcome.artifact is not None
    assert attempts == 2
    assert checkpoint is not None
    assert checkpoint.attempt_count == 2


@pytest.mark.parametrize(
    "intended_state",
    [
        SyncState(catalog_scan_started_at=100, catalog_scan_after_book_id=47),
        SyncState(catalog_watermark=100),
    ],
    ids=["catchup-cursor", "completed-watermark"],
)
def test_publish_retry_preserves_checkpoint_and_refreshes_totals(
    book: Book, tmp_path, monkeypatch, intended_state: SyncState
) -> None:
    publisher = TransientPublisher()
    monkeypatch.setattr(
        publisher,
        "load_sync_state",
        lambda: SyncState(
            catalog_scan_started_at=90,
            catalog_scan_after_book_id=40,
            published_books=123,
            published_sections=456,
            quarantined_books=7,
            audio_seconds_by_language={"English": 3600},
        ),
    )
    monkeypatch.setattr("librivox_mirror.workflow.time.sleep", lambda _: None)
    with StateStore(tmp_path / "state.sqlite3") as state:
        runner = make_runner(
            book,
            PreparedArchive(book, tmp_path),
            tmp_path,
            state,
            publisher=publisher,
        )
        outcome = runner.prepare_book(book)
        result = runner.publish([outcome], intended_state, commit_message="test")

    assert result is not None
    assert publisher.attempts == 2
    assert result.state.catalog_watermark == intended_state.catalog_watermark
    assert result.state.catalog_scan_started_at == intended_state.catalog_scan_started_at
    assert result.state.catalog_scan_after_book_id == intended_state.catalog_scan_after_book_id
    assert result.state.published_books == 123
    assert result.state.published_sections == 456
    assert result.state.quarantined_books == 7
    assert result.state.audio_seconds_by_language == {"English": 3600}


def test_publish_recovers_an_ambiguous_success_without_duplicate_commit(
    book: Book, tmp_path, monkeypatch
) -> None:
    publisher = AmbiguousPublisher()
    monkeypatch.setattr("librivox_mirror.workflow.time.sleep", lambda _: None)
    with StateStore(tmp_path / "state.sqlite3") as state:
        runner = make_runner(
            book,
            PreparedArchive(book, tmp_path),
            tmp_path,
            state,
            publisher=publisher,
        )
        outcome = runner.prepare_book(book)
        result = runner.publish([outcome], SyncState(), commit_message="test")
        checkpoint = state.get(book.id)

    assert result is not None
    assert result.revision == "committed-revision"
    assert result.state.published_books == 1
    assert publisher.attempts == 1
    assert checkpoint is not None
    assert checkpoint.status == BookStatus.PUBLISHED
