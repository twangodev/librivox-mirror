import hashlib
from typing import cast

from librivox_mirror.archive import (
    InternetArchiveClient,
    QuarantinedBookError,
    resolve_original_files,
)
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

    def download_book(self, resolved, destination, *, jobs):
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


def make_runner(
    book: Book,
    archive,
    tmp_path,
    state: StateStore,
    *,
    publisher=None,
) -> MirrorRunner:
    return MirrorRunner(
        catalog=cast(LibriVoxCatalog, None),
        archive=cast(InternetArchiveClient, archive),
        state=state,
        staging_directory=tmp_path / "staging",
        jobs=2,
        publisher=publisher,
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
