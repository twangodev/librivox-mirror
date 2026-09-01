import hashlib
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock
from typing import Any, cast

import httpx
import pytest

from librivox_mirror.archive import (
    DOWNLOAD_ATTEMPTS,
    ArchiveItemMissingError,
    DownloadIntegrityError,
    DownloadPool,
    InternetArchiveClient,
    QuarantinedBookError,
    SourceUnavailableError,
    archive_identifier,
    archive_identifiers,
    resolve_original_files,
    verify_download,
)
from librivox_mirror.models import Book, DownloadedSection, QuarantineCode


class StubDownloadClient(InternetArchiveClient):
    def __init__(self, content: bytes, *, integrity_failures: int = 0) -> None:
        self.content = content
        self.integrity_failures = integrity_failures
        self.attempts = 0

    def _download_once(self, identifier, archive_file, partial, *, progress=None):
        self.attempts += 1
        partial.write_bytes(self.content)
        if progress is not None:
            progress(len(self.content))
        if self.attempts <= self.integrity_failures:
            raise DownloadIntegrityError("transient checksum mismatch")
        return hashlib.sha256(self.content).hexdigest()


class StatusDownloadClient(InternetArchiveClient):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.attempts = 0

    def _download_once(self, identifier, archive_file, partial, *, progress=None):
        self.attempts += 1
        request = httpx.Request("GET", "https://archive.org/download/test/chapter.mp3")
        response = httpx.Response(self.status_code, request=request)
        raise httpx.HTTPStatusError("download failed", request=request, response=response)


class PooledStubDownloadClient(StubDownloadClient):
    def __init__(self, content: bytes, *, download_jobs: int) -> None:
        InternetArchiveClient.__init__(
            self,
            user_agent="librivox-mirror-tests",
            download_jobs=download_jobs,
        )
        self.content = content
        self.integrity_failures = 0
        self.attempts = 0


def archive_rows() -> list[dict[str, str]]:
    content = b"original audio"
    return [
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
            "size": str(len(content)),
            "md5": hashlib.md5(content, usedforsecurity=False).hexdigest(),
            "sha1": hashlib.sha1(content, usedforsecurity=False).hexdigest(),
            "future_archive_field": "preserved",
        },
    ]


def test_archive_identifier_accepts_details_and_download_urls() -> None:
    assert archive_identifier("https://archive.org/details/a_book") == "a_book"
    assert archive_identifier("https://archive.org/download/a_book/file.mp3") == "a_book"
    assert archive_identifier("https://archive.org/compress/a_book/formats=MP3") == "a_book"
    assert archive_identifier("") is None


def test_archive_identifiers_fall_back_to_published_section_urls(book: Book) -> None:
    missing = book.model_copy(update={"url_iarchive": "", "url_zip_file": None})

    assert archive_identifiers(missing) == ("a_test_book",)
    assert archive_identifiers(missing.model_copy(update={"url_librivox": ""})) == ()


def test_book_resolution_tries_each_archive_identifier(book: Book) -> None:
    class FallbackClient(InternetArchiveClient):
        def __init__(self) -> None:
            self.requested = []

        def _get_metadata(self, identifier):
            self.requested.append(identifier)
            if identifier != "a_test_book":
                raise ArchiveItemMissingError(identifier)
            return {"files": archive_rows(), "metadata": {"title": "Archive title"}}

    client = FallbackClient()
    fallback_book = book.model_copy(
        update={
            "url_iarchive": "https://archive.org/details/missing_book",
            "url_zip_file": "https://archive.org/compress/dark_book/formats=MP3",
        }
    )

    resolved = client.resolve_book(fallback_book)

    assert resolved.archive_identifier == "a_test_book"
    assert client.requested == ["missing_book", "dark_book", "a_test_book"]


def test_resolve_original_files_follows_derivative_provenance(book: Book) -> None:
    resolved = resolve_original_files(
        book,
        "a_test_book",
        archive_rows(),
        {"title": "Archive title", "future_item_field": True},
    )

    selected = resolved.sections[0].archive_file
    assert selected.name == "chapter.mp3"
    assert json.loads(selected.source_metadata_json)["future_archive_field"] == "preserved"
    assert json.loads(resolved.archive_metadata_json)["future_item_field"] is True


def test_file_name_is_used_when_the_listen_url_is_stale(book: Book) -> None:
    section = book.sections[0].model_copy(update={"file_name": "correct_128kb.mp3"})
    rows = [
        {
            "name": "correct_128kb.mp3",
            "source": "original",
            "format": "VBR MP3",
            "size": "1",
        },
    ]

    resolved = resolve_original_files(book.model_copy(update={"sections": (section,)}), "id", rows)

    assert resolved.sections[0].archive_file.name == "correct_128kb.mp3"


def test_listen_url_provenance_wins_when_catalog_fields_conflict(book: Book) -> None:
    section = book.sections[0].model_copy(update={"file_name": "wrong.mp3"})
    rows = [
        *archive_rows(),
        {"name": "wrong.mp3", "source": "original", "format": "VBR MP3", "size": "1"},
    ]

    resolved = resolve_original_files(book.model_copy(update={"sections": (section,)}), "id", rows)

    assert resolved.sections[0].archive_file.name == "chapter.mp3"


def test_malformed_bitrate_suffix_resolves_uniquely(book: Book) -> None:
    section = book.sections[0].model_copy(update={"file_name": "chapter_128kp.mp3"})
    rows = [
        {
            "name": "chapter_128kb.mp3",
            "source": "original",
            "format": "VBR MP3",
            "size": "1",
        },
    ]

    resolved = resolve_original_files(book.model_copy(update={"sections": (section,)}), "id", rows)

    assert resolved.sections[0].archive_file.name == "chapter_128kb.mp3"


def test_archive_format_can_identify_an_extensionless_original(book: Book) -> None:
    rows = [
        {
            "name": "chapter_64kb.mp3",
            "source": "derivative",
            "format": "64Kbps MP3",
            "original": "chapter_128kb_mp3",
        },
        {
            "name": "chapter_128kb_mp3",
            "source": "original",
            "format": "VBR MP3",
            "size": "1",
        },
    ]

    resolved = resolve_original_files(book, "id", rows)

    assert resolved.sections[0].archive_file.name == "chapter_128kb_mp3"


def test_book_resolution_fetches_metadata_concurrently(book: Book) -> None:
    simultaneous_requests = Barrier(2)
    requested_urls: list[str] = []

    def metadata_response(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        simultaneous_requests.wait(timeout=5)
        return httpx.Response(
            200,
            json={"files": archive_rows(), "metadata": {"title": "Archive title"}},
        )

    with (
        httpx.Client(transport=httpx.MockTransport(metadata_response)) as http_client,
        InternetArchiveClient(
            user_agent="librivox-mirror-tests",
            request_delay=0,
            download_jobs=1,
            client=http_client,
        ) as archive,
        ThreadPoolExecutor(max_workers=2) as workers,
    ):
        resolved = list(workers.map(archive.resolve_book, (book, book)))

    assert [result.archive_identifier for result in resolved] == ["a_test_book", "a_test_book"]
    assert requested_urls == [
        "https://archive.org/metadata/a_test_book",
        "https://archive.org/metadata/a_test_book",
    ]


def test_missing_archive_item_is_retried_then_quarantined(book: Book, monkeypatch) -> None:
    attempts = 0

    def missing_response(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(200, json={})

    retrying = cast(Any, InternetArchiveClient._get_metadata).retry
    monkeypatch.setattr(retrying, "sleep", lambda _: None)
    with (
        httpx.Client(transport=httpx.MockTransport(missing_response)) as http_client,
        InternetArchiveClient(
            user_agent="librivox-mirror-tests",
            request_delay=0,
            download_jobs=1,
            client=http_client,
        ) as archive,
        pytest.raises(QuarantinedBookError) as caught,
    ):
        archive.resolve_book(book)

    assert caught.value.record.code == QuarantineCode.ARCHIVE_ITEM_MISSING
    assert attempts == 5


def test_missing_original_quarantines_the_entire_book(book: Book) -> None:
    with pytest.raises(QuarantinedBookError) as caught:
        resolve_original_files(book, "a_test_book", archive_rows()[:1])

    assert caught.value.record.code == QuarantineCode.ORIGINAL_FILE_MISSING


def test_ambiguous_original_quarantines_the_entire_book(book: Book) -> None:
    rows = [
        {"name": "chapter.mp3", "source": "original", "format": "VBR MP3", "size": "1"},
        {
            "name": "chapter_vbr.mp3",
            "source": "original",
            "format": "VBR MP3",
            "size": "1",
        },
    ]
    with pytest.raises(QuarantinedBookError) as caught:
        resolve_original_files(book, "a_test_book", rows)

    assert caught.value.record.code == QuarantineCode.ORIGINAL_FILE_AMBIGUOUS


def test_verify_download_checks_source_hashes(book: Book, tmp_path) -> None:
    resolved = resolve_original_files(book, "a_test_book", archive_rows())
    path = tmp_path / "chapter.mp3"
    path.write_bytes(b"original audio")

    assert (
        verify_download(path, resolved.sections[0].archive_file)
        == hashlib.sha256(b"original audio").hexdigest()
    )

    path.write_bytes(b"corrupt")
    with pytest.raises(DownloadIntegrityError):
        verify_download(path, resolved.sections[0].archive_file)


def test_corrupt_staged_download_is_replaced(book: Book, tmp_path) -> None:
    resolved = resolve_original_files(book, "a_test_book", archive_rows())
    client = StubDownloadClient(b"original audio")
    path = tmp_path / "000047-00000091.mp3"
    path.write_bytes(b"corrupt")

    downloaded = client.download_section("a_test_book", resolved.sections[0], tmp_path)

    assert downloaded.path.read_bytes() == b"original audio"
    assert client.attempts == 1


def test_book_download_reports_progress(book: Book, tmp_path) -> None:
    resolved = resolve_original_files(book, "a_test_book", archive_rows())
    progress = []

    with PooledStubDownloadClient(b"original audio", download_jobs=12) as client:
        client.download_book(
            resolved,
            tmp_path,
            progress=lambda completed, total: progress.append((completed, total)),
        )

    assert progress == [(0, 14), (0, 14), (14, 14)]


def test_download_pool_schedules_books_fairly(book: Book, tmp_path) -> None:
    resolved = resolve_original_files(book, "a_test_book", archive_rows()).sections[0]
    downloaded = DownloadedSection(resolved=resolved, path=tmp_path / "audio.mp3", sha256="hash")
    first_started = Event()
    release_first = Event()
    order: list[str] = []

    def task(name: str, *, blocks: bool = False) -> Callable[[], DownloadedSection]:
        def run() -> DownloadedSection:
            order.append(name)
            if blocks:
                first_started.set()
                assert release_first.wait(timeout=5)
            return downloaded

        return run

    pool = DownloadPool(workers=1)
    try:
        first = pool.submit([task("a1", blocks=True), task("a2"), task("a3")])
        assert first_started.wait(timeout=5)
        second = pool.submit([task("b1"), task("b2")])
        release_first.set()

        first.result()
        second.result()
    finally:
        pool.close()

    assert order == ["a1", "a2", "b1", "a3", "b2"]


def test_download_pool_enforces_the_global_worker_limit(book: Book, tmp_path) -> None:
    resolved = resolve_original_files(book, "a_test_book", archive_rows()).sections[0]
    downloaded = DownloadedSection(resolved=resolved, path=tmp_path / "audio.mp3", sha256="hash")
    release = Event()
    capacity_reached = Event()
    lock = Lock()
    active = 0
    peak = 0

    def task() -> Callable[[], DownloadedSection]:
        def run() -> DownloadedSection:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                if active == 3:
                    capacity_reached.set()
            assert release.wait(timeout=5)
            with lock:
                active -= 1
            return downloaded

        return run

    pool = DownloadPool(workers=3)
    try:
        batch = pool.submit([task() for _ in range(10)])
        assert capacity_reached.wait(timeout=5)
        release.set()
        batch.result()
    finally:
        pool.close()

    assert peak == 3


def test_download_retries_integrity_failures(book: Book, tmp_path, monkeypatch) -> None:
    resolved = resolve_original_files(book, "a_test_book", archive_rows())
    client = StubDownloadClient(b"original audio", integrity_failures=1)
    monkeypatch.setattr("librivox_mirror.archive.time.sleep", lambda _: None)

    downloaded = client.download_section("a_test_book", resolved.sections[0], tmp_path)

    assert downloaded.path.read_bytes() == b"original audio"
    assert client.attempts == 2
    assert not downloaded.path.with_suffix(".mp3.partial").exists()


def test_download_exhaustion_becomes_a_deferred_source_failure(
    book: Book, tmp_path, monkeypatch
) -> None:
    resolved = resolve_original_files(book, "a_test_book", archive_rows())
    client = StubDownloadClient(b"original audio", integrity_failures=DOWNLOAD_ATTEMPTS)
    monkeypatch.setattr("librivox_mirror.archive.time.sleep", lambda _: None)

    with pytest.raises(SourceUnavailableError, match=f"after {DOWNLOAD_ATTEMPTS} attempt"):
        client.download_section("a_test_book", resolved.sections[0], tmp_path)

    assert client.attempts == DOWNLOAD_ATTEMPTS
    assert not (tmp_path / "000047-00000091.mp3.partial").exists()


def test_non_retryable_download_status_is_deferred_immediately(book: Book, tmp_path) -> None:
    resolved = resolve_original_files(book, "a_test_book", archive_rows())
    client = StatusDownloadClient(404)

    with pytest.raises(SourceUnavailableError, match="after 1 attempt"):
        client.download_section("a_test_book", resolved.sections[0], tmp_path)

    assert client.attempts == 1
