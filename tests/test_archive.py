import hashlib
import json

import httpx
import pytest

from librivox_mirror.archive import (
    DOWNLOAD_ATTEMPTS,
    DownloadIntegrityError,
    InternetArchiveClient,
    QuarantinedBookError,
    SourceUnavailableError,
    archive_identifier,
    resolve_original_files,
    verify_download,
)
from librivox_mirror.models import Book, QuarantineCode


class StubDownloadClient(InternetArchiveClient):
    def __init__(self, content: bytes, *, integrity_failures: int = 0) -> None:
        self.content = content
        self.integrity_failures = integrity_failures
        self.attempts = 0

    def _download_once(self, identifier, archive_file, partial):
        self.attempts += 1
        partial.write_bytes(self.content)
        if self.attempts <= self.integrity_failures:
            raise DownloadIntegrityError("transient checksum mismatch")
        return hashlib.sha256(self.content).hexdigest()


class StatusDownloadClient(InternetArchiveClient):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.attempts = 0

    def _download_once(self, identifier, archive_file, partial):
        self.attempts += 1
        request = httpx.Request("GET", "https://archive.org/download/test/chapter.mp3")
        response = httpx.Response(self.status_code, request=request)
        raise httpx.HTTPStatusError("download failed", request=request, response=response)


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
    assert archive_identifier("") is None


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
