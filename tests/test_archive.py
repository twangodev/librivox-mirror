import hashlib
import json

import pytest

from librivox_mirror.archive import (
    DownloadIntegrityError,
    QuarantinedBookError,
    archive_identifier,
    resolve_original_files,
    verify_download,
)
from librivox_mirror.models import Book, QuarantineCode


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
