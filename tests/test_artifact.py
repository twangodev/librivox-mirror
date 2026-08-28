import hashlib
import json
import tarfile

from librivox_mirror.archive import resolve_original_files
from librivox_mirror.artifact import build_artifact, verify_artifact
from librivox_mirror.models import Book, DownloadedSection


def mp3_bytes() -> bytes:
    return b"".join(b"\xff\xfb\x90\x64" + bytes(413) for _ in range(20))


def test_artifact_is_byte_deterministic_and_self_describing(book: Book, tmp_path) -> None:
    content = mp3_bytes()
    source_rows = [
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
        },
    ]
    resolved = resolve_original_files(book, "a_test_book", source_rows)
    audio_path = tmp_path / "section.mp3"
    audio_path.write_bytes(content)
    download = DownloadedSection(
        resolved=resolved.sections[0],
        path=audio_path,
        sha256=hashlib.sha256(content).hexdigest(),
    )
    packing_progress = []

    first = build_artifact(
        resolved,
        [download],
        tmp_path / "first",
        progress=lambda completed, total: packing_progress.append((completed, total)),
    )
    second = build_artifact(resolved, [download], tmp_path / "second")

    assert packing_progress[0] == (0, len(content))
    assert packing_progress[-1] == (len(content), len(content))
    assert first.sha256 == second.sha256
    assert first.path.read_bytes() == second.path.read_bytes()
    assert verify_artifact(first.path, first.sha256) == (first.sha256, 1)
    with tarfile.open(first.path) as archive:
        metadata_file = archive.extractfile("000047-00000091.json")
        assert metadata_file is not None
        metadata = json.load(metadata_file)
    assert metadata["book_id"] == 47
    assert metadata["librivox_metadata"]["future_section_field"]["preserved"] is True
