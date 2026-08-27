from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

import mutagen

from librivox_mirror.models import (
    BookArtifact,
    DownloadedSection,
    ResolvedBook,
    canonical_metadata_json,
)


class InvalidAudioError(ValueError):
    pass


class InvalidArtifactError(ValueError):
    pass


def artifact_path(root: Path, book_id: int) -> Path:
    return root / "data" / f"{book_id // 1000:03d}" / f"{book_id:06d}.tar"


def artifact_manifest_path(root: Path, book_id: int) -> Path:
    return root / "manifests" / f"{book_id:06d}.json"


def build_artifact(
    resolved: ResolvedBook,
    downloads: Iterable[DownloadedSection],
    root: Path,
) -> BookArtifact:
    ordered = tuple(sorted(downloads, key=lambda item: item.resolved.section.id))
    expected_ids = {section.section.id for section in resolved.sections}
    actual_ids = {download.resolved.section.id for download in ordered}
    if actual_ids != expected_ids:
        raise InvalidArtifactError(
            f"downloaded section IDs do not match resolved sections: {actual_ids} != {expected_ids}"
        )
    for download in ordered:
        verify_mp3(download.path)

    destination = artifact_path(root, resolved.book.id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".tar.partial")
    with tarfile.open(partial, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for download in ordered:
            key = download.resolved.section.sample_key
            add_path(archive, f"{key}.mp3", download.path)
            metadata = section_metadata(resolved, download)
            add_bytes(archive, f"{key}.json", canonical_json(metadata))
    sync_file(partial)
    partial.replace(destination)
    sync_directory(destination.parent)
    sha256 = file_sha256(destination)
    return BookArtifact(
        book=resolved.book,
        archive_identifier=resolved.archive_identifier,
        path=destination,
        sha256=sha256,
        size=destination.stat().st_size,
        sections=ordered,
        archive_metadata_json=resolved.archive_metadata_json,
    )


def write_artifact_manifest(artifact: BookArtifact, path: Path) -> None:
    payload = {
        "schema_version": 1,
        "artifact": artifact.model_dump(mode="json"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".json.partial")
    partial.write_text(canonical_metadata_json(payload) + "\n", encoding="utf-8")
    sync_file(partial)
    partial.replace(path)
    sync_directory(path.parent)


def load_artifact_manifest(path: Path) -> BookArtifact:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["schema_version"] != 1:
            raise InvalidArtifactError(
                f"unsupported artifact manifest schema {payload['schema_version']!r}"
            )
        return BookArtifact.model_validate(payload["artifact"])
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, InvalidArtifactError):
            raise
        raise InvalidArtifactError(f"invalid artifact manifest {path}") from error


def section_metadata(resolved: ResolvedBook, download: DownloadedSection) -> dict[str, object]:
    section = download.resolved.section
    source = download.resolved.archive_file
    return {
        "book_id": resolved.book.id,
        "section_id": section.id,
        "section_number": section.section_number,
        "title": section.title,
        "language": section.language or resolved.book.language,
        "duration_seconds": section.duration_seconds,
        "readers": [reader.model_dump(mode="json") for reader in section.readers],
        "librivox_metadata": json.loads(section.source_metadata_json),
        "hash_partition": resolved.book.hash_partition,
        "source": {
            "archive_identifier": resolved.archive_identifier,
            "file": source.name,
            "size": source.size,
            "md5": source.md5,
            "sha1": source.sha1,
            "url": (f"https://archive.org/download/{resolved.archive_identifier}/{source.name}"),
            "metadata": json.loads(source.source_metadata_json),
        },
        "mirror_sha256": download.sha256,
    }


def verify_mp3(path: Path) -> None:
    audio = mutagen.File(path)
    if audio is None or not hasattr(audio, "info"):
        raise InvalidAudioError(f"{path} is not a readable MP3")
    length = getattr(audio.info, "length", 0)
    if length <= 0:
        raise InvalidAudioError(f"{path} has no decodable audio frames")


def verify_artifact(path: Path, expected_sha256: str | None = None) -> tuple[str, int]:
    sha256 = file_sha256(path)
    if expected_sha256 and sha256 != expected_sha256:
        raise InvalidArtifactError(f"sha256 {sha256} != {expected_sha256}")
    stems: dict[str, set[str]] = {}
    with tarfile.open(path, mode="r:") as archive:
        for member in archive:
            member_path = PurePosixPath(member.name)
            if member_path.is_absolute() or ".." in member_path.parts or not member.isfile():
                raise InvalidArtifactError(f"unsafe TAR member {member.name!r}")
            suffix = member_path.suffix
            if suffix not in {".mp3", ".json"}:
                raise InvalidArtifactError(f"unexpected TAR member {member.name!r}")
            stems.setdefault(member_path.stem, set()).add(suffix)
            if suffix == ".json":
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise InvalidArtifactError(f"could not read {member.name!r}")
                json.load(extracted)
    incomplete = sorted(stem for stem, suffixes in stems.items() if suffixes != {".mp3", ".json"})
    if incomplete:
        raise InvalidArtifactError(f"samples are missing paired members: {incomplete}")
    return sha256, len(stems)


def add_path(archive: tarfile.TarFile, name: str, path: Path) -> None:
    info = tar_info(name, path.stat().st_size)
    with path.open("rb") as source:
        archive.addfile(info, source)


def add_bytes(archive: tarfile.TarFile, name: str, content: bytes) -> None:
    archive.addfile(tar_info(name, len(content)), io.BytesIO(content))


def tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def canonical_json(value: object) -> bytes:
    return canonical_metadata_json(value).encode()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sync_file(path: Path) -> None:
    with path.open("rb") as file:
        os.fsync(file.fileno())


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
