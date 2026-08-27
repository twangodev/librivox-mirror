import hashlib
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast

import httpx
import pyarrow.parquet as pq
from huggingface_hub import DatasetCard, HfApi

from librivox_mirror.archive import resolve_original_files
from librivox_mirror.artifact import build_artifact
from librivox_mirror.hub import HubPublisher, QuarantineUpdate, dataset_card
from librivox_mirror.models import (
    Book,
    DownloadedSection,
    QuarantineCode,
    QuarantineRecord,
    SyncState,
)


class MissingRemotePublisher(HubPublisher):
    def _download(self, path_in_repo):
        from huggingface_hub.errors import RemoteEntryNotFoundError

        raise RemoteEntryNotFoundError(
            f"missing {path_in_repo}",
            response=httpx.Response(
                404,
                request=httpx.Request("GET", "https://huggingface.co/missing"),
            ),
        )


class FakeApi:
    def __init__(self) -> None:
        self.commits = []
        self.preuploads = []
        self.files = set()

    def create_repo(self, *args, **kwargs):
        return SimpleNamespace(repo_id=args[0])

    def repo_info(self, *args, **kwargs):
        return SimpleNamespace(sha="parent")

    def file_exists(self, repo_id, filename, **kwargs):
        return filename in self.files

    def list_repo_files(self, *args, **kwargs):
        return sorted(self.files)

    def preupload_lfs_files(self, repo_id, additions, **kwargs):
        self.preuploads.append(list(additions))

    def create_commit(self, repo_id, operations, **kwargs):
        operations = list(operations)
        self.commits.append((operations, kwargs))
        for operation in operations:
            path = operation.path_in_repo
            if operation.__class__.__name__ == "CommitOperationDelete":
                self.files.discard(path)
            else:
                self.files.add(path)
        return SimpleNamespace(oid=f"revision-{len(self.commits)}")


def make_artifact(book: Book, tmp_path):
    content = b"".join(b"\xff\xfb\x90\x64" + bytes(413) for _ in range(20))
    rows = [
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
            "full_file_metadata": ["preserved"],
        },
    ]
    resolved = resolve_original_files(
        book,
        "a_test_book",
        rows,
        {"full_item_metadata": {"preserved": True}},
    )
    audio = tmp_path / "chapter.mp3"
    audio.write_bytes(content)
    download = DownloadedSection(
        resolved=resolved.sections[0],
        path=audio,
        sha256=hashlib.sha256(content).hexdigest(),
    )
    return build_artifact(resolved, [download], tmp_path / "repository")


def test_dataset_card_documents_snapshot_license_and_citation() -> None:
    content = dataset_card(
        SyncState(
            published_books=12,
            published_sections=345,
            quarantined_books=6,
            audio_seconds_by_language={"English": 36_000, "French": 9_000},
            updated_at=datetime(2026, 8, 27, 18, 30, tzinfo=UTC),
        ),
        "owner/librivox",
    )
    metadata = DatasetCard(content).data.to_dict()

    assert metadata["pretty_name"] == "LibriVox Mirror"
    assert metadata["language"] == "multilingual"
    assert metadata["license"] == "cc-by-4.0"
    assert metadata["dataset_info"][0]["config_name"] == "audio"
    assert metadata["dataset_info"][0]["features"] == [
        {"name": "mp3", "dtype": "audio"},
        {"name": "json", "dtype": "json"},
        {"name": "__key__", "dtype": "string"},
        {"name": "__url__", "dtype": "string"},
    ]
    configs = {config["config_name"]: config for config in metadata["configs"]}
    assert set(configs) == {"audio"}
    assert configs["audio"]["default"] is True
    assert configs["audio"]["data_files"] == [{"split": "train", "path": "data/**/*.tar"}]
    assert "Last updated (UTC) | `2026-08-27T18:30:00Z`" in content
    assert "Audio hours | 12.5" in content
    assert "Audio languages | 2" in content
    assert "| English | 10.0 |" in content
    assert "| French | 2.5 |" in content
    assert "last_updated-2026--08--27T18%3A30%3A00Z" in content
    assert "logo=creativecommons" in content
    assert "@misc{ding2026librivoxmirror" in content
    assert "author       = {James Ding}" in content
    assert "title        = {LibriVox Mirror}" in content
    assert "ML-Ready" not in content
    assert "https://huggingface.co/datasets/owner/librivox" in content
    assert "is not relicensed by this mirror" in content


def test_publish_builds_atomic_dataset_commit(book: Book, tmp_path) -> None:
    api = FakeApi()
    publisher = MissingRemotePublisher(
        "owner/librivox",
        token=None,
        working_directory=tmp_path / "hub",
        api=cast(HfApi, api),
    )
    artifact = make_artifact(book, tmp_path)

    result = publisher.publish(
        [artifact],
        [],
        SyncState(catalog_watermark=123),
        commit_message="feat(data): mirror book 47",
    )

    assert result.revision == "revision-1"
    assert result.state.published_books == 1
    assert result.state.published_sections == 1
    assert result.state.audio_seconds_by_language == {"English": 1}
    paths = {operation.path_in_repo for operation in api.commits[0][0]}
    assert paths == {
        "data/000/000047.tar",
        "metadata/books/000.parquet",
        "metadata/sections/000.parquet",
        "metadata/quarantine/000.parquet",
        "state/sync.json",
        "README.md",
    }
    assert api.commits[0][1]["parent_commit"] == "parent"
    books = pq.read_table(tmp_path / "hub/metadata/metadata/books/000.parquet").to_pylist()
    sections = pq.read_table(tmp_path / "hub/metadata/metadata/sections/000.parquet").to_pylist()
    assert books[0]["librivox_metadata"] == book.source_metadata_json
    assert books[0]["authors"][0]["last_name"] == "Lovelace"
    assert sections[0]["readers"][0]["display_name"] == "Reader"
    assert sections[0]["archive_file_format"] == "VBR MP3"
    assert "full_item_metadata" in books[0]["archive_metadata"]
    assert "full_file_metadata" in sections[0]["archive_file_metadata"]
    card = (tmp_path / "hub/metadata/README.md").read_text()
    assert "Last updated (UTC)" in card
    assert "license: cc-by-4.0" in card
    assert publisher.has_current_book(book)


def test_load_sync_state_migrates_audio_statistics(book: Book, tmp_path, monkeypatch) -> None:
    api = FakeApi()
    source = MissingRemotePublisher(
        "owner/librivox",
        token=None,
        working_directory=tmp_path / "source",
        api=cast(HfApi, api),
    )
    source.publish(
        [make_artifact(book, tmp_path)],
        [],
        SyncState(),
        commit_message="feat(data): mirror book 47",
    )
    legacy_state = tmp_path / "sync.json"
    legacy_state.write_text(
        SyncState(schema_version=1, published_books=1, published_sections=1).model_dump_json(
            exclude={"audio_seconds_by_language"}
        )
    )
    remote_files = {
        "state/sync.json": legacy_state,
        "metadata/sections/000.parquet": (
            tmp_path / "source/metadata/metadata/sections/000.parquet"
        ),
    }
    publisher = HubPublisher(
        "owner/librivox",
        token=None,
        working_directory=tmp_path / "migrated",
        api=cast(HfApi, api),
    )
    monkeypatch.setattr(publisher, "_download", remote_files.__getitem__)

    state = publisher.load_sync_state()

    assert state.schema_version == 2
    assert state.audio_seconds_by_language == {"English": 1}


def test_quarantine_replaces_current_artifact_and_metadata(book: Book, tmp_path) -> None:
    api = FakeApi()
    publisher = MissingRemotePublisher(
        "owner/librivox",
        token=None,
        working_directory=tmp_path / "hub",
        api=cast(HfApi, api),
    )
    first = publisher.publish(
        [make_artifact(book, tmp_path)],
        [],
        SyncState(),
        commit_message="feat(data): mirror book 47",
    )
    record = QuarantineRecord(
        book_id=book.id,
        title=book.title,
        code=QuarantineCode.ORIGINAL_FILE_MISSING,
        detail="missing",
        archive_identifier="a_test_book",
        source_fingerprint=book.source_fingerprint,
    )

    second = publisher.publish(
        [],
        [QuarantineUpdate(book=book, record=record)],
        first.state,
        commit_message="fix(data): quarantine book 47",
    )

    assert second.state.published_books == 0
    assert second.state.published_sections == 0
    assert second.state.quarantined_books == 1
    assert second.state.audio_seconds_by_language == {}
    assert "data/000/000047.tar" not in api.files
    quarantine = pq.read_table(
        tmp_path / "hub/metadata/metadata/quarantine/000.parquet"
    ).to_pylist()
    assert quarantine[0]["code"] == "original_file_missing"
