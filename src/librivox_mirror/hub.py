from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationDelete,
    HfApi,
    hf_hub_download,
)
from huggingface_hub.errors import (
    EntryNotFoundError,
    RemoteEntryNotFoundError,
    RepositoryNotFoundError,
)

from librivox_mirror.models import (
    Book,
    BookArtifact,
    QuarantineRecord,
    SyncState,
    canonical_metadata_json,
)

BOOK_SCHEMA = pa.schema(
    [
        pa.field("book_id", pa.int64(), nullable=False),
        pa.field("title", pa.string(), nullable=False),
        pa.field("description", pa.string(), nullable=False),
        pa.field("language", pa.string()),
        pa.field("copyright_year", pa.string()),
        pa.field("total_time_seconds", pa.int64()),
        pa.field("section_count", pa.int32(), nullable=False),
        pa.field("hash_partition", pa.int16(), nullable=False),
        pa.field("authors_json", pa.string(), nullable=False),
        pa.field("url_librivox", pa.string(), nullable=False),
        pa.field("url_iarchive", pa.string(), nullable=False),
        pa.field("url_project", pa.string()),
        pa.field("url_rss", pa.string()),
        pa.field("url_text_source", pa.string()),
        pa.field("archive_identifier", pa.string(), nullable=False),
        pa.field("source_fingerprint", pa.string(), nullable=False),
        pa.field("tar_path", pa.string(), nullable=False),
        pa.field("tar_size", pa.int64(), nullable=False),
        pa.field("tar_sha256", pa.string(), nullable=False),
        pa.field("librivox_metadata_json", pa.string(), nullable=False),
        pa.field("archive_metadata_json", pa.string(), nullable=False),
    ]
)

SECTION_SCHEMA = pa.schema(
    [
        pa.field("book_id", pa.int64(), nullable=False),
        pa.field("section_id", pa.int64(), nullable=False),
        pa.field("section_number", pa.int32(), nullable=False),
        pa.field("title", pa.string(), nullable=False),
        pa.field("language", pa.string()),
        pa.field("duration_seconds", pa.int64()),
        pa.field("readers_json", pa.string(), nullable=False),
        pa.field("hash_partition", pa.int16(), nullable=False),
        pa.field("sample_key", pa.string(), nullable=False),
        pa.field("tar_path", pa.string(), nullable=False),
        pa.field("librivox_listen_url", pa.string(), nullable=False),
        pa.field("archive_identifier", pa.string(), nullable=False),
        pa.field("archive_file", pa.string(), nullable=False),
        pa.field("source_url", pa.string(), nullable=False),
        pa.field("source_size", pa.int64(), nullable=False),
        pa.field("source_md5", pa.string()),
        pa.field("source_sha1", pa.string()),
        pa.field("mirror_sha256", pa.string(), nullable=False),
        pa.field("librivox_metadata_json", pa.string(), nullable=False),
        pa.field("archive_file_metadata_json", pa.string(), nullable=False),
    ]
)

QUARANTINE_SCHEMA = pa.schema(
    [
        pa.field("book_id", pa.int64(), nullable=False),
        pa.field("title", pa.string(), nullable=False),
        pa.field("code", pa.string(), nullable=False),
        pa.field("detail", pa.string(), nullable=False),
        pa.field("archive_identifier", pa.string()),
        pa.field("source_fingerprint", pa.string(), nullable=False),
        pa.field("observed_at", pa.string(), nullable=False),
        pa.field("librivox_metadata_json", pa.string(), nullable=False),
    ]
)

MISSING_REMOTE = (EntryNotFoundError, RemoteEntryNotFoundError, RepositoryNotFoundError)


@dataclass(frozen=True)
class QuarantineUpdate:
    book: Book
    record: QuarantineRecord


@dataclass(frozen=True)
class PublishResult:
    revision: str
    state: SyncState


class HubPublisher:
    def __init__(
        self,
        repo_id: str,
        *,
        token: str | None,
        working_directory: Path,
        api: HfApi | None = None,
    ) -> None:
        self.repo_id = repo_id
        self.token = token
        self.working_directory = working_directory
        self.cache_directory = working_directory / "cache"
        self.output_directory = working_directory / "metadata"
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.api = api or HfApi(token=token)
        self._rows_cache: dict[str, list[dict[str, Any]]] = {}

    def ensure_repo(self) -> None:
        self.api.create_repo(
            self.repo_id,
            repo_type="dataset",
            private=False,
            exist_ok=True,
        )

    def load_sync_state(self) -> SyncState:
        try:
            path = self._download("state/sync.json")
        except MISSING_REMOTE:
            return SyncState()
        return SyncState.model_validate_json(path.read_text())

    def has_current_book(self, book: Book) -> bool:
        bucket = book.metadata_bucket
        books = self._load_rows(metadata_path("books", bucket), BOOK_SCHEMA)
        if any(
            row["book_id"] == book.id and row["source_fingerprint"] == book.source_fingerprint
            for row in books
        ):
            return True
        quarantined = self._load_rows(metadata_path("quarantine", bucket), QUARANTINE_SCHEMA)
        return any(
            row["book_id"] == book.id and row["source_fingerprint"] == book.source_fingerprint
            for row in quarantined
        )

    def publish(
        self,
        artifacts: list[BookArtifact],
        quarantines: list[QuarantineUpdate],
        sync_state: SyncState,
        *,
        commit_message: str,
    ) -> PublishResult:
        if not artifacts and not quarantines:
            raise ValueError("a Hub commit requires at least one artifact or quarantine update")
        self.ensure_repo()
        affected_ids = {artifact.book.id for artifact in artifacts}
        affected_ids.update(update.book.id for update in quarantines)
        if len(affected_ids) != len(artifacts) + len(quarantines):
            raise ValueError("a book cannot be published and quarantined in the same commit")

        updated_state = sync_state
        metadata_additions: list[CommitOperationAdd] = []
        for bucket in sorted({book_id // 1000 for book_id in affected_ids}):
            bucket_artifacts = [
                artifact for artifact in artifacts if artifact.book.id // 1000 == bucket
            ]
            bucket_quarantines = [
                update for update in quarantines if update.book.id // 1000 == bucket
            ]
            additions, state_delta = self._update_bucket(
                bucket,
                bucket_artifacts,
                bucket_quarantines,
            )
            metadata_additions.extend(additions)
            updated_state = updated_state.model_copy(
                update={
                    "published_books": updated_state.published_books + state_delta.published_books,
                    "published_sections": updated_state.published_sections
                    + state_delta.published_sections,
                    "quarantined_books": updated_state.quarantined_books
                    + state_delta.quarantined_books,
                }
            )

        updated_state = updated_state.model_copy(update={"updated_at": datetime.now(UTC)})
        state_path = self.output_directory / "sync.json"
        state_path.write_text(canonical_metadata_json(updated_state.model_dump(mode="json")) + "\n")
        card_path = self.output_directory / "README.md"
        card_path.write_text(dataset_card(updated_state, self.repo_id))

        additions = [
            CommitOperationAdd(
                path_in_repo=artifact_repo_path(artifact.book.id),
                path_or_fileobj=artifact.path,
            )
            for artifact in artifacts
        ]
        additions.extend(metadata_additions)
        additions.extend(
            [
                CommitOperationAdd(path_in_repo="state/sync.json", path_or_fileobj=state_path),
                CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card_path),
            ]
        )
        deletions = [
            CommitOperationDelete(path_in_repo=artifact_repo_path(update.book.id))
            for update in quarantines
            if self.api.file_exists(
                self.repo_id,
                artifact_repo_path(update.book.id),
                repo_type="dataset",
            )
        ]
        parent_commit = self.api.repo_info(self.repo_id, repo_type="dataset").sha
        self.api.preupload_lfs_files(
            self.repo_id,
            additions,
            repo_type="dataset",
            num_threads=4,
        )
        commit = self.api.create_commit(
            self.repo_id,
            operations=[*additions, *deletions],
            commit_message=commit_message,
            repo_type="dataset",
            parent_commit=parent_commit,
            num_threads=4,
        )
        return PublishResult(revision=commit.oid, state=updated_state)

    def _update_bucket(
        self,
        bucket: int,
        artifacts: list[BookArtifact],
        quarantines: list[QuarantineUpdate],
    ) -> tuple[list[CommitOperationAdd], SyncState]:
        books_path = metadata_path("books", bucket)
        sections_path = metadata_path("sections", bucket)
        quarantine_path = metadata_path("quarantine", bucket)
        old_books = self._load_rows(books_path, BOOK_SCHEMA)
        old_sections = self._load_rows(sections_path, SECTION_SCHEMA)
        old_quarantine = self._load_rows(quarantine_path, QUARANTINE_SCHEMA)
        updated_ids = {artifact.book.id for artifact in artifacts}
        updated_ids.update(update.book.id for update in quarantines)

        books = [row for row in old_books if row["book_id"] not in updated_ids]
        sections = [row for row in old_sections if row["book_id"] not in updated_ids]
        quarantine_rows = [row for row in old_quarantine if row["book_id"] not in updated_ids]
        for artifact in artifacts:
            books.append(book_row(artifact))
            sections.extend(section_rows(artifact))
        quarantine_rows.extend(quarantine_row(update) for update in quarantines)

        books.sort(key=lambda row: row["book_id"])
        sections.sort(key=lambda row: (row["book_id"], row["section_id"]))
        quarantine_rows.sort(key=lambda row: row["book_id"])
        self._rows_cache[books_path] = books
        self._rows_cache[sections_path] = sections
        self._rows_cache[quarantine_path] = quarantine_rows

        additions = [
            self._parquet_addition(books_path, BOOK_SCHEMA, books),
            self._parquet_addition(sections_path, SECTION_SCHEMA, sections),
            self._parquet_addition(quarantine_path, QUARANTINE_SCHEMA, quarantine_rows),
        ]
        delta = SyncState(
            published_books=len(books) - len(old_books),
            published_sections=len(sections) - len(old_sections),
            quarantined_books=len(quarantine_rows) - len(old_quarantine),
        )
        return additions, delta

    def _parquet_addition(
        self,
        path_in_repo: str,
        schema: pa.Schema,
        rows: list[dict[str, Any]],
    ) -> CommitOperationAdd:
        destination = self.output_directory / path_in_repo
        destination.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(
            table,
            destination,
            version="2.6",
            compression="zstd",
            compression_level=9,
            use_dictionary=True,
            write_statistics=True,
            use_content_defined_chunking=True,
        )
        return CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=destination)

    def _load_rows(self, path_in_repo: str, schema: pa.Schema) -> list[dict[str, Any]]:
        if path_in_repo in self._rows_cache:
            return [dict(row) for row in self._rows_cache[path_in_repo]]
        try:
            path = self._download(path_in_repo)
        except MISSING_REMOTE:
            rows: list[dict[str, Any]] = []
        else:
            rows = pq.read_table(path, schema=schema).to_pylist()
        self._rows_cache[path_in_repo] = rows
        return [dict(row) for row in rows]

    def _download(self, path_in_repo: str) -> Path:
        downloaded = hf_hub_download(
            self.repo_id,
            path_in_repo,
            repo_type="dataset",
            token=self.token,
            cache_dir=self.cache_directory,
        )
        return Path(downloaded)


def metadata_path(kind: str, bucket: int) -> str:
    return f"metadata/{kind}/{bucket:03d}.parquet"


def artifact_repo_path(book_id: int) -> str:
    return f"data/{book_id // 1000:03d}/{book_id:06d}.tar"


def book_row(artifact: BookArtifact) -> dict[str, object]:
    book = artifact.book
    return {
        "book_id": book.id,
        "title": book.title,
        "description": book.description,
        "language": book.language,
        "copyright_year": book.copyright_year,
        "total_time_seconds": book.total_time_seconds,
        "section_count": len(book.sections),
        "hash_partition": book.hash_partition,
        "authors_json": canonical_metadata_json(
            [author.model_dump(mode="json") for author in book.authors]
        ),
        "url_librivox": book.url_librivox,
        "url_iarchive": book.url_iarchive,
        "url_project": book.url_project,
        "url_rss": book.url_rss,
        "url_text_source": book.url_text_source,
        "archive_identifier": artifact.archive_identifier,
        "source_fingerprint": book.source_fingerprint,
        "tar_path": artifact_repo_path(book.id),
        "tar_size": artifact.size,
        "tar_sha256": artifact.sha256,
        "librivox_metadata_json": book.source_metadata_json,
        "archive_metadata_json": artifact.archive_metadata_json,
    }


def section_rows(artifact: BookArtifact) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for download in artifact.sections:
        section = download.resolved.section
        source = download.resolved.archive_file
        rows.append(
            {
                "book_id": artifact.book.id,
                "section_id": section.id,
                "section_number": section.section_number,
                "title": section.title,
                "language": section.language or artifact.book.language,
                "duration_seconds": section.duration_seconds,
                "readers_json": canonical_metadata_json(
                    [reader.model_dump(mode="json") for reader in section.readers]
                ),
                "hash_partition": artifact.book.hash_partition,
                "sample_key": section.sample_key,
                "tar_path": artifact_repo_path(artifact.book.id),
                "librivox_listen_url": section.listen_url,
                "archive_identifier": artifact.archive_identifier,
                "archive_file": source.name,
                "source_url": (
                    "https://archive.org/download/"
                    f"{quote(artifact.archive_identifier, safe='')}/{quote(source.name, safe='/')}"
                ),
                "source_size": source.size,
                "source_md5": source.md5,
                "source_sha1": source.sha1,
                "mirror_sha256": download.sha256,
                "librivox_metadata_json": section.source_metadata_json,
                "archive_file_metadata_json": source.source_metadata_json,
            }
        )
    return rows


def quarantine_row(update: QuarantineUpdate) -> dict[str, object]:
    record = update.record
    return {
        "book_id": record.book_id,
        "title": record.title,
        "code": record.code,
        "detail": record.detail,
        "archive_identifier": record.archive_identifier,
        "source_fingerprint": record.source_fingerprint,
        "observed_at": record.observed_at.isoformat(),
        "librivox_metadata_json": update.book.source_metadata_json,
    }


def dataset_card(state: SyncState, repo_id: str) -> str:
    updated_at = (
        state.updated_at.isoformat().replace("+00:00", "Z") if state.updated_at else "not yet"
    )
    return f"""---
pretty_name: LibriVox Mirror
language: multilingual
license: cc-by-4.0
task_categories:
- automatic-speech-recognition
- text-to-speech
tags:
- audio
- librivox
- public-domain
- webdataset
- datasets
configs:
- config_name: default
  data_files:
  - split: train
    path: data/**/*.tar
- config_name: books
  data_files:
  - split: train
    path: metadata/books/*.parquet
- config_name: sections
  data_files:
  - split: train
    path: metadata/sections/*.parquet
---

# LibriVox Mirror

An ML-ready, continuously updated mirror of the LibriVox catalog, maintained by
James Ding. Original MP3 bytes are stored as one deterministic WebDataset TAR per
book. Compact Parquet indexes expose book and section metadata without duplicating
the audio.

## Current snapshot

| Metric | Value |
| --- | ---: |
| Published books | {state.published_books:,} |
| Published sections | {state.published_sections:,} |
| Quarantined books | {state.quarantined_books:,} |
| Last updated (UTC) | `{updated_at}` |

The timestamp records the most recent successful dataset commit. No-change update
runs do not rewrite the card.

## Dataset structure

All audio belongs to the `train` split. Use `language` and `hash_partition` from the
Parquet indexes to construct stable downstream subsets or evaluation splits.

The normalized columns cover common ML queries. The `*_metadata_json` columns retain
the complete source records from LibriVox and Internet Archive, including fields not
currently understood by the mirror.

## License and attribution

The copyrightable mirror-specific compilation, curation, normalized metadata,
indexes, and documentation are made available by James Ding under the [Creative
Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/).
When using those parts of the dataset, provide attribution to James Ding and cite
this dataset.

Suggested attribution:

> LibriVox Mirror by James Ding, licensed under CC BY 4.0. Original LibriVox audio
> is public domain in the United States and is not relicensed by the mirror.

The CC BY 4.0 license does not apply to elements already in the public domain and
does not impose new restrictions on them. LibriVox states that its recordings are
in the public domain in the United States. Copyright and public-domain rules differ
by jurisdiction; downstream users are responsible for checking the rules that
apply to them.

## Citation

```bibtex
@misc{{ding2026librivoxmirror,
  author       = {{Ding, James}},
  title        = {{LibriVox Mirror: An ML-Ready Mirror of the LibriVox Catalog}},
  year         = {{2026}},
  publisher    = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/datasets/{repo_id}}}}},
  note         = {{Continuously updated dataset}}
}}
```

## Provenance and integrity

Metadata links each sample to its LibriVox project and exact Internet Archive source
file. Upstream checksums and a mirror SHA-256 are preserved. Original audio bytes are
not transcoded. Please also credit the LibriVox readers and source works when
appropriate.
"""
