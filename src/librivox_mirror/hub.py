from __future__ import annotations

import json
import shutil
import tarfile
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from string import Template
from threading import RLock
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
)

AUTHOR_TYPE = pa.struct(
    [
        pa.field("id", pa.int64()),
        pa.field("first_name", pa.string(), nullable=False),
        pa.field("last_name", pa.string(), nullable=False),
        pa.field("dob", pa.string()),
        pa.field("dod", pa.string()),
    ]
)

GENRE_TYPE = pa.struct(
    [
        pa.field("id", pa.int64()),
        pa.field("name", pa.string(), nullable=False),
    ]
)

READER_TYPE = pa.struct(
    [
        pa.field("id", pa.int64()),
        pa.field("display_name", pa.string(), nullable=False),
        pa.field("url_text", pa.string()),
    ]
)

JSON_TYPE = pa.json_()

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
        pa.field("authors", pa.list_(AUTHOR_TYPE), nullable=False),
        pa.field("translators", pa.list_(AUTHOR_TYPE), nullable=False),
        pa.field("genres", pa.list_(GENRE_TYPE), nullable=False),
        pa.field("url_librivox", pa.string(), nullable=False),
        pa.field("url_iarchive", pa.string(), nullable=False),
        pa.field("url_project", pa.string()),
        pa.field("url_rss", pa.string()),
        pa.field("url_text_source", pa.string()),
        pa.field("url_other", pa.string()),
        pa.field("url_zip_file", pa.string()),
        pa.field("archive_identifier", pa.string(), nullable=False),
        pa.field("source_fingerprint", pa.string(), nullable=False),
        pa.field("tar_path", pa.string(), nullable=False),
        pa.field("tar_size", pa.int64(), nullable=False),
        pa.field("tar_sha256", pa.string(), nullable=False),
        pa.field("librivox_metadata", JSON_TYPE, nullable=False),
        pa.field("archive_metadata", JSON_TYPE, nullable=False),
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
        pa.field("readers", pa.list_(READER_TYPE), nullable=False),
        pa.field("hash_partition", pa.int16(), nullable=False),
        pa.field("sample_key", pa.string(), nullable=False),
        pa.field("tar_path", pa.string(), nullable=False),
        pa.field("librivox_file_name", pa.string()),
        pa.field("librivox_listen_url", pa.string(), nullable=False),
        pa.field("archive_identifier", pa.string(), nullable=False),
        pa.field("archive_file", pa.string(), nullable=False),
        pa.field("archive_file_format", pa.string()),
        pa.field("archive_file_source", pa.string()),
        pa.field("archive_original_file", pa.string()),
        pa.field("source_url", pa.string(), nullable=False),
        pa.field("source_size", pa.int64(), nullable=False),
        pa.field("source_md5", pa.string()),
        pa.field("source_sha1", pa.string()),
        pa.field("mirror_sha256", pa.string(), nullable=False),
        pa.field("librivox_metadata", JSON_TYPE, nullable=False),
        pa.field("archive_file_metadata", JSON_TYPE, nullable=False),
    ]
)

AUDIO_TYPE = pa.struct(
    [
        pa.field("bytes", pa.binary()),
        pa.field("path", pa.string()),
    ]
)


def huggingface_feature(data_type: pa.DataType) -> dict[str, Any]:
    if data_type == JSON_TYPE:
        return {"_type": "Json"}
    if pa.types.is_list(data_type):
        return {"feature": huggingface_feature(data_type.value_type), "_type": "List"}
    if pa.types.is_struct(data_type):
        return {field.name: huggingface_feature(field.type) for field in data_type}
    return {"dtype": str(data_type), "_type": "Value"}


def with_huggingface_features(
    schema: pa.Schema,
    overrides: dict[str, dict[str, Any]],
) -> pa.Schema:
    features = {
        field.name: overrides.get(field.name, huggingface_feature(field.type)) for field in schema
    }
    metadata = {
        **(schema.metadata or {}),
        b"huggingface": json.dumps(
            {"info": {"features": features}},
            separators=(",", ":"),
        ).encode(),
    }
    return schema.with_metadata(metadata)


PREVIEW_SCHEMA = with_huggingface_features(
    pa.schema([pa.field("audio", AUDIO_TYPE), *SECTION_SCHEMA]),
    {"audio": {"_type": "Audio"}},
)
PREVIEW_LIMIT = 32
PREVIEW_METADATA_PATH = "metadata/preview/preview.parquet"

QUARANTINE_SCHEMA = pa.schema(
    [
        pa.field("book_id", pa.int64(), nullable=False),
        pa.field("title", pa.string(), nullable=False),
        pa.field("code", pa.string(), nullable=False),
        pa.field("detail", pa.string(), nullable=False),
        pa.field("archive_identifier", pa.string()),
        pa.field("source_fingerprint", pa.string(), nullable=False),
        pa.field("observed_at", pa.string(), nullable=False),
        pa.field("librivox_metadata", JSON_TYPE, nullable=False),
    ]
)

MISSING_REMOTE = (EntryNotFoundError, RemoteEntryNotFoundError, RepositoryNotFoundError)
_DATASET_CARD_TEMPLATE = Template(
    files("librivox_mirror").joinpath("dataset_card.md").read_text(encoding="utf-8")
)


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
        upload_jobs: int = 4,
    ) -> None:
        if upload_jobs < 1:
            raise ValueError("upload worker count must be positive")
        self.repo_id = repo_id
        self.token = token
        self.working_directory = working_directory
        self.cache_directory = working_directory / "cache"
        self.output_directory = working_directory / "metadata"
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.api = api or HfApi(token=token)
        self.upload_jobs = upload_jobs
        self._cache_lock = RLock()
        self._rows_cache: dict[str, list[dict[str, Any]]] = {}

    def ensure_repo(self) -> None:
        self.api.create_repo(
            self.repo_id,
            repo_type="dataset",
            private=False,
            exist_ok=True,
        )

    def invalidate_cache(self) -> None:
        with self._cache_lock:
            self._rows_cache.clear()

    def current_revision(self) -> str:
        revision = self.api.repo_info(self.repo_id, repo_type="dataset").sha
        if revision is None:
            raise RuntimeError(f"dataset {self.repo_id!r} has no current revision")
        return revision

    def load_sync_state(self) -> SyncState:
        try:
            path = self._download("state/sync.json")
        except MISSING_REMOTE:
            return SyncState()
        state = SyncState.model_validate_json(path.read_text())
        if state.schema_version < 2:
            state = state.model_copy(
                update={
                    "audio_seconds_by_language": self._load_audio_seconds_by_language(),
                }
            )
        if state.schema_version < 3:
            state = state.model_copy(update={"schema_version": 3})
        return state

    def _load_audio_seconds_by_language(self) -> dict[str, int]:
        paths = self.api.list_repo_files(self.repo_id, repo_type="dataset")
        section_paths = sorted(
            path
            for path in paths
            if path.startswith("metadata/sections/") and path.endswith(".parquet")
        )
        sections = [row for path in section_paths for row in self._load_rows(path)]
        return audio_seconds_by_language(sections)

    def has_current_book(self, book: Book, *, include_quarantined: bool = True) -> bool:
        bucket = book.metadata_bucket
        books = self._load_rows(metadata_path("books", bucket))
        if any(
            row["book_id"] == book.id and row["source_fingerprint"] == book.source_fingerprint
            for row in books
        ):
            return True
        if not include_quarantined:
            return False
        quarantined = self._load_rows(metadata_path("quarantine", bucket))
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
                    "audio_seconds_by_language": merge_audio_seconds_by_language(
                        updated_state.audio_seconds_by_language,
                        state_delta.audio_seconds_by_language,
                    ),
                }
            )

        preview_additions, preview_deletions, preview_available = self._update_preview(
            artifacts,
            quarantines,
        )
        updated_state = updated_state.model_copy(update={"updated_at": datetime.now(UTC)})
        state_path = self.output_directory / "sync.json"
        state_path.write_text(
            json.dumps(
                updated_state.model_dump(mode="json"),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )
        card_path = self.output_directory / "README.md"
        card_path.write_text(
            dataset_card(updated_state, self.repo_id, preview_available=preview_available)
        )

        additions = [
            CommitOperationAdd(
                path_in_repo=artifact_repo_path(artifact.book.id),
                path_or_fileobj=artifact.path,
            )
            for artifact in artifacts
        ]
        additions.extend(metadata_additions)
        additions.extend(preview_additions)
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
        deletions.extend(preview_deletions)
        parent_commit = self.api.repo_info(self.repo_id, repo_type="dataset").sha
        self.api.preupload_lfs_files(
            self.repo_id,
            additions,
            repo_type="dataset",
            num_threads=self.upload_jobs,
        )
        commit = self.api.create_commit(
            self.repo_id,
            operations=[*additions, *deletions],
            commit_message=commit_message,
            repo_type="dataset",
            parent_commit=parent_commit,
            num_threads=self.upload_jobs,
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
        old_books = self._load_rows(books_path)
        old_sections = self._load_rows(sections_path)
        old_quarantine = self._load_rows(quarantine_path)
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
            audio_seconds_by_language=audio_seconds_delta(old_sections, sections),
        )
        return additions, delta

    def _update_preview(
        self,
        artifacts: list[BookArtifact],
        quarantines: list[QuarantineUpdate],
    ) -> tuple[list[CommitOperationAdd], list[CommitOperationDelete], bool]:
        old_rows = self._load_rows(PREVIEW_METADATA_PATH)
        affected_ids = {artifact.book.id for artifact in artifacts}
        affected_ids.update(update.book.id for update in quarantines)
        rows = [row for row in old_rows if row["book_id"] not in affected_ids]
        audio_additions: list[CommitOperationAdd] = []

        for artifact in sorted(artifacts, key=lambda item: item.book.id):
            rows_by_key = {row["sample_key"]: row for row in section_rows(artifact)}
            for download in sorted(
                artifact.sections,
                key=lambda item: item.resolved.section.sample_key,
            ):
                if len(rows) >= PREVIEW_LIMIT:
                    break
                sample_key = download.resolved.section.sample_key
                rows.append(preview_row(rows_by_key[sample_key], self.repo_id))
                audio_additions.append(self._preview_audio_addition(artifact, sample_key))

        rows.sort(key=lambda row: row["sample_key"])
        self._rows_cache[PREVIEW_METADATA_PATH] = rows
        old_paths = {preview_audio_path(row["sample_key"]) for row in old_rows}
        new_paths = {preview_audio_path(row["sample_key"]) for row in rows}
        deletion_paths = old_paths - new_paths
        if not rows and self.api.file_exists(
            self.repo_id,
            PREVIEW_METADATA_PATH,
            repo_type="dataset",
        ):
            deletion_paths.add(PREVIEW_METADATA_PATH)
        deletions = [CommitOperationDelete(path_in_repo=path) for path in sorted(deletion_paths)]
        additions = [*audio_additions]
        if rows:
            additions.insert(
                0,
                self._parquet_addition(PREVIEW_METADATA_PATH, PREVIEW_SCHEMA, rows),
            )
        return additions, deletions, bool(rows)

    def _preview_audio_addition(
        self,
        artifact: BookArtifact,
        sample_key: str,
    ) -> CommitOperationAdd:
        path_in_repo = preview_audio_path(sample_key)
        destination = self.output_directory / path_in_repo
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(artifact.path, mode="r:") as archive:
            source = archive.extractfile(f"{sample_key}.mp3")
            if source is None:
                raise ValueError(f"artifact is missing preview audio {sample_key!r}")
            with source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)
        return CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=destination)

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
            write_page_index=True,
            use_content_defined_chunking=True,
            row_group_size=1000,
        )
        return CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=destination)

    def _load_rows(self, path_in_repo: str) -> list[dict[str, Any]]:
        with self._cache_lock:
            if path_in_repo in self._rows_cache:
                return [dict(row) for row in self._rows_cache[path_in_repo]]
            try:
                path = self._download(path_in_repo)
            except MISSING_REMOTE:
                rows: list[dict[str, Any]] = []
            else:
                rows = pq.read_table(path).to_pylist()
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


def audio_seconds_by_language(sections: list[dict[str, Any]]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for section in sections:
        language = (section["language"] or "").strip() or "Unknown"
        totals[language] += section["duration_seconds"] or 0
    return dict(totals)


def audio_seconds_delta(
    old_sections: list[dict[str, Any]],
    new_sections: list[dict[str, Any]],
) -> dict[str, int]:
    delta = Counter(audio_seconds_by_language(new_sections))
    delta.subtract(audio_seconds_by_language(old_sections))
    return dict(delta)


def merge_audio_seconds_by_language(
    current: dict[str, int],
    delta: dict[str, int],
) -> dict[str, int]:
    merged = Counter(current)
    merged.update(delta)
    return {language: seconds for language, seconds in merged.items() if seconds > 0}


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
        "authors": [author.model_dump(mode="json") for author in book.authors],
        "translators": [translator.model_dump(mode="json") for translator in book.translators],
        "genres": [genre.model_dump(mode="json") for genre in book.genres],
        "url_librivox": book.url_librivox,
        "url_iarchive": book.url_iarchive,
        "url_project": book.url_project,
        "url_rss": book.url_rss,
        "url_text_source": book.url_text_source,
        "url_other": book.url_other,
        "url_zip_file": book.url_zip_file,
        "archive_identifier": artifact.archive_identifier,
        "source_fingerprint": book.source_fingerprint,
        "tar_path": artifact_repo_path(book.id),
        "tar_size": artifact.size,
        "tar_sha256": artifact.sha256,
        "librivox_metadata": book.source_metadata_json,
        "archive_metadata": artifact.archive_metadata_json,
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
                "readers": [reader.model_dump(mode="json") for reader in section.readers],
                "hash_partition": artifact.book.hash_partition,
                "sample_key": section.sample_key,
                "tar_path": artifact_repo_path(artifact.book.id),
                "librivox_file_name": section.file_name,
                "librivox_listen_url": section.listen_url,
                "archive_identifier": artifact.archive_identifier,
                "archive_file": source.name,
                "archive_file_format": source.format,
                "archive_file_source": source.source,
                "archive_original_file": source.original,
                "source_url": (
                    "https://archive.org/download/"
                    f"{quote(artifact.archive_identifier, safe='')}/{quote(source.name, safe='/')}"
                ),
                "source_size": source.size,
                "source_md5": source.md5,
                "source_sha1": source.sha1,
                "mirror_sha256": download.sha256,
                "librivox_metadata": section.source_metadata_json,
                "archive_file_metadata": source.source_metadata_json,
            }
        )
    return rows


def preview_row(section: dict[str, Any], repo_id: str) -> dict[str, Any]:
    return {
        "audio": {
            "bytes": None,
            "path": preview_audio_uri(repo_id, section["sample_key"]),
        },
        **section,
    }


def preview_audio_path(sample_key: str) -> str:
    return f"preview/{quote(sample_key, safe='')}.mp3"


def preview_audio_uri(repo_id: str, sample_key: str) -> str:
    repository = quote(repo_id, safe="/")
    return f"hf://datasets/{repository}@main/{preview_audio_path(sample_key)}"


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
        "librivox_metadata": update.book.source_metadata_json,
    }


def dataset_card(
    state: SyncState,
    repo_id: str,
    *,
    preview_available: bool | None = None,
) -> str:
    updated_at = (
        state.updated_at.isoformat().replace("+00:00", "Z") if state.updated_at else "not yet"
    )
    repo_url = f"https://huggingface.co/datasets/{quote(repo_id, safe='/')}"
    updated_at_badge = quote(updated_at.replace("-", "--"), safe="")
    total_audio_seconds = sum(state.audio_seconds_by_language.values())
    has_preview = state.published_sections > 0 if preview_available is None else preview_available
    return _DATASET_CARD_TEMPLATE.substitute(
        repo_url=repo_url,
        updated_at=updated_at,
        updated_at_badge=updated_at_badge,
        published_books=f"{state.published_books:,}",
        published_sections=f"{state.published_sections:,}",
        quarantined_books=f"{state.quarantined_books:,}",
        audio_hours=format_audio_hours(total_audio_seconds),
        audio_languages=f"{len(state.audio_seconds_by_language):,}",
        audio_by_language=audio_by_language_table(state.audio_seconds_by_language),
        configs=dataset_configs(has_preview),
    )


def dataset_configs(has_preview: bool) -> str:
    configs = []
    if has_preview:
        configs.extend(
            [
                "- config_name: preview",
                "  default: true",
                "  data_files:",
                "  - split: train",
                "    path: metadata/preview/*.parquet",
            ]
        )
    configs.extend(
        [
            "- config_name: sections",
            *([] if has_preview else ["  default: true"]),
            "  data_files:",
            "  - split: train",
            "    path: metadata/sections/*.parquet",
            "- config_name: books",
            "  data_files:",
            "  - split: train",
            "    path: metadata/books/*.parquet",
        ]
    )
    return "\n".join(configs)


def audio_by_language_table(audio_seconds: dict[str, int]) -> str:
    if not audio_seconds:
        return "No audio published yet."
    rows = ["| Language | Hours |", "| --- | ---: |"]
    ordered = sorted(audio_seconds.items(), key=lambda item: (-item[1], item[0].casefold()))
    rows.extend(
        f"| {language.replace('|', '&#124;')} | {format_audio_hours(seconds)} |"
        for language, seconds in ordered
    )
    return "\n".join(rows)


def format_audio_hours(seconds: int) -> str:
    return f"{seconds / 3600:,.1f}"
