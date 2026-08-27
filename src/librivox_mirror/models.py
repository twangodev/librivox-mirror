from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, computed_field


class MirrorModel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class Author(MirrorModel):
    id: int | None = None
    first_name: str = ""
    last_name: str = ""
    dob: str | None = None
    dod: str | None = None


class Reader(MirrorModel):
    id: int | None = None
    display_name: str
    url_text: str | None = None


class Section(MirrorModel):
    id: int
    book_id: int
    section_number: int
    title: str
    language: str | None = None
    duration_seconds: int | None = None
    listen_url: str
    readers: tuple[Reader, ...] = ()

    @computed_field
    @property
    def sample_key(self) -> str:
        return f"{self.book_id:06d}-{self.id:08d}"


class Book(MirrorModel):
    id: int
    title: str
    description: str = ""
    language: str | None = None
    copyright_year: str | None = None
    total_time_seconds: int | None = None
    url_librivox: str
    url_iarchive: str
    url_project: str | None = None
    url_rss: str | None = None
    url_text_source: str | None = None
    authors: tuple[Author, ...] = ()
    sections: tuple[Section, ...] = ()

    @computed_field
    @property
    def metadata_bucket(self) -> int:
        return self.id // 1000

    @computed_field
    @property
    def hash_partition(self) -> int:
        digest = hashlib.sha256(str(self.id).encode()).digest()
        return int.from_bytes(digest[:8], "big") % 100

    @computed_field
    @property
    def source_fingerprint(self) -> str:
        payload = self.model_dump(mode="json", exclude={"source_fingerprint"})
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


class ArchiveFile(MirrorModel):
    name: str
    size: int
    md5: str | None = None
    sha1: str | None = None
    format: str | None = None
    source: str | None = None
    original: str | None = None


class ResolvedSection(MirrorModel):
    section: Section
    archive_file: ArchiveFile


class ResolvedBook(MirrorModel):
    book: Book
    archive_identifier: str
    sections: tuple[ResolvedSection, ...]


class DownloadedSection(MirrorModel):
    resolved: ResolvedSection
    path: Path
    sha256: str


class BookArtifact(MirrorModel):
    book: Book
    archive_identifier: str
    path: Path
    sha256: str
    size: int
    sections: tuple[DownloadedSection, ...]


class QuarantineCode(StrEnum):
    ARCHIVE_IDENTIFIER_MISSING = "archive_identifier_missing"
    ARCHIVE_ITEM_MISSING = "archive_item_missing"
    ORIGINAL_FILE_MISSING = "original_file_missing"
    ORIGINAL_FILE_AMBIGUOUS = "original_file_ambiguous"
    SECTION_MAPPING_DUPLICATE = "section_mapping_duplicate"


class QuarantineRecord(MirrorModel):
    book_id: int
    title: str
    code: QuarantineCode
    detail: str
    archive_identifier: str | None = None
    source_fingerprint: str
    observed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SyncState(MirrorModel):
    catalog_watermark: int = 0
    updated_at: datetime | None = None
    last_commit: str | None = None


class BookStatus(StrEnum):
    DISCOVERED = "discovered"
    RESOLVED = "resolved"
    DOWNLOADED = "downloaded"
    VERIFIED = "verified"
    PACKED = "packed"
    PUBLISHED = "published"
    QUARANTINED = "quarantined"
