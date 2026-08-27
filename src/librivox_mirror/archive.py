from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

import httpx
import requests
from internetarchive.session import ArchiveSession
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from librivox_mirror.models import (
    ArchiveFile,
    Book,
    DownloadedSection,
    QuarantineCode,
    QuarantineRecord,
    ResolvedBook,
    ResolvedSection,
    Section,
    canonical_metadata_json,
)
from librivox_mirror.network import is_transient_http_error

ARCHIVE_DOWNLOAD_URL = "https://archive.org/download/{identifier}/{filename}"
DOWNLOAD_ATTEMPTS = 10
logger = logging.getLogger(__name__)


class QuarantinedBookError(Exception):
    def __init__(self, record: QuarantineRecord) -> None:
        super().__init__(f"book {record.book_id}: {record.code}: {record.detail}")
        self.record = record


class DownloadIntegrityError(OSError):
    pass


class SourceUnavailableError(OSError):
    pass


class RequestLimiter:
    def __init__(self, delay_seconds: float) -> None:
        self._delay_seconds = max(0, delay_seconds)
        self._next_request = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            remaining = self._next_request - now
            if remaining > 0:
                time.sleep(remaining)
            self._next_request = time.monotonic() + self._delay_seconds


class InternetArchiveClient:
    def __init__(
        self,
        *,
        user_agent: str,
        request_delay: float = 1,
        timeout: float = 120,
        archive_session: ArchiveSession | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.timeout = timeout
        self._limiter = RequestLimiter(request_delay)
        self._archive_session = archive_session or ArchiveSession(
            config={"general": {"user_agent_suffix": user_agent}}
        )
        self._owns_client = client is None
        self._client = client or httpx.Client(
            headers={"User-Agent": user_agent},
            follow_redirects=True,
            timeout=timeout,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
        self._archive_session.close()

    def __enter__(self) -> InternetArchiveClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def resolve_book(self, book: Book) -> ResolvedBook:
        identifier = archive_identifier(book.url_iarchive)
        if identifier is None:
            raise quarantine(
                book,
                QuarantineCode.ARCHIVE_IDENTIFIER_MISSING,
                f"could not parse an Internet Archive identifier from {book.url_iarchive!r}",
            )
        try:
            item = self._get_item(identifier)
        except requests.RequestException as error:
            raise SourceUnavailableError(
                f"could not load Internet Archive item {identifier!r}: {error}"
            ) from error
        if not item.exists:
            raise quarantine(
                book,
                QuarantineCode.ARCHIVE_ITEM_MISSING,
                f"Internet Archive item {identifier!r} does not exist",
                identifier,
            )
        return resolve_original_files(book, identifier, item.files, item.metadata)

    def download_book(
        self,
        resolved: ResolvedBook,
        destination: Path,
        *,
        jobs: int,
    ) -> tuple[DownloadedSection, ...]:
        destination.mkdir(parents=True, exist_ok=True)
        worker_count = max(1, min(jobs, 4))
        downloads: dict[int, DownloadedSection] = {}
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    self.download_section, resolved.archive_identifier, section, destination
                ): section
                for section in resolved.sections
            }
            for future in as_completed(futures):
                downloaded = future.result()
                downloads[downloaded.resolved.section.id] = downloaded
        return tuple(downloads[section.section.id] for section in resolved.sections)

    def download_section(
        self,
        identifier: str,
        resolved: ResolvedSection,
        destination: Path,
    ) -> DownloadedSection:
        path = destination / f"{resolved.section.sample_key}.mp3"
        if path.exists():
            try:
                sha256 = verify_download(path, resolved.archive_file)
            except DownloadIntegrityError as error:
                logger.warning("Replacing corrupt staged download %s: %s", path, error)
                path.unlink()
            else:
                return DownloadedSection(resolved=resolved, path=path, sha256=sha256)

        partial = path.with_suffix(".mp3.partial")
        for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
            try:
                sha256 = self._download_once(identifier, resolved.archive_file, partial)
                partial.replace(path)
                return DownloadedSection(resolved=resolved, path=path, sha256=sha256)
            except (httpx.TransportError, httpx.HTTPStatusError, DownloadIntegrityError) as error:
                partial.unlink(missing_ok=True)
                retryable = isinstance(error, DownloadIntegrityError) or is_transient_http_error(
                    error
                )
                if not retryable or attempt == DOWNLOAD_ATTEMPTS:
                    raise SourceUnavailableError(
                        f"could not download {resolved.archive_file.name!r} after "
                        f"{attempt} attempt(s): {error}"
                    ) from error
                logger.warning(
                    "Retrying %s after attempt %s/%s failed: %s",
                    resolved.archive_file.name,
                    attempt,
                    DOWNLOAD_ATTEMPTS,
                    error,
                )
                time.sleep(min(2**attempt, 30))
        raise AssertionError("download retry loop terminated unexpectedly")

    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=1, max=30),
        reraise=True,
    )
    def _get_item(self, identifier: str):
        self._limiter.wait()
        return self._archive_session.get_item(
            identifier,
            request_kwargs={"timeout": self.timeout},
        )

    def _download_once(self, identifier: str, archive_file: ArchiveFile, partial: Path) -> str:
        self._limiter.wait()
        url = ARCHIVE_DOWNLOAD_URL.format(
            identifier=quote(identifier, safe=""),
            filename=quote(archive_file.name, safe="/"),
        )
        md5 = hashlib.md5(usedforsecurity=False)
        sha1 = hashlib.sha1(usedforsecurity=False)
        sha256 = hashlib.sha256()
        size = 0
        with self._client.stream("GET", url) as response:
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    time.sleep(min(int(retry_after), 300))
            response.raise_for_status()
            with partial.open("wb") as output:
                for chunk in response.iter_bytes(1024 * 1024):
                    output.write(chunk)
                    md5.update(chunk)
                    sha1.update(chunk)
                    sha256.update(chunk)
                    size += len(chunk)
        verify_hashes(archive_file, size=size, md5=md5.hexdigest(), sha1=sha1.hexdigest())
        return sha256.hexdigest()


def archive_identifier(url: str) -> str | None:
    path = unquote(urlparse(url).path).strip("/")
    if not path:
        return None
    parts = path.split("/")
    for marker in ("details", "download", "metadata"):
        if marker in parts:
            index = parts.index(marker)
            return parts[index + 1] if len(parts) > index + 1 else None
    return parts[-1]


def resolve_original_files(
    book: Book,
    identifier: str,
    rows: Sequence[Mapping[str, Any]],
    archive_metadata: Mapping[str, Any] | None = None,
) -> ResolvedBook:
    indexed = {str(row.get("name")): row for row in rows if row.get("name")}
    originals = {name: archive_file(row) for name, row in indexed.items() if is_original_mp3(row)}
    mapped: list[ResolvedSection] = []
    used_files: set[str] = set()
    for section in book.sections:
        candidates = original_candidates(section, indexed, originals)
        if not candidates:
            raise quarantine(
                book,
                QuarantineCode.ORIGINAL_FILE_MISSING,
                f"section {section.id} has no matching original MP3",
                identifier,
            )
        if len(candidates) > 1:
            names = ", ".join(sorted(candidates))
            raise quarantine(
                book,
                QuarantineCode.ORIGINAL_FILE_AMBIGUOUS,
                f"section {section.id} matches multiple original MP3s: {names}",
                identifier,
            )
        name = next(iter(candidates))
        if name in used_files:
            raise quarantine(
                book,
                QuarantineCode.SECTION_MAPPING_DUPLICATE,
                f"multiple sections map to original MP3 {name!r}",
                identifier,
            )
        used_files.add(name)
        mapped.append(ResolvedSection(section=section, archive_file=originals[name]))
    return ResolvedBook(
        book=book,
        archive_identifier=identifier,
        sections=tuple(mapped),
        archive_metadata_json=canonical_metadata_json(archive_metadata or {}),
    )


def original_candidates(
    section: Section,
    indexed: Mapping[str, Mapping[str, Any]],
    originals: Mapping[str, ArchiveFile],
) -> set[str]:
    listened_name = Path(unquote(urlparse(section.listen_url).path)).name
    listened = indexed.get(listened_name)
    derived_from = str(listened.get("original")) if listened and listened.get("original") else None
    if derived_from is not None and derived_from in originals:
        return {derived_from}
    if listened_name in originals:
        return {listened_name}
    canonical = canonical_audio_name(listened_name)
    return {name for name in originals if canonical_audio_name(name) == canonical}


def canonical_audio_name(name: str) -> str:
    stem = Path(name).stem.casefold()
    return re.sub(r"(?:[_-](?:64|128)kb|[_-]vbr)$", "", stem)


def is_original_mp3(row: Mapping[str, Any]) -> bool:
    name = str(row.get("name") or "")
    source = str(row.get("source") or "").casefold()
    format_name = str(row.get("format") or "").casefold()
    return name.casefold().endswith(".mp3") and source == "original" and "mp3" in format_name


def archive_file(row: Mapping[str, Any]) -> ArchiveFile:
    return ArchiveFile(
        name=str(row["name"]),
        size=int(row.get("size") or 0),
        md5=_optional_string(row.get("md5")),
        sha1=_optional_string(row.get("sha1")),
        format=_optional_string(row.get("format")),
        source=_optional_string(row.get("source")),
        original=_optional_string(row.get("original")),
        source_metadata_json=canonical_metadata_json(row),
    )


def verify_download(path: Path, expected: ArchiveFile) -> str:
    md5 = hashlib.md5(usedforsecurity=False)
    sha1 = hashlib.sha1(usedforsecurity=False)
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            md5.update(chunk)
            sha1.update(chunk)
            sha256.update(chunk)
            size += len(chunk)
    verify_hashes(expected, size=size, md5=md5.hexdigest(), sha1=sha1.hexdigest())
    return sha256.hexdigest()


def verify_hashes(expected: ArchiveFile, *, size: int, md5: str, sha1: str) -> None:
    mismatches = []
    if expected.size and size != expected.size:
        mismatches.append(f"size {size} != {expected.size}")
    if expected.md5 and md5.casefold() != expected.md5.casefold():
        mismatches.append(f"md5 {md5} != {expected.md5}")
    if expected.sha1 and sha1.casefold() != expected.sha1.casefold():
        mismatches.append(f"sha1 {sha1} != {expected.sha1}")
    if mismatches:
        raise DownloadIntegrityError("; ".join(mismatches))


def quarantine(
    book: Book,
    code: QuarantineCode,
    detail: str,
    identifier: str | None = None,
) -> QuarantinedBookError:
    return QuarantinedBookError(
        QuarantineRecord(
            book_id=book.id,
            title=book.title,
            code=code,
            detail=detail,
            archive_identifier=identifier,
            source_fingerprint=book.source_fingerprint,
        )
    )


def _optional_string(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None
