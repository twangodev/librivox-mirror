from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote, unquote, urlparse

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

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
ARCHIVE_METADATA_URL = "https://archive.org/metadata/{identifier}"
DOWNLOAD_ATTEMPTS = 10
logger = logging.getLogger(__name__)

type DownloadProgress = Callable[[int, int], None]
type DownloadTask = Callable[[], DownloadedSection]


class QuarantinedBookError(Exception):
    def __init__(self, record: QuarantineRecord) -> None:
        super().__init__(f"book {record.book_id}: {record.code}: {record.detail}")
        self.record = record


class DownloadIntegrityError(OSError):
    pass


class SourceUnavailableError(OSError):
    pass


class InvalidMetadataResponseError(ValueError):
    pass


class ArchiveItemMissingError(LookupError):
    pass


def is_retryable_metadata_error(error: BaseException) -> bool:
    return isinstance(error, (ArchiveItemMissingError, InvalidMetadataResponseError)) or (
        isinstance(error, Exception) and is_transient_http_error(error)
    )


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


class DownloadBatch:
    def __init__(self, tasks: Sequence[DownloadTask]) -> None:
        self.pending = deque(enumerate(tasks))
        self.results: list[DownloadedSection | None] = [None] * len(tasks)
        self.remaining = len(tasks)
        self.error: BaseException | None = None
        self.done = threading.Event()
        if not tasks:
            self.done.set()

    def result(self) -> tuple[DownloadedSection, ...]:
        self.done.wait()
        if self.error is not None:
            raise self.error
        return tuple(cast(DownloadedSection, result) for result in self.results)


class DownloadPool:
    def __init__(self, workers: int) -> None:
        if workers < 1:
            raise ValueError("download worker count must be positive")
        self._ready: deque[DownloadBatch] = deque()
        self._condition = threading.Condition()
        self._closed = False
        self._workers = tuple(
            threading.Thread(
                target=self._work,
                name=f"audio-downloader-{index}",
            )
            for index in range(1, workers + 1)
        )
        for worker in self._workers:
            worker.start()

    def submit(self, tasks: Sequence[DownloadTask]) -> DownloadBatch:
        batch = DownloadBatch(tasks)
        with self._condition:
            if self._closed:
                raise RuntimeError("download pool is closed")
            if batch.pending:
                self._ready.append(batch)
                self._condition.notify_all()
        return batch

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        for worker in self._workers:
            worker.join()

    def _work(self) -> None:
        while True:
            assignment = self._next()
            if assignment is None:
                return
            batch, index, task = assignment
            try:
                result = task()
            except BaseException as error:
                self._finish(batch, index, error=error)
            else:
                self._finish(batch, index, result=result)

    def _next(self) -> tuple[DownloadBatch, int, DownloadTask] | None:
        with self._condition:
            self._condition.wait_for(lambda: self._ready or self._closed)
            if not self._ready:
                return None
            batch = self._ready.popleft()
            index, task = batch.pending.popleft()
            if batch.pending:
                self._ready.append(batch)
            return batch, index, task

    def _finish(
        self,
        batch: DownloadBatch,
        index: int,
        *,
        result: DownloadedSection | None = None,
        error: BaseException | None = None,
    ) -> None:
        with self._condition:
            batch.results[index] = result
            if error is not None and batch.error is None:
                batch.error = error
            batch.remaining -= 1
            if batch.remaining == 0:
                batch.done.set()


class InternetArchiveClient:
    def __init__(
        self,
        *,
        user_agent: str,
        request_delay: float = 1,
        timeout: float = 120,
        download_jobs: int = 4,
        client: httpx.Client | None = None,
    ) -> None:
        self._metadata_limiter = RequestLimiter(request_delay)
        self._owns_client = client is None
        self._client = client or httpx.Client(
            headers={"User-Agent": user_agent},
            follow_redirects=True,
            timeout=timeout,
        )
        self._download_pool = DownloadPool(download_jobs)

    def close(self) -> None:
        self._download_pool.close()
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> InternetArchiveClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def resolve_book(self, book: Book) -> ResolvedBook:
        identifiers = archive_identifiers(book)
        if not identifiers:
            raise quarantine(
                book,
                QuarantineCode.ARCHIVE_IDENTIFIER_MISSING,
                f"could not parse an Internet Archive identifier from {book.url_iarchive!r}",
            )
        missing = []
        for identifier in identifiers:
            try:
                payload = self._get_metadata(identifier)
            except ArchiveItemMissingError:
                missing.append(identifier)
                continue
            except (httpx.HTTPError, InvalidMetadataResponseError) as error:
                raise SourceUnavailableError(
                    f"could not load Internet Archive item {identifier!r}: {error}"
                ) from error
            files = payload.get("files")
            metadata = payload.get("metadata")
            if not isinstance(files, list) or not all(isinstance(row, Mapping) for row in files):
                raise SourceUnavailableError(
                    f"Internet Archive item {identifier!r} returned invalid file metadata"
                )
            if not isinstance(metadata, Mapping):
                raise SourceUnavailableError(
                    f"Internet Archive item {identifier!r} returned invalid item metadata"
                )
            return resolve_original_files(book, identifier, files, metadata)
        identifiers_text = ", ".join(repr(identifier) for identifier in missing)
        raise quarantine(
            book,
            QuarantineCode.ARCHIVE_ITEM_MISSING,
            f"Internet Archive item(s) {identifiers_text} do not exist",
            missing[0],
        )

    def download_book(
        self,
        resolved: ResolvedBook,
        destination: Path,
        *,
        progress: DownloadProgress | None = None,
    ) -> tuple[DownloadedSection, ...]:
        destination.mkdir(parents=True, exist_ok=True)
        completed_bytes: dict[int, int] = {}
        completed_total = 0
        progress_lock = threading.Lock()
        total_bytes = sum(section.archive_file.size for section in resolved.sections)

        def report(section_id: int, section_bytes: int) -> None:
            nonlocal completed_total
            if progress is None:
                return
            with progress_lock:
                completed_total += section_bytes - completed_bytes.get(section_id, 0)
                completed_bytes[section_id] = section_bytes
                progress(completed_total, total_bytes)

        if progress is not None:
            progress(0, total_bytes)
        tasks = [
            partial(
                self.download_section,
                resolved.archive_identifier,
                section,
                destination,
                progress=partial(report, section.section.id),
            )
            for section in resolved.sections
        ]
        return self._download_pool.submit(tasks).result()

    def download_section(
        self,
        identifier: str,
        resolved: ResolvedSection,
        destination: Path,
        *,
        progress: Callable[[int], None] | None = None,
    ) -> DownloadedSection:
        path = destination / f"{resolved.section.sample_key}.mp3"
        if path.exists():
            try:
                sha256 = verify_download(path, resolved.archive_file)
            except DownloadIntegrityError as error:
                logger.warning("Replacing corrupt staged download %s: %s", path, error)
                path.unlink()
            else:
                if progress is not None:
                    progress(resolved.archive_file.size)
                return DownloadedSection(resolved=resolved, path=path, sha256=sha256)

        partial = path.with_suffix(".mp3.partial")
        for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
            if progress is not None:
                progress(0)
            try:
                sha256 = self._download_once(
                    identifier,
                    resolved.archive_file,
                    partial,
                    progress=progress,
                )
                partial.replace(path)
                return DownloadedSection(resolved=resolved, path=path, sha256=sha256)
            except (httpx.TransportError, httpx.HTTPStatusError, DownloadIntegrityError) as error:
                partial.unlink(missing_ok=True)
                if progress is not None:
                    progress(0)
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
        retry=retry_if_exception(is_retryable_metadata_error),
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=1, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _get_metadata(self, identifier: str) -> Mapping[str, Any]:
        self._metadata_limiter.wait()
        url = ARCHIVE_METADATA_URL.format(identifier=quote(identifier, safe=""))
        response = self._client.get(url)
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError as error:
            raise InvalidMetadataResponseError("response was not valid JSON") from error
        if not isinstance(payload, Mapping):
            raise InvalidMetadataResponseError("response was not a JSON object")
        if not payload or payload.get("is_dark") is True:
            raise ArchiveItemMissingError(f"Internet Archive item {identifier!r} is missing")
        return payload

    def _download_once(
        self,
        identifier: str,
        archive_file: ArchiveFile,
        partial: Path,
        *,
        progress: Callable[[int], None] | None = None,
    ) -> str:
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
                    if progress is not None:
                        progress(size)
        verify_hashes(archive_file, size=size, md5=md5.hexdigest(), sha1=sha1.hexdigest())
        return sha256.hexdigest()


def archive_identifier(url: str) -> str | None:
    path = unquote(urlparse(url).path).strip("/")
    if not path:
        return None
    parts = path.split("/")
    for marker in ("details", "download", "metadata", "compress"):
        if marker in parts:
            index = parts.index(marker)
            return parts[index + 1] if len(parts) > index + 1 else None
    return parts[-1]


def archive_identifiers(book: Book) -> tuple[str, ...]:
    urls = [book.url_iarchive, book.url_zip_file or ""]
    if book.url_librivox:
        urls.extend(section.listen_url for section in book.sections)
    return tuple(
        dict.fromkeys(identifier for url in urls if (identifier := archive_identifier(url)))
    )


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
    listened_candidates = candidates_for_name(listened_name, indexed, originals)
    if listened_candidates:
        return listened_candidates
    return candidates_for_name(section.file_name or "", indexed, originals)


def candidates_for_name(
    name: str,
    indexed: Mapping[str, Mapping[str, Any]],
    originals: Mapping[str, ArchiveFile],
) -> set[str]:
    if not name:
        return set()
    stripped_name = name.strip()
    referenced = indexed.get(name) or indexed.get(stripped_name)
    if referenced and referenced.get("original"):
        original_name = str(referenced["original"])
        for candidate in (original_name, original_name.strip()):
            if candidate in originals:
                return {candidate}
    for candidate in (name, stripped_name):
        if candidate in originals:
            return {candidate}
    canonical = canonical_audio_name(stripped_name)
    return {
        original_name
        for original_name in originals
        if canonical and canonical_audio_name(original_name) == canonical
    }


def canonical_audio_name(name: str) -> str:
    stem = name.strip().casefold()
    while stem:
        previous = stem
        stem = re.sub(r"\.mp3$", "", stem)
        stem = re.sub(
            r"(?:[._\s-]+(?:(?:\d{2,4})?[._]?k[bpslex]*|128|vbr))$",
            "",
            stem,
        )
        if stem == previous:
            break
    return re.sub(r"[^a-z0-9]+", "", stem)


def is_original_mp3(row: Mapping[str, Any]) -> bool:
    source = str(row.get("source") or "").casefold()
    format_name = str(row.get("format") or "").casefold()
    return source == "original" and "mp3" in format_name


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
