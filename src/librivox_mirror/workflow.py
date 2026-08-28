from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from librivox_mirror.archive import (
    InternetArchiveClient,
    QuarantinedBookError,
    SourceUnavailableError,
)
from librivox_mirror.artifact import (
    InvalidArtifactError,
    InvalidAudioError,
    artifact_manifest_path,
    build_artifact,
    load_artifact_manifest,
    verify_artifact,
    verify_mp3,
    write_artifact_manifest,
)
from librivox_mirror.capacity import StagingCapacity
from librivox_mirror.catalog import LibriVoxCatalog
from librivox_mirror.hub import HubPublisher, PublishResult, QuarantineUpdate
from librivox_mirror.models import Book, BookArtifact, BookStatus, QuarantineRecord, SyncState
from librivox_mirror.network import is_transient_http_error
from librivox_mirror.state import BookCheckpoint, StateStore

PUBLISH_MAX_RETRY_DELAY = 300
TAR_STAGING_OVERHEAD_PER_SECTION = 64 * 1024
logger = logging.getLogger(__name__)


class BookProgress(Protocol):
    def stage(self, stage: str) -> None: ...

    def download(self, completed_bytes: int, total_bytes: int) -> None: ...


@dataclass(frozen=True)
class BookOutcome:
    book: Book
    artifact: BookArtifact | None = None
    quarantine: QuarantineRecord | None = None
    skipped: bool = False
    error: str | None = None


class MirrorRunner:
    def __init__(
        self,
        *,
        catalog: LibriVoxCatalog,
        archive: InternetArchiveClient,
        state: StateStore,
        staging_directory: Path,
        publisher: HubPublisher | None = None,
        source_index: HubPublisher | None = None,
        staging_capacity: StagingCapacity | None = None,
    ) -> None:
        self.catalog = catalog
        self.archive = archive
        self.state = state
        self.staging_directory = staging_directory
        self.publisher = publisher
        self.source_index = source_index
        self.staging_capacity = staging_capacity

    def prepare_book(
        self,
        book: Book,
        *,
        progress: BookProgress | None = None,
    ) -> BookOutcome:
        checkpoint = self.state.discover(book)
        self.state.begin_attempt(book.id)
        started_at = time.monotonic()
        if progress is not None:
            progress.stage("checking")
        try:
            outcome = self._prepare_book(book, checkpoint, progress=progress)
        except Exception as error:
            self.state.record_failure(book.id, error)
            raise
        logger.info("Finished book %s in %.1fs", book.id, time.monotonic() - started_at)
        return outcome

    def prepare_book_resiliently(
        self,
        book: Book,
        *,
        progress: BookProgress | None = None,
    ) -> BookOutcome:
        try:
            return self.prepare_book(book, progress=progress)
        except (SourceUnavailableError, InvalidAudioError) as error:
            self.cleanup_downloads(book.id)
            if self.staging_capacity is not None:
                self.staging_capacity.release(book.id)
            detail = f"{type(error).__name__}: {error}"
            logger.error("Deferred book %s after a source failure: %s", book.id, detail)
            return BookOutcome(book=book, error=detail)

    def _prepare_book(
        self,
        book: Book,
        checkpoint: BookCheckpoint,
        *,
        progress: BookProgress | None,
    ) -> BookOutcome:
        if self.source_index and self.source_index.has_current_book(book):
            self.cleanup_paths(book.id, checkpoint.artifact_path)
            if checkpoint.status != BookStatus.PUBLISHED:
                self.state.transition(book.id, BookStatus.PUBLISHED)
            logger.info("Book %s already matches the Hub", book.id)
            return BookOutcome(book=book, skipped=True)
        if checkpoint.status == BookStatus.PUBLISHED and self.publisher is None:
            self.cleanup_paths(book.id, checkpoint.artifact_path)
            logger.info("Book %s is already published", book.id)
            return BookOutcome(book=book, skipped=True)
        if checkpoint.status == BookStatus.PACKED:
            if progress is not None:
                progress.stage("restoring")
            artifact = self.restore_artifact(book, checkpoint.artifact_path)
            if artifact is not None:
                if self.staging_capacity is not None:
                    if progress is not None:
                        progress.stage("waiting for staging")
                    self.staging_capacity.reserve(book.id, artifact.size, existing=True)
                self.cleanup_downloads(book.id)
                logger.info("Resumed packed book %s from %s", book.id, artifact.path)
                return BookOutcome(book=book, artifact=artifact)
        if checkpoint.status != BookStatus.DISCOVERED:
            self.discard_artifact(book.id, checkpoint.artifact_path)
            self.state.restart(book.id)

        if progress is not None:
            progress.stage("resolving")
        logger.info("Resolving original files for book %s", book.id)
        resolve_started_at = time.monotonic()
        try:
            resolved = self.archive.resolve_book(book)
        except QuarantinedBookError as error:
            self.state.quarantine(error.record)
            logger.warning("Quarantined book %s: %s", book.id, error.record.detail)
            return BookOutcome(book=book, quarantine=error.record)
        source_bytes = sum(section.archive_file.size for section in resolved.sections)
        logger.info(
            "Resolved %s sections and %.1f MiB for book %s in %.1fs",
            len(resolved.sections),
            source_bytes / 1024**2,
            book.id,
            time.monotonic() - resolve_started_at,
        )
        self.state.transition(
            book.id,
            BookStatus.RESOLVED,
            archive_identifier=resolved.archive_identifier,
        )

        if self.staging_capacity is not None:
            if progress is not None:
                progress.stage("waiting for staging")
            self.staging_capacity.reserve(
                book.id,
                source_bytes * 2 + len(resolved.sections) * TAR_STAGING_OVERHEAD_PER_SECTION,
            )

        download_directory = self.staging_directory / "downloads" / f"{book.id:06d}"
        logger.info(
            "Downloading %s original MP3 files for book %s", len(resolved.sections), book.id
        )
        if progress is not None:
            progress.stage("downloading")
        download_started_at = time.monotonic()
        downloads = self.archive.download_book(
            resolved,
            download_directory,
            progress=progress.download if progress is not None else None,
        )
        download_seconds = time.monotonic() - download_started_at
        self.state.transition(book.id, BookStatus.DOWNLOADED)
        logger.info(
            "Downloaded %.1f MiB for book %s in %.1fs (%.1f MiB/s effective)",
            source_bytes / 1024**2,
            book.id,
            download_seconds,
            source_bytes / 1024**2 / max(download_seconds, 0.001),
        )
        if progress is not None:
            progress.stage("verifying")
        verify_started_at = time.monotonic()
        for download in downloads:
            verify_mp3(download.path)
        self.state.transition(book.id, BookStatus.VERIFIED)
        logger.info(
            "Verified %s MP3 files for book %s in %.1fs",
            len(downloads),
            book.id,
            time.monotonic() - verify_started_at,
        )

        if progress is not None:
            progress.stage("packing")
        pack_started_at = time.monotonic()
        artifact = build_artifact(resolved, downloads, self.staging_directory / "repository")
        write_artifact_manifest(artifact, self.manifest_path(book.id))
        self.state.transition(
            book.id,
            BookStatus.PACKED,
            artifact_path=artifact.path,
            artifact_sha256=artifact.sha256,
        )
        self.cleanup_downloads(book.id)
        if self.staging_capacity is not None:
            self.staging_capacity.resize(book.id, artifact.size)
        logger.info(
            "Packed book %s into %.1f MiB at %s in %.1fs",
            book.id,
            artifact.size / 1024**2,
            artifact.path,
            time.monotonic() - pack_started_at,
        )
        return BookOutcome(book=book, artifact=artifact)

    def publish(
        self,
        outcomes: list[BookOutcome],
        sync_state: SyncState,
        *,
        commit_message: str,
    ) -> PublishResult | None:
        artifacts = [outcome.artifact for outcome in outcomes if outcome.artifact is not None]
        quarantines = [
            QuarantineUpdate(book=outcome.book, record=outcome.quarantine)
            for outcome in outcomes
            if outcome.quarantine is not None
        ]
        if not artifacts and not quarantines:
            return None
        if self.publisher is None:
            logger.info("Kept %s prepared books locally without publishing", len(artifacts))
            return None

        artifact_bytes = sum(artifact.size for artifact in artifacts)
        logger.info(
            "Publishing %s books and %s quarantines (%.1f MiB)",
            len(artifacts),
            len(quarantines),
            artifact_bytes / 1024**2,
        )
        publish_started_at = time.monotonic()
        try:
            result = self._publish_resiliently(
                artifacts,
                quarantines,
                sync_state,
                commit_message=commit_message,
            )
        except Exception as error:
            for artifact in artifacts:
                self.state.record_failure(artifact.book.id, error)
            for update in quarantines:
                self.state.record_failure(update.book.id, error)
            raise
        publish_seconds = time.monotonic() - publish_started_at
        for artifact in artifacts:
            self.state.transition(
                artifact.book.id,
                BookStatus.PUBLISHED,
                published_revision=result.revision,
            )
            self.cleanup(artifact)
            if self.staging_capacity is not None:
                self.staging_capacity.release(artifact.book.id)
        logger.info(
            "Published revision %s in %.1fs (%.1f MiB/s effective)",
            result.revision,
            publish_seconds,
            artifact_bytes / 1024**2 / max(publish_seconds, 0.001),
        )
        return result

    def _publish_resiliently(
        self,
        artifacts: list[BookArtifact],
        quarantines: list[QuarantineUpdate],
        sync_state: SyncState,
        *,
        commit_message: str,
    ) -> PublishResult:
        if self.publisher is None:
            raise RuntimeError("cannot publish without a Hub publisher")

        books = [artifact.book for artifact in artifacts]
        books.extend(update.book for update in quarantines)
        current_state = sync_state
        attempt = 1
        while True:
            try:
                if attempt > 1:
                    self.publisher.invalidate_cache()
                    if all(self.publisher.has_current_book(book) for book in books):
                        return PublishResult(
                            revision=self.publisher.current_revision(),
                            state=self.publisher.load_sync_state(),
                        )
                    current_state = self.publisher.load_sync_state()
                return self.publisher.publish(
                    artifacts,
                    quarantines,
                    current_state,
                    commit_message=commit_message,
                )
            except Exception as error:
                if not is_transient_http_error(error):
                    raise
                delay = min(2**attempt, PUBLISH_MAX_RETRY_DELAY)
                logger.warning(
                    "Retrying Hub publication after attempt %s failed; waiting %ss: %s",
                    attempt,
                    delay,
                    error,
                )
                time.sleep(delay)
                attempt += 1

    def cleanup(self, artifact: BookArtifact) -> None:
        self.cleanup_paths(artifact.book.id, artifact.path)

    def cleanup_paths(self, book_id: int, artifact_path: Path | None) -> None:
        self.cleanup_downloads(book_id)
        self.discard_artifact(book_id, artifact_path)

    def cleanup_downloads(self, book_id: int) -> None:
        download_directory = self.staging_directory / "downloads" / f"{book_id:06d}"
        shutil.rmtree(download_directory, ignore_errors=True)

    def discard_artifact(self, book_id: int, artifact_path: Path | None) -> None:
        if artifact_path:
            artifact_path.unlink(missing_ok=True)
        self.manifest_path(book_id).unlink(missing_ok=True)

    def manifest_path(self, book_id: int) -> Path:
        return artifact_manifest_path(self.staging_directory, book_id)

    def restore_artifact(self, book: Book, checkpoint_path: Path | None) -> BookArtifact | None:
        try:
            if checkpoint_path is None:
                raise InvalidArtifactError("packed checkpoint has no artifact path")
            artifact = load_artifact_manifest(self.manifest_path(book.id))
            if artifact.book.source_fingerprint != book.source_fingerprint:
                raise InvalidArtifactError("artifact source fingerprint does not match the catalog")
            if artifact.path != checkpoint_path:
                raise InvalidArtifactError("artifact path does not match its checkpoint")
            checkpoint = self.state.get(book.id)
            if checkpoint is None or artifact.sha256 != checkpoint.artifact_sha256:
                raise InvalidArtifactError("artifact sha256 does not match its checkpoint")
            if artifact.path.stat().st_size != artifact.size:
                raise InvalidArtifactError("artifact size does not match its manifest")
            _, sample_count = verify_artifact(artifact.path, artifact.sha256)
            if sample_count != len(artifact.sections):
                raise InvalidArtifactError("artifact sample count does not match its manifest")
        except (InvalidArtifactError, OSError) as error:
            logger.warning("Discarding unusable checkpoint for book %s: %s", book.id, error)
            return None
        return artifact
