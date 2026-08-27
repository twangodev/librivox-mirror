from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from librivox_mirror.archive import InternetArchiveClient, QuarantinedBookError
from librivox_mirror.artifact import (
    InvalidArtifactError,
    artifact_manifest_path,
    build_artifact,
    load_artifact_manifest,
    verify_artifact,
    verify_mp3,
    write_artifact_manifest,
)
from librivox_mirror.catalog import LibriVoxCatalog
from librivox_mirror.hub import HubPublisher, PublishResult, QuarantineUpdate
from librivox_mirror.models import Book, BookArtifact, BookStatus, QuarantineRecord, SyncState
from librivox_mirror.state import StateStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BookOutcome:
    book: Book
    artifact: BookArtifact | None = None
    quarantine: QuarantineRecord | None = None
    skipped: bool = False


class MirrorRunner:
    def __init__(
        self,
        *,
        catalog: LibriVoxCatalog,
        archive: InternetArchiveClient,
        state: StateStore,
        staging_directory: Path,
        jobs: int,
        publisher: HubPublisher | None = None,
    ) -> None:
        self.catalog = catalog
        self.archive = archive
        self.state = state
        self.staging_directory = staging_directory
        self.jobs = jobs
        self.publisher = publisher

    def prepare_book(self, book: Book) -> BookOutcome:
        checkpoint = self.state.discover(book)
        if self.publisher and self.publisher.has_current_book(book):
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
            artifact = self.restore_artifact(book, checkpoint.artifact_path)
            if artifact is not None:
                self.cleanup_downloads(book.id)
                logger.info("Resumed packed book %s from %s", book.id, artifact.path)
                return BookOutcome(book=book, artifact=artifact)
        if checkpoint.status != BookStatus.DISCOVERED:
            self.discard_artifact(book.id, checkpoint.artifact_path)
            self.state.restart(book.id)

        logger.info("Resolving original files for book %s", book.id)
        try:
            resolved = self.archive.resolve_book(book)
        except QuarantinedBookError as error:
            self.state.quarantine(error.record)
            logger.warning("Quarantined book %s: %s", book.id, error.record.detail)
            return BookOutcome(book=book, quarantine=error.record)
        self.state.transition(
            book.id,
            BookStatus.RESOLVED,
            archive_identifier=resolved.archive_identifier,
        )

        download_directory = self.staging_directory / "downloads" / f"{book.id:06d}"
        logger.info(
            "Downloading %s original MP3 files for book %s", len(resolved.sections), book.id
        )
        downloads = self.archive.download_book(resolved, download_directory, jobs=self.jobs)
        self.state.transition(book.id, BookStatus.DOWNLOADED)
        for download in downloads:
            verify_mp3(download.path)
        self.state.transition(book.id, BookStatus.VERIFIED)

        artifact = build_artifact(resolved, downloads, self.staging_directory / "repository")
        write_artifact_manifest(artifact, self.manifest_path(book.id))
        self.state.transition(
            book.id,
            BookStatus.PACKED,
            artifact_path=artifact.path,
            artifact_sha256=artifact.sha256,
        )
        self.cleanup_downloads(book.id)
        logger.info("Packed book %s into %s", book.id, artifact.path)
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

        result = self.publisher.publish(
            artifacts,
            quarantines,
            sync_state,
            commit_message=commit_message,
        )
        for artifact in artifacts:
            self.state.transition(
                artifact.book.id,
                BookStatus.PUBLISHED,
                published_revision=result.revision,
            )
            self.cleanup(artifact)
        logger.info("Published revision %s", result.revision)
        return result

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
