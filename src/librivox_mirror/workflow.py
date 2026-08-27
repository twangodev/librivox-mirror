from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from librivox_mirror.archive import InternetArchiveClient, QuarantinedBookError
from librivox_mirror.artifact import build_artifact, verify_mp3
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
        if checkpoint.status == BookStatus.PUBLISHED:
            self.cleanup_paths(book.id, checkpoint.artifact_path)
            logger.info("Book %s is already published", book.id)
            return BookOutcome(book=book, skipped=True)
        if self.publisher and self.publisher.has_current_book(book):
            self.cleanup_paths(book.id, checkpoint.artifact_path)
            self.state.transition(book.id, BookStatus.PUBLISHED)
            logger.info("Book %s already matches the Hub", book.id)
            return BookOutcome(book=book, skipped=True)

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
        self.state.transition(
            book.id,
            BookStatus.PACKED,
            artifact_path=artifact.path,
            artifact_sha256=artifact.sha256,
        )
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
        download_directory = self.staging_directory / "downloads" / f"{book_id:06d}"
        shutil.rmtree(download_directory, ignore_errors=True)
        if artifact_path:
            artifact_path.unlink(missing_ok=True)
