from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from librivox_mirror.models import Book, BookStatus, QuarantineRecord

STATUS_ORDER = {
    BookStatus.DISCOVERED: 0,
    BookStatus.RESOLVED: 1,
    BookStatus.DOWNLOADED: 2,
    BookStatus.VERIFIED: 3,
    BookStatus.PACKED: 4,
    BookStatus.PUBLISHED: 5,
}


@dataclass(frozen=True)
class BookCheckpoint:
    book_id: int
    source_fingerprint: str
    status: BookStatus
    archive_identifier: str | None
    artifact_path: Path | None
    artifact_sha256: str | None
    error_code: str | None
    error_detail: str | None
    published_revision: str | None
    updated_at: datetime


class StateStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._connection = sqlite3.connect(path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS books (
                book_id INTEGER PRIMARY KEY,
                source_fingerprint TEXT NOT NULL,
                status TEXT NOT NULL,
                archive_identifier TEXT,
                artifact_path TEXT,
                artifact_sha256 TEXT,
                error_code TEXT,
                error_detail TEXT,
                published_revision TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> StateStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def get(self, book_id: int) -> BookCheckpoint | None:
        row = self._connection.execute(
            "SELECT * FROM books WHERE book_id = ?",
            (book_id,),
        ).fetchone()
        return checkpoint(row) if row else None

    def discover(self, book: Book) -> BookCheckpoint:
        now = utc_now()
        self._connection.execute(
            """
            INSERT INTO books (book_id, source_fingerprint, status, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(book_id) DO UPDATE SET
                source_fingerprint = excluded.source_fingerprint,
                status = CASE
                    WHEN books.source_fingerprint != excluded.source_fingerprint
                    THEN excluded.status ELSE books.status
                END,
                archive_identifier = CASE
                    WHEN books.source_fingerprint != excluded.source_fingerprint
                    THEN NULL ELSE books.archive_identifier
                END,
                artifact_path = CASE
                    WHEN books.source_fingerprint != excluded.source_fingerprint
                    THEN NULL ELSE books.artifact_path
                END,
                artifact_sha256 = CASE
                    WHEN books.source_fingerprint != excluded.source_fingerprint
                    THEN NULL ELSE books.artifact_sha256
                END,
                error_code = CASE
                    WHEN books.source_fingerprint != excluded.source_fingerprint
                    THEN NULL ELSE books.error_code
                END,
                error_detail = CASE
                    WHEN books.source_fingerprint != excluded.source_fingerprint
                    THEN NULL ELSE books.error_detail
                END,
                published_revision = CASE
                    WHEN books.source_fingerprint != excluded.source_fingerprint
                    THEN NULL ELSE books.published_revision
                END,
                updated_at = CASE
                    WHEN books.source_fingerprint != excluded.source_fingerprint
                    THEN excluded.updated_at ELSE books.updated_at
                END
            """,
            (book.id, book.source_fingerprint, BookStatus.DISCOVERED, now),
        )
        self._connection.commit()
        discovered = self.get(book.id)
        if discovered is None:
            raise RuntimeError(f"failed to persist book {book.id}")
        return discovered

    def transition(
        self,
        book_id: int,
        status: BookStatus,
        *,
        archive_identifier: str | None = None,
        artifact_path: Path | None = None,
        artifact_sha256: str | None = None,
        published_revision: str | None = None,
    ) -> BookCheckpoint:
        current = self.get(book_id)
        if current is None:
            raise KeyError(f"book {book_id} has not been discovered")
        if status == BookStatus.QUARANTINED:
            raise ValueError("use quarantine() to store a quarantine record")
        current_order = STATUS_ORDER.get(current.status, -1)
        if current.status != BookStatus.QUARANTINED and STATUS_ORDER[status] < current_order:
            raise ValueError(f"cannot move book {book_id} from {current.status} back to {status}")
        self._connection.execute(
            """
            UPDATE books SET
                status = ?,
                archive_identifier = COALESCE(?, archive_identifier),
                artifact_path = COALESCE(?, artifact_path),
                artifact_sha256 = COALESCE(?, artifact_sha256),
                published_revision = COALESCE(?, published_revision),
                error_code = NULL,
                error_detail = NULL,
                updated_at = ?
            WHERE book_id = ?
            """,
            (
                status,
                archive_identifier,
                str(artifact_path) if artifact_path else None,
                artifact_sha256,
                published_revision,
                utc_now(),
                book_id,
            ),
        )
        self._connection.commit()
        transitioned = self.get(book_id)
        if transitioned is None:
            raise RuntimeError(f"failed to transition book {book_id}")
        return transitioned

    def quarantine(self, record: QuarantineRecord) -> BookCheckpoint:
        self._connection.execute(
            """
            INSERT INTO books (
                book_id, source_fingerprint, status, archive_identifier,
                error_code, error_detail, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(book_id) DO UPDATE SET
                source_fingerprint = excluded.source_fingerprint,
                status = excluded.status,
                archive_identifier = excluded.archive_identifier,
                artifact_path = NULL,
                artifact_sha256 = NULL,
                error_code = excluded.error_code,
                error_detail = excluded.error_detail,
                published_revision = NULL,
                updated_at = excluded.updated_at
            """,
            (
                record.book_id,
                record.source_fingerprint,
                BookStatus.QUARANTINED,
                record.archive_identifier,
                record.code,
                record.detail,
                record.observed_at.isoformat(),
            ),
        )
        self._connection.commit()
        quarantined = self.get(record.book_id)
        if quarantined is None:
            raise RuntimeError(f"failed to quarantine book {record.book_id}")
        return quarantined

    def list(self, *statuses: BookStatus) -> tuple[BookCheckpoint, ...]:
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            rows = self._connection.execute(
                f"SELECT * FROM books WHERE status IN ({placeholders}) ORDER BY book_id",
                tuple(statuses),
            ).fetchall()
        else:
            rows = self._connection.execute("SELECT * FROM books ORDER BY book_id").fetchall()
        return tuple(checkpoint(row) for row in rows)


def checkpoint(row: sqlite3.Row) -> BookCheckpoint:
    return BookCheckpoint(
        book_id=row["book_id"],
        source_fingerprint=row["source_fingerprint"],
        status=BookStatus(row["status"]),
        archive_identifier=row["archive_identifier"],
        artifact_path=Path(row["artifact_path"]) if row["artifact_path"] else None,
        artifact_sha256=row["artifact_sha256"],
        error_code=row["error_code"],
        error_detail=row["error_detail"],
        published_revision=row["published_revision"],
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def utc_now() -> str:
    return datetime.now(UTC).isoformat()
