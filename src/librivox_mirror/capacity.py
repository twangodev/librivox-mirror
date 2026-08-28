from __future__ import annotations

import logging
import shutil
from pathlib import Path
from threading import Condition

logger = logging.getLogger(__name__)


class InsufficientStagingSpaceError(OSError):
    pass


class StagingCapacity:
    def __init__(
        self,
        path: Path,
        *,
        max_bytes: int,
        minimum_free_bytes: int,
    ) -> None:
        if max_bytes < 1:
            raise ValueError("maximum staging bytes must be positive")
        if minimum_free_bytes < 0:
            raise ValueError("minimum free bytes cannot be negative")
        path.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.max_bytes = max_bytes
        self.minimum_free_bytes = minimum_free_bytes
        self._condition = Condition()
        self._reservations: dict[int, int] = {}

    @property
    def reserved_bytes(self) -> int:
        with self._condition:
            return sum(self._reservations.values())

    def reserve(self, book_id: int, size: int, *, existing: bool = False) -> None:
        if size < 0:
            raise ValueError("staging reservation cannot be negative")
        requested = max(size, 1)
        logged_wait = False
        with self._condition:
            while True:
                other_bytes = self._reserved_by_other_books(book_id)
                available = shutil.disk_usage(self.path).free - self.minimum_free_bytes
                has_disk_capacity = existing or requested <= available
                if self._fits(requested, other_bytes) and has_disk_capacity:
                    break
                if not has_disk_capacity and not other_bytes:
                    raise InsufficientStagingSpaceError(
                        f"book {book_id} needs {requested / 1024**3:.1f} GiB of staging "
                        f"capacity but only {max(available, 0) / 1024**3:.1f} GiB is available"
                    )
                if not logged_wait:
                    logger.info(
                        "Book %s waiting for staging capacity (%.1f/%.1f GiB reserved)",
                        book_id,
                        sum(self._reservations.values()) / 1024**3,
                        self.max_bytes / 1024**3,
                    )
                    logged_wait = True
                self._condition.wait(timeout=30)
            self._reservations[book_id] = requested

    def resize(self, book_id: int, size: int) -> None:
        if size < 0:
            raise ValueError("staging reservation cannot be negative")
        requested = max(size, 1)
        with self._condition:
            if book_id not in self._reservations:
                raise KeyError(f"book {book_id} has no staging reservation")
            other_bytes = self._reserved_by_other_books(book_id)
            if requested > self._reservations[book_id] and not self._fits(requested, other_bytes):
                raise InsufficientStagingSpaceError(
                    f"book {book_id} exceeded its staging reservation"
                )
            self._reservations[book_id] = requested
            self._condition.notify_all()

    def release(self, book_id: int) -> None:
        with self._condition:
            if self._reservations.pop(book_id, None) is not None:
                self._condition.notify_all()

    def _reserved_by_other_books(self, book_id: int) -> int:
        return sum(
            size
            for reserved_book_id, size in self._reservations.items()
            if reserved_book_id != book_id
        )

    def _fits(self, requested: int, other_bytes: int) -> bool:
        return not other_bytes or other_bytes + requested <= self.max_bytes
