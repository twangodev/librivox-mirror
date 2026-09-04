from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from librivox_mirror.catalog import LibriVoxCatalog
from librivox_mirror.models import Book, SyncState

CATALOG_OVERLAP_SECONDS = 48 * 60 * 60


class BookIndex(Protocol):
    def has_current_book(self, book: Book) -> bool: ...


@dataclass(frozen=True)
class CatalogBatch:
    """Selection from a catalog query; exhaustion alone does not mean all work fits."""

    books: list[Book]
    scanned_count: int
    reached_catalog_end: bool
    all_pending_selected: bool


def select_pending_batch(books: Iterable[Book], index: BookIndex, max_books: int) -> CatalogBatch:
    """Stop at the pending-book limit; reaching the limit does not prove exhaustion."""
    selected = []
    seen = set()
    for book in books:
        if book.id in seen:
            continue
        seen.add(book.id)
        if index.has_current_book(book):
            continue
        selected.append(book)
        if len(selected) == max_books:
            return CatalogBatch(
                books=selected,
                scanned_count=len(seen),
                reached_catalog_end=False,
                all_pending_selected=False,
            )
    return CatalogBatch(
        books=selected,
        scanned_count=len(seen),
        reached_catalog_end=True,
        all_pending_selected=True,
    )


def select_catalog_catchup(
    catalog: LibriVoxCatalog, index: BookIndex, state: SyncState, max_books: int
) -> CatalogBatch:
    after_id = state.catalog_scan_after_book_id
    start_id = after_id + 1 if after_id is not None else None
    return select_pending_batch(catalog.iter_books(since=0, start_id=start_id), index, max_books)


def select_incremental_sync(
    catalog: LibriVoxCatalog, index: BookIndex, state: SyncState, max_books: int
) -> CatalogBatch:
    since = max(0, state.catalog_watermark - CATALOG_OVERLAP_SECONDS)
    candidates = sorted(
        {book.id: book for book in catalog.iter_books(since=since)}.values(),
        key=lambda book: book.id,
    )
    pending = [book for book in candidates if not index.has_current_book(book)]
    return CatalogBatch(
        books=pending[:max_books],
        scanned_count=len(candidates),
        reached_catalog_end=True,
        all_pending_selected=len(pending) <= max_books,
    )
