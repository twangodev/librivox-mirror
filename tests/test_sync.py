from unittest.mock import Mock

import pytest

from librivox_mirror.catalog import LibriVoxCatalog
from librivox_mirror.models import Book, SyncState
from librivox_mirror.sync import CATALOG_OVERLAP_SECONDS, select_incremental_sync


@pytest.mark.parametrize("max_books", [1, 2])
def test_incremental_sync_distinguishes_catalog_exhaustion_from_batch_completion(
    book: Book, max_books: int
) -> None:
    newer = book.model_copy(update={"id": book.id + 1})
    current = book.model_copy(update={"id": book.id + 2})
    catalog = Mock(spec=LibriVoxCatalog)
    catalog.iter_books.return_value = iter([newer, book, current, book])
    index = Mock()
    index.has_current_book.side_effect = lambda candidate: candidate.id == current.id
    state = SyncState(catalog_watermark=CATALOG_OVERLAP_SECONDS + 100)

    batch = select_incremental_sync(catalog, index, state, max_books)

    catalog.iter_books.assert_called_once_with(since=100)
    assert [candidate.id for candidate in batch.books] == [book.id, newer.id][:max_books]
    assert batch.scanned_count == 3
    assert batch.reached_catalog_end is True
    assert batch.all_pending_selected is (max_books == 2)
