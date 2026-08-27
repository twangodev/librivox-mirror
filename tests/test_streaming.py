from threading import Event

import pytest

from librivox_mirror.streaming import prefetch


def test_prefetch_streams_items_before_the_source_finishes() -> None:
    release_source = Event()

    def source():
        yield 1
        release_source.wait(timeout=1)
        yield 2

    with prefetch(source(), capacity=1) as items:
        assert next(items) == 1
        release_source.set()
        assert list(items) == [2]


def test_prefetch_propagates_producer_failures() -> None:
    def source():
        yield 1
        raise RuntimeError("catalog unavailable")

    with prefetch(source(), capacity=1) as items:
        assert next(items) == 1
        with pytest.raises(RuntimeError, match="catalog unavailable"):
            next(items)


def test_prefetch_rejects_an_empty_buffer() -> None:
    with pytest.raises(ValueError, match="positive"), prefetch([], capacity=0):
        pass
