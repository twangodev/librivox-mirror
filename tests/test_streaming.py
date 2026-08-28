import time
from threading import Event

import pytest

from librivox_mirror.streaming import ordered_parallel_map, prefetch


def test_parallel_map_preserves_source_order() -> None:
    def transform(item: int) -> int:
        time.sleep((3 - item) * 0.001)
        return item * 2

    results = ordered_parallel_map(range(4), transform, workers=4, capacity=4)

    assert list(results) == [0, 2, 4, 6]


def test_parallel_map_continues_while_the_consumer_is_paused() -> None:
    later_work_started = Event()
    release = Event()

    def transform(item: int) -> int:
        if item >= 2:
            later_work_started.set()
        if item > 0:
            release.wait(timeout=1)
        return item

    results = ordered_parallel_map(range(4), transform, workers=2, capacity=4)
    assert next(results) == 0
    assert later_work_started.wait(timeout=1)
    release.set()
    assert list(results) == [1, 2, 3]


def test_parallel_map_rejects_an_undersized_buffer() -> None:
    with pytest.raises(ValueError, match="at least"):
        list(ordered_parallel_map([], lambda item: item, workers=2, capacity=1))


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
