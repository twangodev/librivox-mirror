from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from queue import Full, Queue
from threading import Event, Thread
from typing import cast

_COMPLETE = object()


@contextmanager
def prefetch[Item](source: Iterable[Item], *, capacity: int) -> Iterator[Iterator[Item]]:
    if capacity < 1:
        raise ValueError("prefetch capacity must be positive")

    queue: Queue[Item | object] = Queue(maxsize=capacity)
    stopped = Event()
    failures: list[BaseException] = []

    def enqueue(item: Item | object) -> bool:
        while not stopped.is_set():
            try:
                queue.put(item, timeout=0.1)
            except Full:
                continue
            return True
        return False

    def produce() -> None:
        try:
            for item in source:
                if not enqueue(item):
                    return
        except BaseException as error:
            failures.append(error)
        enqueue(_COMPLETE)

    def consume() -> Iterator[Item]:
        while True:
            item = queue.get()
            if item is _COMPLETE:
                if failures:
                    raise failures[0]
                return
            yield cast(Item, item)

    producer = Thread(target=produce, name="catalog-prefetch", daemon=True)
    producer.start()
    try:
        yield consume()
    finally:
        stopped.set()
        producer.join()
