from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from queue import Full, Queue
from threading import Event, Thread
from typing import cast

_COMPLETE = object()


def ordered_parallel_map[Source, Result](
    source: Iterable[Source],
    function: Callable[[Source], Result],
    *,
    workers: int,
    capacity: int,
) -> Iterator[Result]:
    if workers < 1:
        raise ValueError("worker count must be positive")
    if capacity < workers:
        raise ValueError("capacity must be at least the worker count")

    items = iter(source)
    pending: deque[Future[Result]] = deque()
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="book-preparer")
    try:
        for _ in range(capacity):
            try:
                item = next(items)
            except StopIteration:
                break
            pending.append(executor.submit(function, item))

        while pending:
            result = pending.popleft().result()
            try:
                item = next(items)
            except StopIteration:
                pass
            else:
                pending.append(executor.submit(function, item))
            yield result
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


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
