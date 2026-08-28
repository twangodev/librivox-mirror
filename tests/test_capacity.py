from threading import Event, Thread

from librivox_mirror.capacity import StagingCapacity


def test_staging_capacity_blocks_until_space_is_released(tmp_path) -> None:
    capacity = StagingCapacity(tmp_path, max_bytes=100, minimum_free_bytes=0)
    waiting = Event()
    acquired = Event()
    capacity.reserve(1, 80)

    def reserve() -> None:
        waiting.set()
        capacity.reserve(2, 40)
        acquired.set()

    worker = Thread(target=reserve)
    worker.start()
    assert waiting.wait(timeout=1)
    assert not acquired.wait(timeout=0.05)

    capacity.release(1)

    assert acquired.wait(timeout=1)
    worker.join(timeout=1)
    assert capacity.reserved_bytes == 40


def test_staging_capacity_allows_one_oversized_book(tmp_path) -> None:
    capacity = StagingCapacity(tmp_path, max_bytes=100, minimum_free_bytes=0)

    capacity.reserve(1, 120)

    assert capacity.reserved_bytes == 120


def test_staging_capacity_tracks_packed_artifact_size(tmp_path) -> None:
    capacity = StagingCapacity(tmp_path, max_bytes=100, minimum_free_bytes=0)
    capacity.reserve(1, 80)

    capacity.resize(1, 30)
    capacity.release(1)

    assert capacity.reserved_bytes == 0
