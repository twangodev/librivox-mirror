import os

import pytest

from librivox_mirror.archive import InternetArchiveClient
from librivox_mirror.artifact import verify_mp3
from librivox_mirror.catalog import LibriVoxCatalog


@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("RUN_LIBRIVOX_LIVE_TESTS") != "1",
    reason="set RUN_LIBRIVOX_LIVE_TESTS=1 to access live source services",
)
def test_one_live_original_mp3(tmp_path) -> None:
    user_agent = os.environ.get(
        "LIBRIVOX_MIRROR_USER_AGENT",
        "librivox-mirror live test (https://pypi.org/project/librivox-mirror/)",
    )
    book_id = int(os.environ.get("LIBRIVOX_LIVE_BOOK_ID", "1"))
    with (
        LibriVoxCatalog(user_agent=user_agent) as catalog,
        InternetArchiveClient(user_agent=user_agent, request_delay=1) as archive,
    ):
        resolved = archive.resolve_book(catalog.get_book(book_id))
        downloaded = archive.download_section(
            resolved.archive_identifier,
            resolved.sections[0],
            tmp_path,
        )

    verify_mp3(downloaded.path)
    assert downloaded.path.stat().st_size == downloaded.resolved.archive_file.size
