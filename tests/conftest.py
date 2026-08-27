import json

import pytest

from librivox_mirror.models import Author, Book, Reader, Section


@pytest.fixture
def book() -> Book:
    section_payload = {
        "id": "91",
        "section_number": "1",
        "title": "Chapter One",
        "listen_url": "https://archive.org/download/a_test_book/chapter_64kb.mp3",
        "playtime": "1",
        "future_section_field": {"preserved": True},
    }
    book_payload = {
        "id": "47",
        "title": "A Test Book",
        "future_book_field": ["preserved"],
        "sections": [section_payload],
    }
    return Book(
        id=47,
        title="A Test Book",
        language="English",
        url_librivox="https://librivox.org/a-test-book/",
        url_iarchive="https://archive.org/details/a_test_book",
        authors=(Author(id=1, first_name="Ada", last_name="Lovelace"),),
        sections=(
            Section(
                id=91,
                book_id=47,
                section_number=1,
                title="Chapter One",
                language="English",
                duration_seconds=1,
                listen_url="https://archive.org/download/a_test_book/chapter_64kb.mp3",
                readers=(Reader(id=2, display_name="Reader"),),
                source_metadata_json=json.dumps(section_payload, sort_keys=True),
            ),
        ),
        source_metadata_json=json.dumps(book_payload, sort_keys=True),
    )
