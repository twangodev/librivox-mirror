import json

import httpx
import respx

from librivox_mirror.catalog import CATALOG_URL, LibriVoxCatalog


def catalog_payload() -> dict[str, object]:
    return {
        "books": [
            {
                "id": "47",
                "title": "A Test Book",
                "description": "Description",
                "language": "English",
                "copyright_year": "1901",
                "totaltimesecs": 12,
                "url_librivox": "https://librivox.org/a-test-book/",
                "url_iarchive": "https://archive.org/details/a_test_book",
                "url_other": "https://example.com/book",
                "url_zip_file": "https://archive.org/download/a_test_book/book.zip",
                "future_book_field": {"preserved": True},
                "authors": [
                    {
                        "id": "1",
                        "first_name": "Ada",
                        "last_name": "Lovelace",
                        "dob": "1815",
                        "dod": "1852",
                    }
                ],
                "translators": [
                    {
                        "id": "3",
                        "first_name": "Grace",
                        "last_name": "Hopper",
                        "dob": "1906",
                        "dod": "1992",
                    }
                ],
                "genres": [{"id": "9", "name": "Fiction"}],
                "sections": [
                    {
                        "id": "91",
                        "section_number": "1",
                        "title": "Chapter One",
                        "language": "English",
                        "playtime": "12",
                        "file_name": "chapter.mp3",
                        "listen_url": ("https://archive.org/download/a_test_book/chapter_64kb.mp3"),
                        "future_section_field": [1, 2, 3],
                        "readers": [{"reader_id": "2", "display_name": "Reader"}],
                    }
                ],
            }
        ]
    }


@respx.mock
def test_get_book_parses_normalized_fields_and_preserves_source_metadata() -> None:
    route = respx.get(CATALOG_URL).mock(return_value=httpx.Response(200, json=catalog_payload()))

    with LibriVoxCatalog(user_agent="test") as catalog:
        book = catalog.get_book(47)

    assert route.called
    assert book.id == 47
    assert book.authors[0].last_name == "Lovelace"
    assert book.translators[0].last_name == "Hopper"
    assert book.genres[0].name == "Fiction"
    assert book.url_zip_file == "https://archive.org/download/a_test_book/book.zip"
    assert book.sections[0].duration_seconds == 12
    assert book.sections[0].file_name == "chapter.mp3"
    assert json.loads(book.source_metadata_json)["future_book_field"] == {"preserved": True}
    assert json.loads(book.sections[0].source_metadata_json)["future_section_field"] == [1, 2, 3]


@respx.mock
def test_iter_books_paginates_until_a_short_page() -> None:
    route = respx.get(CATALOG_URL).mock(
        side_effect=[
            httpx.Response(200, json=catalog_payload()),
            httpx.Response(200, json={"books": []}),
        ]
    )

    with LibriVoxCatalog(user_agent="test") as catalog:
        books = list(catalog.iter_books(page_size=1))

    assert [book.id for book in books] == [47]
    assert route.call_count == 2
