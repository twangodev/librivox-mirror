import json

import httpx
import respx

from librivox_mirror.catalog import CATALOG_URL, LibriVoxCatalog


def catalog_payload() -> dict[str, list[dict[str, object]]]:
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


@respx.mock
def test_iter_books_seeks_to_a_missing_start_id() -> None:
    book_ids = [book_id for book_id in range(1, 3501) if book_id % 10]
    page_offsets = []

    def response(request: httpx.Request) -> httpx.Response:
        if requested_id := request.url.params.get("id"):
            selected = [int(requested_id)] if int(requested_id) in book_ids else []
            offset = None
        else:
            offset = int(request.url.params["offset"])
            limit = int(request.url.params["limit"])
            selected = book_ids[offset : offset + limit]
        extended = request.url.params.get("extended") == "1"
        if extended and offset is not None:
            page_offsets.append(offset)
        rows = []
        for book_id in selected:
            if extended:
                row = dict(catalog_payload()["books"][0])
                row["id"] = str(book_id)
            else:
                row = {"id": str(book_id)}
            rows.append(row)
        return httpx.Response(200, json={"books": rows})

    route = respx.get(CATALOG_URL).mock(side_effect=response)

    with LibriVoxCatalog(user_agent="test") as catalog:
        books = list(catalog.iter_books(start_id=3010, end_id=3012))

    assert [book.id for book in books] == [3011, 3012]
    assert page_offsets == [2710]
    assert route.call_count == 3


@respx.mock
def test_iter_books_seeks_past_the_catalog() -> None:
    book_ids = [1, 3, 7, 15]
    page_offsets = []

    def response(request: httpx.Request) -> httpx.Response:
        offset = int(request.url.params["offset"])
        limit = int(request.url.params["limit"])
        selected = book_ids[offset : offset + limit]
        if request.url.params.get("extended") == "1":
            page_offsets.append(offset)
        return httpx.Response(
            200,
            json={"books": [{"id": str(book_id)} for book_id in selected]},
        )

    respx.get(CATALOG_URL).mock(side_effect=response)

    with LibriVoxCatalog(user_agent="test") as catalog:
        books = list(catalog.iter_books(start_id=100))

    assert books == []
    assert page_offsets == []
