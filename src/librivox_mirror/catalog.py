from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from librivox_mirror.models import Author, Book, Genre, Reader, Section, canonical_metadata_json

CATALOG_URL = "https://librivox.org/api/feed/audiobooks"
logger = logging.getLogger(__name__)


class BookNotFoundError(LookupError):
    def __init__(self, book_id: int) -> None:
        super().__init__(f"LibriVox book {book_id} was not found")
        self.book_id = book_id


class LibriVoxCatalog:
    def __init__(
        self,
        *,
        user_agent: str,
        timeout: float = 30,
        client: httpx.Client | None = None,
    ) -> None:
        self._owns_client = client is None
        self._client = client or httpx.Client(
            headers={"User-Agent": user_agent},
            timeout=timeout,
            follow_redirects=True,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> LibriVoxCatalog:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def get_book(self, book_id: int) -> Book:
        payload = self._request({"id": book_id, "extended": 1})
        books = payload.get("books") or []
        if not books:
            raise BookNotFoundError(book_id)
        return parse_book(books[0])

    def iter_books(
        self,
        *,
        since: int | None = None,
        start_id: int | None = None,
        end_id: int | None = None,
        page_size: int = 50,
    ) -> Iterator[Book]:
        offset = 0
        while True:
            if offset == 0 or offset % 1000 == 0:
                logger.info("Scanning LibriVox catalog at offset %s", offset)
            params: dict[str, str | int] = {
                "extended": 1,
                "limit": page_size,
                "offset": offset,
            }
            if since is not None:
                params["since"] = since
            payload = self._request(params)
            rows = payload.get("books") or []
            for row in rows:
                book = parse_book(row)
                if start_id is not None and book.id < start_id:
                    continue
                if end_id is not None and book.id > end_id:
                    return
                yield book
            if len(rows) < page_size:
                return
            offset += page_size

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=1, max=30),
        reraise=True,
    )
    def _request(self, params: Mapping[str, str | int]) -> dict[str, Any]:
        response = self._client.get(CATALOG_URL, params={**params, "format": "json"})
        response.raise_for_status()
        return response.json()


def parse_book(row: Mapping[str, Any]) -> Book:
    book_id = _required_int(row.get("id"), "book id")
    sections = tuple(parse_section(section, book_id) for section in row.get("sections") or ())
    authors = tuple(parse_author(author) for author in row.get("authors") or ())
    translators = tuple(parse_author(author) for author in row.get("translators") or ())
    genres = tuple(parse_genre(genre) for genre in row.get("genres") or ())
    return Book(
        id=book_id,
        title=_text(row.get("title")),
        description=_text(row.get("description")),
        language=_optional_text(row.get("language")),
        copyright_year=_optional_text(row.get("copyright_year")),
        total_time_seconds=_optional_int(row.get("totaltimesecs")),
        url_librivox=_text(row.get("url_librivox")),
        url_iarchive=_text(row.get("url_iarchive")),
        url_project=_optional_text(row.get("url_project")),
        url_rss=_optional_text(row.get("url_rss")),
        url_text_source=_optional_text(row.get("url_text_source")),
        url_other=_optional_text(row.get("url_other")),
        url_zip_file=_optional_text(row.get("url_zip_file")),
        authors=authors,
        translators=translators,
        genres=genres,
        sections=sections,
        source_metadata_json=canonical_metadata_json(row),
    )


def parse_section(row: Mapping[str, Any], book_id: int) -> Section:
    readers = tuple(parse_reader(reader) for reader in row.get("readers") or ())
    return Section(
        id=_required_int(row.get("id"), "section id"),
        book_id=book_id,
        section_number=_required_int(row.get("section_number"), "section number"),
        title=_text(row.get("title")),
        language=_optional_text(row.get("language")),
        duration_seconds=_optional_int(row.get("playtime")),
        file_name=_optional_text(row.get("file_name")),
        listen_url=_text(row.get("listen_url")),
        readers=readers,
        source_metadata_json=canonical_metadata_json(row),
    )


def parse_author(row: Mapping[str, Any]) -> Author:
    return Author(
        id=_optional_int(row.get("id")),
        first_name=_text(row.get("first_name")),
        last_name=_text(row.get("last_name")),
        dob=_optional_text(row.get("dob")),
        dod=_optional_text(row.get("dod")),
    )


def parse_reader(row: Mapping[str, Any]) -> Reader:
    return Reader(
        id=_optional_int(row.get("reader_id") or row.get("id")),
        display_name=_text(row.get("display_name")),
        url_text=_optional_text(row.get("url_text")),
    )


def parse_genre(row: Mapping[str, Any]) -> Genre:
    return Genre(
        id=_optional_int(row.get("id")),
        name=_text(row.get("name")),
    )


def _required_int(value: object, name: str) -> int:
    converted = _optional_int(value)
    if converted is None:
        raise ValueError(f"missing {name}")
    return converted


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(str(value))


def _text(value: object) -> str:
    return str(value or "").strip()


def _optional_text(value: object) -> str | None:
    text = _text(value)
    return text or None
