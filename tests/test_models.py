from librivox_mirror.models import Author, Book, Reader, Section, SyncState


def make_book() -> Book:
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
                listen_url="https://archive.org/download/a_test_book/chapter_64kb.mp3",
                readers=(Reader(id=2, display_name="Reader"),),
            ),
        ),
    )


def test_book_computed_fields_are_stable() -> None:
    book = make_book()

    assert book.metadata_bucket == 0
    assert 0 <= book.hash_partition < 100
    assert book.sections[0].sample_key == "000047-00000091"
    assert book.source_fingerprint == make_book().source_fingerprint


def test_source_fingerprint_changes_with_source_metadata() -> None:
    book = make_book()

    assert (
        book.source_fingerprint != book.model_copy(update={"title": "Changed"}).source_fingerprint
    )


def test_sync_state_serializes_derived_audio_hours() -> None:
    state = SyncState(audio_seconds_by_language={"English": 36_000, "French": 9_000})

    assert state.audio_hours == 12.5
    assert state.model_dump(mode="json")["audio_hours"] == 12.5
