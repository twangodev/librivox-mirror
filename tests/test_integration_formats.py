import hashlib
import io
import json

import pytest

from librivox_mirror.archive import resolve_original_files
from librivox_mirror.artifact import build_artifact
from librivox_mirror.hub import BOOK_SCHEMA, SECTION_SCHEMA
from librivox_mirror.models import DownloadedSection

datasets = pytest.importorskip("datasets")
soundfile = pytest.importorskip("soundfile")
webdataset = pytest.importorskip("webdataset")


@pytest.mark.integration
def test_metadata_schemas_are_native_hugging_face_features() -> None:
    books = datasets.Features.from_arrow_schema(BOOK_SCHEMA)
    sections = datasets.Features.from_arrow_schema(SECTION_SCHEMA)

    assert isinstance(books["authors"], datasets.List)
    assert isinstance(books["librivox_metadata"], datasets.Json)
    assert isinstance(sections["readers"], datasets.List)
    assert isinstance(sections["archive_file_metadata"], datasets.Json)


@pytest.mark.integration
def test_artifact_loads_with_datasets_and_webdataset(book, tmp_path) -> None:
    content = b"".join(b"\xff\xfb\x90\x64" + bytes(413) for _ in range(20))
    resolved = resolve_original_files(
        book,
        "a_test_book",
        [
            {
                "name": "chapter_64kb.mp3",
                "source": "derivative",
                "format": "64Kbps MP3",
                "original": "chapter.mp3",
            },
            {
                "name": "chapter.mp3",
                "source": "original",
                "format": "VBR MP3",
                "size": str(len(content)),
            },
        ],
    )
    audio = tmp_path / "chapter.mp3"
    audio.write_bytes(content)
    artifact = build_artifact(
        resolved,
        [
            DownloadedSection(
                resolved=resolved.sections[0],
                path=audio,
                sha256=hashlib.sha256(content).hexdigest(),
            )
        ],
        tmp_path / "repository",
    )

    sample = next(iter(webdataset.WebDataset(str(artifact.path), shardshuffle=False)))
    assert sample["__key__"] == "000047-00000091"
    assert sample["mp3"] == content
    assert json.loads(sample["json"])["book_id"] == 47

    dataset = datasets.load_dataset(
        "webdataset",
        data_files={"train": [str(artifact.path)]},
        split="train",
        cache_dir=tmp_path / "hf-cache",
        features=datasets.Features({"mp3": datasets.Value("binary")}),
    )
    record = dataset[0]
    assert dataset.num_rows == 1
    samples, sample_rate = soundfile.read(io.BytesIO(record["mp3"]))
    assert samples.size > 0
    assert sample_rate == 44_100
