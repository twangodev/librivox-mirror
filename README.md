# librivox-mirror

Fast, structured, continuously updated LibriVox audio mirror.

Original MP3s are mirrored to Hugging Face as deterministic WebDataset TARs with
compact Parquet indexes. Dedicated infrastructure handles the initial backfill;
GitHub Actions handles updates.

## Install

```console
uvx librivox-mirror --help
```

## Use

```console
librivox-mirror plan
librivox-mirror mirror 47
librivox-mirror backfill --start-id 1 --end-id 1000
```

Set `HF_DATASET_REPO` and `HF_TOKEN` to publish. Local SQLite checkpoints make
backfills resumable.

## Dataset

- `sections` (default): one Parquet row per audio section
- `books`: one Parquet row per book
- `data/**/*.tar`: streaming WebDataset audio and sample metadata

Audio is never transcoded. Complete LibriVox and Internet Archive metadata and
checksums are preserved.

## Develop

```console
uv sync --locked --group dev --no-group integration
```

Python 3.12–3.14 is supported. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Licenses

Code is MIT. Mirror-specific curation, indexes, metadata, and documentation are CC
BY 4.0. Original LibriVox audio remains public domain in the United States.
