# librivox-mirror

[![CI](https://img.shields.io/github/actions/workflow/status/twangodev/librivox-mirror/python.yml?branch=main&style=flat-square&logo=githubactions&logoColor=white&label=CI)](https://github.com/twangodev/librivox-mirror/actions/workflows/python.yml)
[![PyPI](https://img.shields.io/pypi/v/librivox-mirror?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/librivox-mirror/)
[![Mirrored books](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fdatasets%2Ftwangodev%2Flibrivox-mirror%2Fresolve%2Fmain%2Fstate%2Fsync.json&query=%24.published_books&label=books&style=flat-square&logo=huggingface&logoColor=000&color=FFD21E&cacheSeconds=300)](https://huggingface.co/datasets/twangodev/librivox-mirror)

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

- `preview` (default): browser-playable original MP3 samples
- `sections`: one Parquet row per audio section
- `books`: one Parquet row per book
- `data/`: streaming WebDataset audio and sample metadata

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
