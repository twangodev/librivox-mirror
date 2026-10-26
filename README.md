# librivox-mirror

[![CI](https://img.shields.io/github/actions/workflow/status/twangodev/librivox-mirror/python.yml?branch=main&style=flat&logo=github&logoColor=white&label=CI)](https://github.com/twangodev/librivox-mirror/actions/workflows/python.yml)
[![PyPI](https://img.shields.io/pypi/v/librivox-mirror?style=flat&logo=pypi&logoColor=white)](https://pypi.org/project/librivox-mirror/)
[![Mirrored books](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fdatasets%2Ftwangodev%2Flibrivox-mirror%2Fresolve%2Fmain%2Fstate%2Fsync.json&query=%24.published_books&label=books&style=flat&logo=bookstack&logoColor=white&color=FFD21E&cacheSeconds=300)](https://huggingface.co/datasets/twangodev/librivox-mirror)
[![Audio hours](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fdatasets%2Ftwangodev%2Flibrivox-mirror%2Fresolve%2Fmain%2Fstate%2Fsync.json&query=%24.audio_hours&suffix=%20hours&label=audio&style=flat&logo=audacity&logoColor=white&color=FFD21E&cacheSeconds=300)](https://huggingface.co/datasets/twangodev/librivox-mirror)
[![Last updated](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fdatasets%2Ftwangodev%2Flibrivox-mirror%2Fresolve%2Fmain%2Fstate%2Fsync.json&query=%24.updated_at&label=updated&style=flat&logo=huggingface&logoColor=white&color=0969DA&cacheSeconds=300)](https://huggingface.co/datasets/twangodev/librivox-mirror)

Fast, structured, continuously updated LibriVox audio mirror.

Original MP3s are stored as WebDataset TARs with Parquet indexes.

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

Set `HF_DATASET_REPO` and `HF_TOKEN` to publish. Backfills resume from local SQLite
checkpoints.

## Dataset

- `preview` (default): browser-playable original MP3 samples
- `sections`: one Parquet row per audio section
- `books`: one Parquet row per book
- `data/`: streaming WebDataset audio and sample metadata

Audio is unmodified; source metadata and checksums are preserved.

## Licenses

Code is MIT. Mirror-specific curation, indexes, metadata, and documentation are CC
BY 4.0. Original LibriVox audio remains public domain in the United States.
