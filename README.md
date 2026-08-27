# librivox-mirror

Fast, structured, continuously updated LibriVox audio mirror.

Original MP3 bytes are stored in deterministic, one-book WebDataset TARs. Compact
Parquet indexes make the catalog easy to browse without downloading the audio. The
initial backfill runs on dedicated infrastructure; GitHub Actions handles updates.

## Install

```console
uvx librivox-mirror --help
```

For development:

```console
uv sync --locked --group dev --no-group integration
uv run librivox-mirror --help
```

Python 3.12 through 3.14 is supported.

## Dataset

The Hugging Face dataset exposes three configurations:

- `default`: streaming WebDataset audio and sample metadata
- `books`: one Parquet row per book
- `sections`: one Parquet row per section

```python
from datasets import load_dataset

books = load_dataset("owner/librivox", "books", split="train")
audio = load_dataset("owner/librivox", split="train", streaming=True)
```

Original audio is never transcoded. Complete LibriVox and Internet Archive source
records are retained alongside normalized fields and checksums.

## Usage

```console
librivox-mirror plan
librivox-mirror mirror 47
librivox-mirror backfill --start-id 1 --end-id 1000 --jobs 4
```

Set `HF_DATASET_REPO` and `HF_TOKEN` to publish instead of only staging locally.
SQLite under `.librivox-mirror/` stores durable checkpoints, so interrupted backfills
can resume safely. Run `librivox-mirror --help` for all commands and options.

## Automation

Configure these GitHub settings before enabling publication:

- repository variable `HF_DATASET_REPO`
- repository secret `HF_TOKEN` with dataset write access
- PyPI Trusted Publisher for the `pypi` environment

`python.yml` validates Python and manages Release Please. `huggingface.yml` runs the
daily sync, monthly reconciliation, and manual dataset operations. Dependencies use
a three-day release cooldown.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development checks and conventions.

## Licenses

The code is MIT licensed. Copyrightable mirror-specific curation, indexes, metadata,
and documentation are CC BY 4.0. Original LibriVox audio is not relicensed and
remains public domain in the United States. Rules may vary by jurisdiction.
