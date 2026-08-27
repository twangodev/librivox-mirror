# librivox-mirror

`librivox-mirror` builds and maintains an ML-ready Hugging Face mirror of the
LibriVox catalog. It preserves original MP3 bytes, groups one book per deterministic
WebDataset TAR, and publishes compact Parquet indexes for discovery.

The slow initial backfill is designed for dedicated infrastructure. Daily updates
are small enough to run in GitHub Actions without relying on a long-lived runner.

## Install

Run the published CLI without managing an environment:

```console
uvx librivox-mirror --help
```

For development, clone the repository and let uv create the locked environment:

```console
uv sync --locked --group dev --no-group integration
uv run librivox-mirror --help
```

The package supports Python 3.12 through 3.14. Ruff formats and lints the project,
ty checks types, and pytest runs the test suite. uv excludes every package uploaded
within the previous three days when refreshing `uv.lock`.

## Dataset layout

```text
data/000/000047.tar
metadata/books/000.parquet
metadata/sections/000.parquet
metadata/quarantine/000.parquet
state/sync.json
README.md
```

Each TAR contains matching `<book_id>-<section_id>.mp3` and `.json` records. TAR
headers, record ordering, and JSON encoding are normalized, so rebuilding unchanged
source data produces identical bytes. Original LibriVox audio bytes are never
transcoded.

The generated Hugging Face dataset exposes three configurations:

- `default`: streaming WebDataset audio and sample metadata
- `books`: one compact Parquet row per book
- `sections`: one compact Parquet row per section with TAR path and sample key

Common ML fields are normalized for efficient filtering. The complete LibriVox
book and section records, Internet Archive item metadata, and selected file records
are retained in `*_metadata_json` columns, including upstream fields unknown to this
version of the mirror.

```python
from datasets import load_dataset

books = load_dataset("owner/librivox", "books", split="train")
english = books.filter(lambda row: row["language"] == "English")

audio = load_dataset("owner/librivox", split="train", streaming=True)
first_sample = next(iter(audio))
```

All audio belongs to the `train` split. The `language` and stable
`hash_partition` columns let downstream users construct their own reproducible
subsets and evaluation splits.

## CLI

Inspect recent catalog changes without writing state or downloading audio:

```console
librivox-mirror plan
librivox-mirror plan --start-id 40 --end-id 50 --resolve
```

Mirror one book locally:

```console
librivox-mirror mirror 47
```

Publish it by configuring a dataset repository and token:

```console
export HF_DATASET_REPO=owner/librivox
export HF_TOKEN=hf_...
export LIBRIVOX_MIRROR_USER_AGENT="librivox-mirror/0.1 (contact: you@example.com)"
librivox-mirror mirror 47
```

Run an explicit backfill range on the dedicated host:

```console
librivox-mirror backfill \
  --start-id 1 \
  --end-id 1000 \
  --commit-size 1 \
  --jobs 4
```

SQLite under `.librivox-mirror/` stores durable local checkpoints. Re-running the
same command resumes verified downloads and skips source fingerprints already on
the Hub. SQLite is never uploaded; the Parquet indexes and `state/sync.json` are the
shared source of truth.

`--commit-size` may be raised to 20 when the backfill host has enough staging disk.
GitHub Actions deliberately uses one book per commit to keep hosted-runner disk
usage bounded. A TAR and its downloaded MP3s are removed only after the atomic Hub
commit succeeds.

Every command supports human terminal output, with Rich progress and tracebacks on
interactive terminals. Put `--json` before the command for stable JSON on stdout;
logs remain on stderr and terminal colors are automatically disabled in CI and
pipes.

## Updating and reconciliation

The daily workflow queries LibriVox using the last committed watermark with a
48-hour overlap. It compares stable source fingerprints against the Hub, publishes
only changed books, and advances the watermark only after the entire selected
window succeeds. A no-change run creates no Hub commit.

The monthly workflow walks the catalog and republishes changed fingerprints. A
book is quarantined as a unit when any section lacks one unambiguous original MP3.
Quarantine rows preserve the reason and complete LibriVox source record. Transient
HTTP failures are retried and fail the run instead of being misclassified as source
problems.

Internet Archive requests use a descriptive User-Agent, bounded concurrency,
inter-request delay, upstream checksums, exponential retries, and `Retry-After`
handling. GitHub-hosted IPs are not assumed to be privileged; throttling is treated
as normal backpressure.

## Repository automation

Configure these GitHub settings before enabling scheduled publication:

- repository variable `HF_DATASET_REPO`, such as `owner/librivox`
- repository secret `HF_TOKEN` with write access to that dataset
- PyPI Trusted Publisher for the `pypi` environment before pushing a `v*` tag

GitHub Actions and uv are pinned to immutable, cooldown-cleared versions. Renovate
keeps the lockfile and Action digests current while applying the same three-day
minimum release age.

## Rights

The code is MIT licensed. LibriVox states that its recordings are in the public
domain in the United States. Copyright and public-domain rules vary by jurisdiction;
dataset users are responsible for checking the rules that apply to them.
