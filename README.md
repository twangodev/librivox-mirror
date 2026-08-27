# librivox-mirror

`librivox-mirror` builds and maintains an ML-ready Hugging Face mirror of the
LibriVox catalog. It preserves original MP3 bytes, groups one book per
deterministic WebDataset TAR, and publishes compact Parquet indexes for discovery.

The project is under active development. The CLI never starts a full backfill by
default; every command is bounded by an explicit book ID, range, or update window.
