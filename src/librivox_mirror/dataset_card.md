---
pretty_name: LibriVox Mirror
language: multilingual
license: cc-by-4.0
task_categories:
- automatic-speech-recognition
- text-to-speech
tags:
- audio
- librivox
- public-domain
- webdataset
- datasets
configs:
- config_name: default
  data_files:
  - split: train
    path: data/**/*.tar
- config_name: books
  data_files:
  - split: train
    path: metadata/books/*.parquet
- config_name: sections
  data_files:
  - split: train
    path: metadata/sections/*.parquet
---

# LibriVox Mirror

[![License: CC BY 4.0](https://img.shields.io/badge/license-CC_BY_4.0-2ea44f?style=flat-square&logo=creativecommons&logoColor=white)](https://creativecommons.org/licenses/by/4.0/)
[![Hugging Face dataset](https://img.shields.io/badge/Hugging_Face-dataset-FFD21E?style=flat-square&logo=huggingface&logoColor=000)]($repo_url)
![Last updated](https://img.shields.io/badge/last_updated-$updated_at_badge-0969da?style=flat-square&logo=githubactions&logoColor=white)

An ML-ready, continuously updated mirror of original LibriVox MP3 files. Audio is
stored as one deterministic WebDataset TAR per book, with compact Parquet indexes
for book and section metadata.

## Current snapshot

| Metric | Value |
| --- | ---: |
| Published books | $published_books |
| Published sections | $published_sections |
| Quarantined books | $quarantined_books |
| Last updated (UTC) | `$updated_at` |

## Dataset structure

All audio belongs to the `train` split. Use `language` and `hash_partition` from the
Parquet indexes to construct stable downstream subsets or evaluation splits.

The `*_metadata_json` columns retain complete source records from LibriVox and
Internet Archive.

## License and attribution

The copyrightable mirror-specific compilation, curation, normalized metadata,
indexes, and documentation are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Please cite this dataset when using those parts. Original LibriVox audio remains
public domain in the United States and is not relicensed by this mirror. Rules may
differ by jurisdiction.

## Citation

```bibtex
@misc{librivoxmirror2026,
  author       = {LibriVox Mirror maintainers},
  title        = {LibriVox Mirror: An ML-Ready Mirror of the LibriVox Catalog},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{$repo_url}},
  note         = {Continuously updated dataset}
}
```

## Provenance and integrity

Each sample links to its LibriVox project and exact Internet Archive source file.
Upstream checksums and a mirror SHA-256 are preserved, and audio is not transcoded.
