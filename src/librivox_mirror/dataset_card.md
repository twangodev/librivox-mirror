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
- config_name: sections
  default: true
  data_files:
  - split: train
    path: metadata/sections/*.parquet
- config_name: books
  data_files:
  - split: train
    path: metadata/books/*.parquet
---

# LibriVox Mirror

[![License: CC BY 4.0](https://img.shields.io/badge/license-CC_BY_4.0-2ea44f?style=flat-square&logo=creativecommons&logoColor=white)](https://creativecommons.org/licenses/by/4.0/)
[![Hugging Face dataset](https://img.shields.io/badge/Hugging_Face-dataset-FFD21E?style=flat-square&logo=huggingface&logoColor=000)]($repo_url)
![Last updated](https://img.shields.io/badge/last_updated-$updated_at_badge-0969da?style=flat-square&logo=githubactions&logoColor=white)

Fast, structured, continuously updated LibriVox audio mirror.

## Current snapshot

| Metric | Value |
| --- | ---: |
| Published books | $published_books |
| Published sections | $published_sections |
| Audio hours | $audio_hours |
| Audio languages | $audio_languages |
| Quarantined books | $quarantined_books |
| Last updated (UTC) | `$updated_at` |

## Audio by language

$audio_by_language

## Dataset structure

The Viewer exposes `sections` and `books` as typed Parquet configs. Original audio
shards remain under `data/` for WebDataset streaming.

`mirror_tar_url` links to the mirrored shard; `sample_key` selects its audio sample.
`source_url` is the exact Internet Archive file, while `librivox_listen_url` is
LibriVox's playback link.

Use `language` and `hash_partition` to construct stable downstream subsets or
evaluation splits. Source metadata features preserve unmodeled LibriVox and
Internet Archive fields for provenance.

## License and attribution

The copyrightable mirror-specific compilation, curation, normalized metadata,
indexes, and documentation are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Please cite this dataset when using those parts. Original LibriVox audio remains
public domain in the United States and is not relicensed by this mirror. Rules may
differ by jurisdiction.

## Citation

```bibtex
@misc{ding2026librivoxmirror,
  author       = {James Ding},
  title        = {LibriVox Mirror},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{$repo_url}},
  note         = {Continuously updated dataset}
}
```

## Provenance and integrity

Each sample links to its LibriVox project and exact Internet Archive source file.
Upstream checksums and a mirror SHA-256 are preserved, and audio is not transcoded.
