# Changelog

## [0.2.1](https://github.com/twangodev/librivox-mirror/compare/v0.2.0...v0.2.1) (2026-09-05)


### Bug Fixes

* **sync:** resume catalog catch-up across CI runs ([5599c21](https://github.com/twangodev/librivox-mirror/commit/5599c21d2fd9a2f7718947df27211283ae885343))

## [0.2.0](https://github.com/twangodev/librivox-mirror/compare/v0.1.0...v0.2.0) (2026-09-01)


### Features

* **backfill:** allow eight download jobs ([ce26e1c](https://github.com/twangodev/librivox-mirror/commit/ce26e1c34d13c05d0263f9166c354c0a9b1509c5))
* **backfill:** overlap downloads and uploads ([55bfad2](https://github.com/twangodev/librivox-mirror/commit/55bfad221eea01138336b6662da22bc7a4a8c78b))
* **backfill:** retry quarantined books ([96b80e3](https://github.com/twangodev/librivox-mirror/commit/96b80e3733ed1d842832b19c7e51bdd396c2ae79))
* **backfill:** stream catalog through bounded queue ([a226ed2](https://github.com/twangodev/librivox-mirror/commit/a226ed2bcf5be37a43d291e059fa6ecc7b8fe96c))
* **cli:** add live backfill observability ([8ac6396](https://github.com/twangodev/librivox-mirror/commit/8ac639605802e7dcc3a8a3000f3c3053e7bfc9db))
* **dataset:** publish audio hours metric ([f666f59](https://github.com/twangodev/librivox-mirror/commit/f666f59e262b38de176bdfd1d1b20413154c727f))
* **mirror:** report transfer progress ([2f39128](https://github.com/twangodev/librivox-mirror/commit/2f391288942a44f7799f0eab66dfcf5ccc213977))
* **pipeline:** add durable preparation workers ([60e7c8d](https://github.com/twangodev/librivox-mirror/commit/60e7c8dbf89fa32e655246f530a723943a1cba43))
* **pipeline:** bound staged artifact bytes ([60799f6](https://github.com/twangodev/librivox-mirror/commit/60799f63b85180729c6474e7f0b4e319a537c707))
* **pipeline:** share global download workers ([1bd4d2e](https://github.com/twangodev/librivox-mirror/commit/1bd4d2e7d7ece0aae813508053580f9446c9b6ca))
* **progress:** show packing throughput ([6501010](https://github.com/twangodev/librivox-mirror/commit/6501010e53c66fd85b4a699b9f2cfae506009a97))
* **repair:** queue quarantines directly ([4e64e2e](https://github.com/twangodev/librivox-mirror/commit/4e64e2e8dba92d5838ae5daf59ab1df190226a14))


### Bug Fixes

* **archive:** recover malformed LibriVox mappings ([156e055](https://github.com/twangodev/librivox-mirror/commit/156e0552b90dfa4ff406e3f5b364801e70109d15))
* **backfill:** persist through catalog outages ([6ac810a](https://github.com/twangodev/librivox-mirror/commit/6ac810a876ad485abc751f8eda05e3d0f8733384))
* **backfill:** recover from packing short reads ([e0d89a2](https://github.com/twangodev/librivox-mirror/commit/e0d89a22f24b1438c58e2d2f804c4d068d185251))
* **backfill:** survive transient service failures ([a6ec488](https://github.com/twangodev/librivox-mirror/commit/a6ec4886ed26f1a61b612a12582d5a18dbeae213))
* **catalog:** bound resume seek requests ([5b73ee7](https://github.com/twangodev/librivox-mirror/commit/5b73ee770312ba79c6df75b7a912a1e800c048bd))
* **cli:** prevent competing progress renderers ([8026ae1](https://github.com/twangodev/librivox-mirror/commit/8026ae17e2732bb760c99aa44ccc8f0b70bc623c))
* **network:** drop stale requests compatibility ([86bf3e3](https://github.com/twangodev/librivox-mirror/commit/86bf3e3b0f5eb96fb5742f2b84b5ea5fae74b1da))
* **pipeline:** schedule downloads fairly ([695c20b](https://github.com/twangodev/librivox-mirror/commit/695c20bd03e1246e99d6e60142280706ac7a21d2))


### Performance Improvements

* **archive:** resolve metadata concurrently ([d7abbe7](https://github.com/twangodev/librivox-mirror/commit/d7abbe7c87d5ebdb5ea41f970f3997c0db821c72))
* **catalog:** seek directly to resume id ([6122f3d](https://github.com/twangodev/librivox-mirror/commit/6122f3dd33a78c20f89a7bb14a6922a36424e1d9))

## [0.1.0](https://github.com/twangodev/librivox-mirror/compare/v0.0.0...v0.1.0) (2026-08-27)


### Features

* **dataset:** expose mirrored tar URLs ([2e8c838](https://github.com/twangodev/librivox-mirror/commit/2e8c8388dcd9ae88ee9502431810d6519759971f))


### Bug Fixes

* **dataset:** add playable audio previews ([f15f1fa](https://github.com/twangodev/librivox-mirror/commit/f15f1fa45038a45cab755968d0b18df1361646d3))
* **dataset:** restore durable Parquet viewer ([e1c5330](https://github.com/twangodev/librivox-mirror/commit/e1c5330518d6005adf41f4dc5fc28d3ff3f72fe4))
* **dataset:** show playable audio in viewer ([f4315f1](https://github.com/twangodev/librivox-mirror/commit/f4315f13b9e73afe4f49943a173d065df82a3b1b))
