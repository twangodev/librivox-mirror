# Contributing

Install the locked development environment and run the same checks as CI:

```console
uv sync --locked --group dev --no-group integration
uv run ruff format .
uv run ruff check .
uv run ty check
uv run pytest
uv build --no-sources
```

Dataset-format tests use a separate lightweight dependency group:

```console
uv sync --locked --group dev --group integration
uv run pytest -m integration
```

An opt-in source smoke test downloads one original section:

```console
RUN_LIBRIVOX_LIVE_TESTS=1 uv run pytest -m live
```

Keep modules concrete and names explicit. Add a shared abstraction only after two
real call sites establish what it represents. Preserve complete upstream metadata
at source boundaries, and keep transformations deterministic.

Use Conventional Commits. Keep changes narrow enough that each commit can be
formatted, type-checked, tested, and reviewed independently.
