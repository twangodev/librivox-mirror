from __future__ import annotations

import logging
import time
from collections.abc import Generator, Iterable, Sequence, Sized
from contextlib import suppress
from dataclasses import dataclass, field
from itertools import batched, islice
from pathlib import Path
from typing import Annotated, cast

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from librivox_mirror import __version__
from librivox_mirror.archive import InternetArchiveClient, QuarantinedBookError
from librivox_mirror.artifact import verify_artifact
from librivox_mirror.capacity import StagingCapacity
from librivox_mirror.catalog import BookNotFoundError, LibriVoxCatalog
from librivox_mirror.hub import HubPublisher
from librivox_mirror.models import Book, BookStatus, SyncState, canonical_metadata_json
from librivox_mirror.state import RunLock, StateStore
from librivox_mirror.streaming import ordered_parallel_map, prefetch
from librivox_mirror.workflow import BookOutcome, BookProgress, MirrorRunner

DEFAULT_STATE = Path(".librivox-mirror/state.sqlite3")
DEFAULT_STAGING = Path(".librivox-mirror/staging")
DEFAULT_STAGING_LIMIT_GIB = 64
MINIMUM_FREE_GIB = 10
DEFAULT_USER_AGENT = (
    f"librivox-mirror/{__version__} "
    "(ML dataset mirroring; https://pypi.org/project/librivox-mirror/)"
)
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="librivox-mirror",
    help="Build and maintain an ML-ready Hugging Face mirror of LibriVox.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


@dataclass(frozen=True)
class AppSettings:
    json_output: bool
    verbose: bool
    output: Console
    errors: Console


@dataclass
class TerminalBookProgress:
    progress: Progress
    task_id: TaskID
    description: str
    heartbeat: LoggingBookProgress

    def stage(self, stage: str) -> None:
        self.progress.update(self.task_id, description=f"{self.description} · {stage}")

    def download(self, completed_bytes: int, total_bytes: int) -> None:
        self.progress.update(self.task_id, completed=completed_bytes, total=total_bytes)
        self.heartbeat.download(completed_bytes, total_bytes)


@dataclass
class LoggingBookProgress:
    book_id: int
    interval_seconds: float = 10
    started_at: float | None = field(default=None, init=False)
    last_reported_at: float = field(default=0, init=False)

    def stage(self, stage: str) -> None:
        return

    def download(self, completed_bytes: int, total_bytes: int) -> None:
        now = time.monotonic()
        if self.started_at is None:
            self.started_at = now
        if completed_bytes < total_bytes and now - self.last_reported_at < self.interval_seconds:
            return
        elapsed = max(now - self.started_at, 0.001)
        percent = completed_bytes / total_bytes * 100 if total_bytes else 100
        logger.info(
            "Book %s download: %.1f/%.1f MiB (%.1f%%, %.1f MiB/s)",
            self.book_id,
            completed_bytes / 1024**2,
            total_bytes / 1024**2,
            percent,
            completed_bytes / 1024**2 / elapsed,
        )
        self.last_reported_at = now


@app.callback()
def root(
    context: typer.Context,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Write machine-readable JSON to stdout."),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Include debug logs and source paths."),
    ] = False,
    log_file: Annotated[
        Path | None,
        typer.Option(
            "--log-file",
            dir_okay=False,
            help="Append plain-text logs to a file.",
            envvar="LIBRIVOX_MIRROR_LOG_FILE",
        ),
    ] = None,
) -> None:
    """Build and maintain an ML-ready Hugging Face mirror of LibriVox."""
    output = Console(no_color=json_output)
    errors = Console(stderr=True, no_color=json_output)
    handlers: list[logging.Handler] = [
        RichHandler(
            console=errors,
            markup=False,
            rich_tracebacks=errors.is_terminal,
            show_path=verbose,
            show_time=False,
        )
    ]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            )
        )
        handlers.append(file_handler)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=handlers,
        force=True,
    )
    for logger_name in ("httpcore", "httpx", "huggingface_hub", "internetarchive", "urllib3"):
        logging.getLogger(logger_name).setLevel(logging.DEBUG if verbose else logging.WARNING)
    context.obj = AppSettings(
        json_output=json_output,
        verbose=verbose,
        output=output,
        errors=errors,
    )


@app.command()
def version(context: typer.Context) -> None:
    """Print the installed package version."""
    emit(context, {"version": __version__}, __version__)


@app.command()
def status(
    context: typer.Context,
    state_path: Annotated[Path, typer.Option("--state")] = DEFAULT_STATE,
    recent: Annotated[
        int,
        typer.Option(min=0, max=100, help="Recent failures and quarantines to include."),
    ] = 10,
) -> None:
    """Inspect persistent backfill checkpoints and recent problems."""
    counts = {status.value: 0 for status in BookStatus}
    active_run = RunLock.inspect(state_path)
    if not state_path.exists():
        emit(
            context,
            {
                "state": str(state_path),
                "exists": False,
                "active_run": active_run,
                "books": 0,
                "counts": counts,
            },
            f"No state database at {state_path}",
        )
        return

    with StateStore(state_path, read_only=True) as state:
        stored_counts = state.counts()
        counts.update({status.value: count for status, count in stored_counts.items()})
        checkpoints = state.list()
    problems = sorted(
        (
            checkpoint
            for checkpoint in checkpoints
            if checkpoint.last_error or checkpoint.status == BookStatus.QUARANTINED
        ),
        key=lambda checkpoint: checkpoint.updated_at,
        reverse=True,
    )[:recent]
    problem_rows = [
        {
            "book_id": checkpoint.book_id,
            "status": checkpoint.status,
            "attempts": checkpoint.attempt_count,
            "error": checkpoint.last_error or f"{checkpoint.error_code}: {checkpoint.error_detail}",
            "updated_at": checkpoint.updated_at.isoformat(),
        }
        for checkpoint in problems
    ]
    packed_bytes = 0
    for checkpoint in checkpoints:
        if checkpoint.artifact_path is None:
            continue
        with suppress(FileNotFoundError):
            packed_bytes += checkpoint.artifact_path.stat().st_size
    payload = {
        "state": str(state_path),
        "exists": True,
        "integrity": "ok",
        "active_run": active_run,
        "books": len(checkpoints),
        "counts": counts,
        "packed_bytes": packed_bytes,
        "recent_problems": problem_rows,
    }
    count_text = (
        ", ".join(f"{name}={count}" for name, count in counts.items() if count > 0)
        or "no checkpoints"
    )
    lines = [
        f"State: {state_path} ({len(checkpoints)} books, integrity ok)",
        f"Active run: {active_run or 'none'}",
        count_text,
        f"Recoverable packed artifacts: {packed_bytes / 1024**2:.1f} MiB",
    ]
    if problem_rows:
        lines.append("Recent problems:")
        lines.extend(
            f"  book {row['book_id']} [{row['status']}]: {row['error']}" for row in problem_rows
        )
    emit(context, payload, "\n".join(lines))


@app.command("plan")
def plan_command(
    context: typer.Context,
    since: Annotated[
        int | None,
        typer.Option(help="Unix timestamp for the LibriVox update window."),
    ] = None,
    start_id: Annotated[int | None, typer.Option(min=1)] = None,
    end_id: Annotated[int | None, typer.Option(min=1)] = None,
    max_books: Annotated[int, typer.Option(min=1, max=1000)] = 100,
    resolve: Annotated[
        bool,
        typer.Option(help="Resolve Internet Archive originals and report source bytes."),
    ] = False,
    request_delay: Annotated[float, typer.Option(min=0)] = 1,
    user_agent: Annotated[
        str,
        typer.Option(envvar="LIBRIVOX_MIRROR_USER_AGENT"),
    ] = DEFAULT_USER_AGENT,
) -> None:
    """Inspect a bounded set of candidate books without changing local or remote state."""
    validate_range(start_id, end_id)
    effective_since = since
    if effective_since is None and start_id is None:
        effective_since = int(time.time()) - 48 * 60 * 60
    with LibriVoxCatalog(user_agent=user_agent) as catalog:
        books = take(
            catalog.iter_books(since=effective_since, start_id=start_id, end_id=end_id),
            max_books,
        )
    rows: list[dict[str, object]] = []
    if resolve:
        with InternetArchiveClient(
            user_agent=user_agent,
            request_delay=request_delay,
        ) as archive:
            for book in books:
                try:
                    resolved = archive.resolve_book(book)
                except QuarantinedBookError as error:
                    rows.append(
                        {
                            "book_id": book.id,
                            "title": book.title,
                            "status": "quarantined",
                            "reason": error.record.code,
                            "detail": error.record.detail,
                        }
                    )
                    continue
                rows.append(
                    {
                        "book_id": book.id,
                        "title": book.title,
                        "status": "resolved",
                        "sections": len(resolved.sections),
                        "source_bytes": sum(
                            section.archive_file.size for section in resolved.sections
                        ),
                    }
                )
    else:
        rows = [book_summary(book) for book in books]
    emit(context, {"books": rows, "count": len(rows)}, f"Planned {len(rows)} books")


@app.command()
def mirror(
    context: typer.Context,
    book_id: Annotated[int, typer.Argument(min=1)],
    repo: Annotated[
        str | None,
        typer.Option(help="Hugging Face dataset repository.", envvar="HF_DATASET_REPO"),
    ] = None,
    token: Annotated[
        str | None,
        typer.Option(help="Hugging Face token.", envvar="HF_TOKEN", show_default=False),
    ] = None,
    state_path: Annotated[Path, typer.Option("--state")] = DEFAULT_STATE,
    staging: Annotated[Path, typer.Option()] = DEFAULT_STAGING,
    download_jobs: Annotated[
        int,
        typer.Option(
            "--download-jobs",
            "--jobs",
            min=1,
            help="Maximum concurrent MP3 downloads.",
        ),
    ] = 4,
    request_delay: Annotated[float, typer.Option(min=0)] = 1,
    dry_run: Annotated[bool, typer.Option()] = False,
    user_agent: Annotated[
        str,
        typer.Option(envvar="LIBRIVOX_MIRROR_USER_AGENT"),
    ] = DEFAULT_USER_AGENT,
) -> None:
    """Mirror one book locally, optionally publishing it to Hugging Face."""
    publisher = HubPublisher(repo, token=token, working_directory=staging / "hub") if repo else None
    with (
        LibriVoxCatalog(user_agent=user_agent) as catalog,
        InternetArchiveClient(
            user_agent=user_agent,
            request_delay=request_delay,
            download_jobs=download_jobs,
        ) as archive,
    ):
        book = catalog.get_book(book_id)
        if dry_run:
            report_resolution(context, archive, book)
            return
        with RunLock(state_path), StateStore(state_path) as state:
            runner = MirrorRunner(
                catalog=catalog,
                archive=archive,
                state=state,
                staging_directory=staging,
                publisher=publisher,
                source_index=publisher,
            )
            outcome = prepare_books(context, runner, [book], resilient=False)[0]
            sync_state = publisher.load_sync_state() if publisher else SyncState()
            result = runner.publish(
                [outcome],
                sync_state,
                commit_message=f"feat(data): mirror LibriVox book {book.id}",
            )
    payload = outcome_summary(outcome)
    if result:
        payload["revision"] = result.revision
    emit(context, payload, human_outcome(outcome, result.revision if result else None))


@app.command()
def backfill(
    context: typer.Context,
    start_id: Annotated[int, typer.Option(min=1)],
    end_id: Annotated[int, typer.Option(min=1)],
    repo: Annotated[
        str,
        typer.Option(help="Hugging Face dataset repository.", envvar="HF_DATASET_REPO"),
    ],
    token: Annotated[
        str | None,
        typer.Option(help="Hugging Face token.", envvar="HF_TOKEN", show_default=False),
    ] = None,
    state_path: Annotated[Path, typer.Option("--state")] = DEFAULT_STATE,
    staging: Annotated[Path, typer.Option()] = DEFAULT_STAGING,
    download_jobs: Annotated[
        int,
        typer.Option(
            "--download-jobs",
            "--jobs",
            min=1,
            help="Maximum concurrent MP3 downloads across all books.",
        ),
    ] = 4,
    book_jobs: Annotated[
        int,
        typer.Option(min=1, help="Books prepared concurrently."),
    ] = 2,
    upload_jobs: Annotated[
        int,
        typer.Option(min=1, help="Concurrent Hub upload workers."),
    ] = 8,
    request_delay: Annotated[
        float,
        typer.Option(min=0, help="Minimum seconds between Internet Archive requests."),
    ] = 1,
    staging_limit_gib: Annotated[
        int,
        typer.Option(min=1, help="Maximum in-flight staging reservation in GiB."),
    ] = DEFAULT_STAGING_LIMIT_GIB,
    max_books: Annotated[int | None, typer.Option(min=1)] = None,
    commit_size: Annotated[int, typer.Option(min=1, max=20)] = 20,
    dry_run: Annotated[bool, typer.Option()] = False,
    user_agent: Annotated[
        str,
        typer.Option(envvar="LIBRIVOX_MIRROR_USER_AGENT"),
    ] = DEFAULT_USER_AGENT,
) -> None:
    """Mirror an explicit LibriVox ID range in commits of at most 20 books."""
    validate_range(start_id, end_id)
    publisher = HubPublisher(
        repo,
        token=token,
        working_directory=staging / "hub",
        upload_jobs=upload_jobs,
    )
    source_index = HubPublisher(
        repo,
        token=token,
        working_directory=staging / "hub-reader",
    )
    staging_capacity = StagingCapacity(
        staging,
        max_bytes=staging_limit_gib * 1024**3,
        minimum_free_bytes=MINIMUM_FREE_GIB * 1024**3,
    )
    with LibriVoxCatalog(user_agent=user_agent) as catalog:
        books: Iterable[Book] = catalog.iter_books(start_id=start_id, end_id=end_id)
        if max_books is not None:
            books = islice(books, max_books)
        if dry_run:
            selected = list(books)
            emit(
                context,
                {"books": [book_summary(book) for book in selected], "count": len(selected)},
                f"Would backfill {len(selected)} books",
            )
            return
        with (
            InternetArchiveClient(
                user_agent=user_agent,
                request_delay=request_delay,
                download_jobs=download_jobs,
            ) as archive,
            RunLock(state_path),
            StateStore(state_path) as state,
        ):
            runner = MirrorRunner(
                catalog=catalog,
                archive=archive,
                state=state,
                staging_directory=staging,
                publisher=publisher,
                source_index=source_index,
                staging_capacity=staging_capacity,
            )
            logger.info(
                "Streaming LibriVox catalog books %s-%s into %s-book commits with "
                "%s book workers, %s total download workers, %s upload workers, and a %s GiB "
                "staging limit",
                start_id,
                end_id,
                commit_size,
                book_jobs,
                download_jobs,
                upload_jobs,
                staging_limit_gib,
            )
            with prefetch(catalog_progress(books), capacity=max(50, commit_size * 2)) as queued:
                outcomes, revisions = publish_batches(
                    context,
                    runner,
                    queued,
                    publisher.load_sync_state(),
                    commit_size,
                    book_jobs=book_jobs,
                )
    emit(
        context,
        {"outcomes": outcomes, "revisions": revisions},
        f"Processed {len(outcomes)} books in {len(revisions)} Hub commits",
    )


@app.command()
def sync(
    context: typer.Context,
    repo: Annotated[
        str,
        typer.Option(help="Hugging Face dataset repository.", envvar="HF_DATASET_REPO"),
    ],
    token: Annotated[
        str | None,
        typer.Option(help="Hugging Face token.", envvar="HF_TOKEN", show_default=False),
    ] = None,
    state_path: Annotated[Path, typer.Option("--state")] = DEFAULT_STATE,
    staging: Annotated[Path, typer.Option()] = DEFAULT_STAGING,
    download_jobs: Annotated[
        int,
        typer.Option(
            "--download-jobs",
            "--jobs",
            min=1,
            help="Maximum concurrent MP3 downloads.",
        ),
    ] = 2,
    request_delay: Annotated[float, typer.Option(min=0)] = 1,
    max_books: Annotated[int, typer.Option(min=1, max=20)] = 20,
    commit_size: Annotated[int, typer.Option(min=1, max=20)] = 1,
    dry_run: Annotated[bool, typer.Option()] = False,
    user_agent: Annotated[
        str,
        typer.Option(envvar="LIBRIVOX_MIRROR_USER_AGENT"),
    ] = DEFAULT_USER_AGENT,
) -> None:
    """Publish daily LibriVox changes using a 48-hour overlap window."""
    run_started = int(time.time())
    publisher = HubPublisher(repo, token=token, working_directory=staging / "hub")
    remote_state = publisher.load_sync_state()
    since = max(0, remote_state.catalog_watermark - 48 * 60 * 60)
    with LibriVoxCatalog(user_agent=user_agent) as catalog:
        candidates = deduplicate(catalog.iter_books(since=since))
        pending = [book for book in candidates if not publisher.has_current_book(book)]
        selected = pending[:max_books]
        if dry_run or not selected:
            emit(
                context,
                {
                    "books": [book_summary(book) for book in selected],
                    "candidate_count": len(candidates),
                    "pending_count": len(pending),
                    "complete_window": len(pending) <= max_books,
                },
                f"{'Would sync' if dry_run else 'Found'} {len(selected)} pending books",
            )
            return
        with (
            InternetArchiveClient(
                user_agent=user_agent,
                request_delay=request_delay,
                download_jobs=download_jobs,
            ) as archive,
            RunLock(state_path),
            StateStore(state_path) as state,
        ):
            runner = MirrorRunner(
                catalog=catalog,
                archive=archive,
                state=state,
                staging_directory=staging,
                publisher=publisher,
            )
            outcomes, revisions = publish_batches(
                context,
                runner,
                selected,
                remote_state,
                commit_size,
                final_catalog_watermark=(run_started if len(pending) <= max_books else None),
            )
    complete_window = len(pending) <= max_books and not any(
        outcome["status"] == "failed" for outcome in outcomes
    )
    emit(
        context,
        {
            "outcomes": outcomes,
            "revisions": revisions,
            "complete_window": complete_window,
        },
        f"Synced {len(outcomes)} books",
    )


@app.command()
def reconcile(
    context: typer.Context,
    repo: Annotated[
        str,
        typer.Option(help="Hugging Face dataset repository.", envvar="HF_DATASET_REPO"),
    ],
    token: Annotated[
        str | None,
        typer.Option(help="Hugging Face token.", envvar="HF_TOKEN", show_default=False),
    ] = None,
    start_id: Annotated[int | None, typer.Option(min=1)] = None,
    end_id: Annotated[int | None, typer.Option(min=1)] = None,
    state_path: Annotated[Path, typer.Option("--state")] = DEFAULT_STATE,
    staging: Annotated[Path, typer.Option()] = DEFAULT_STAGING,
    download_jobs: Annotated[
        int,
        typer.Option(
            "--download-jobs",
            "--jobs",
            min=1,
            help="Maximum concurrent MP3 downloads.",
        ),
    ] = 4,
    request_delay: Annotated[float, typer.Option(min=0)] = 1,
    max_books: Annotated[int, typer.Option(min=1, max=20)] = 20,
    commit_size: Annotated[int, typer.Option(min=1, max=20)] = 1,
    dry_run: Annotated[bool, typer.Option()] = False,
    user_agent: Annotated[
        str,
        typer.Option(envvar="LIBRIVOX_MIRROR_USER_AGENT"),
    ] = DEFAULT_USER_AGENT,
) -> None:
    """Find and republish changed catalog records, bounded to 20 changes per run."""
    validate_range(start_id, end_id)
    publisher = HubPublisher(repo, token=token, working_directory=staging / "hub")
    with LibriVoxCatalog(user_agent=user_agent) as catalog:
        pending = []
        for book in catalog.iter_books(start_id=start_id, end_id=end_id):
            if not publisher.has_current_book(book):
                pending.append(book)
                if len(pending) == max_books:
                    break
        if dry_run or not pending:
            emit(
                context,
                {"books": [book_summary(book) for book in pending], "count": len(pending)},
                f"{'Would reconcile' if dry_run else 'Found'} {len(pending)} changed books",
            )
            return
        with (
            InternetArchiveClient(
                user_agent=user_agent,
                request_delay=request_delay,
                download_jobs=download_jobs,
            ) as archive,
            RunLock(state_path),
            StateStore(state_path) as state,
        ):
            runner = MirrorRunner(
                catalog=catalog,
                archive=archive,
                state=state,
                staging_directory=staging,
                publisher=publisher,
            )
            outcomes, revisions = publish_batches(
                context,
                runner,
                pending,
                publisher.load_sync_state(),
                commit_size,
            )
    emit(
        context,
        {
            "outcomes": outcomes,
            "revisions": revisions,
        },
        f"Reconciled {len(outcomes)} books",
    )


@app.command()
def verify(
    context: typer.Context,
    artifact: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    sha256: Annotated[str | None, typer.Option(help="Expected TAR SHA-256.")] = None,
) -> None:
    """Verify a local WebDataset TAR and its paired sample records."""
    actual_sha256, samples = verify_artifact(artifact, sha256)
    emit(
        context,
        {"path": str(artifact), "sha256": actual_sha256, "samples": samples},
        f"Verified {samples} samples in {artifact}",
    )


def publish_batches(
    context: typer.Context,
    runner: MirrorRunner,
    books: Iterable[Book],
    sync_state: SyncState,
    commit_size: int,
    final_catalog_watermark: int | None = None,
    book_jobs: int = 1,
) -> tuple[list[dict[str, object]], list[str]]:
    if final_catalog_watermark is not None and not isinstance(books, Sized):
        raise ValueError("a final catalog watermark requires a sized book collection")

    total_books = len(books) if isinstance(books, Sized) else None
    all_outcomes: list[dict[str, object]] = []
    revisions: list[str] = []
    current_state = sync_state
    processed_books = 0
    has_failures = False
    prepared = prepare_books_stream(
        context,
        runner,
        books,
        workers=book_jobs,
        capacity=commit_size + book_jobs,
    )
    try:
        for batch_tuple in batched(prepared, commit_size):
            batch = list(batch_tuple)
            all_outcomes.extend(outcome_summary(outcome) for outcome in batch)
            processed_books += len(batch)
            has_failures = has_failures or any(outcome.error for outcome in batch)
            ids = [outcome.book.id for outcome in batch]
            batch_state = current_state
            if (
                processed_books == total_books
                and final_catalog_watermark is not None
                and not has_failures
            ):
                batch_state = current_state.model_copy(
                    update={"catalog_watermark": final_catalog_watermark}
                )
            result = runner.publish(
                batch,
                batch_state,
                commit_message=f"feat(data): mirror LibriVox books {min(ids)}-{max(ids)}",
            )
            if result:
                revisions.append(result.revision)
                current_state = result.state
    finally:
        prepared.close()
    return all_outcomes, revisions


def prepare_books(
    context: typer.Context,
    runner: MirrorRunner,
    books: Sequence[Book],
    *,
    resilient: bool = True,
) -> list[BookOutcome]:
    return list(
        prepare_books_stream(
            context,
            runner,
            books,
            workers=1,
            capacity=1,
            resilient=resilient,
        )
    )


def prepare_books_stream(
    context: typer.Context,
    runner: MirrorRunner,
    books: Iterable[Book],
    *,
    workers: int,
    capacity: int,
    resilient: bool = True,
) -> Generator[BookOutcome]:
    settings = app_settings(context)
    interactive = settings.errors.is_terminal and not settings.json_output
    total_books = len(books) if isinstance(books, Sized) else None
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        DownloadColumn(binary_units=True),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=settings.errors,
        disable=not interactive,
        transient=True,
    ) as progress:

        def prepare(indexed_book: tuple[int, Book]) -> BookOutcome:
            index, book = indexed_book
            position = f"{index}/{total_books}" if total_books is not None else str(index)
            description = f"[{position}] Book {book.id}"
            task_id = progress.add_task(f"{description} · queued", total=None)
            book_progress: BookProgress
            if interactive:
                book_progress = TerminalBookProgress(
                    progress,
                    task_id,
                    description,
                    LoggingBookProgress(book.id),
                )
            else:
                book_progress = LoggingBookProgress(book.id)
            prepare_book = runner.prepare_book_resiliently if resilient else runner.prepare_book
            try:
                return prepare_book(book, progress=book_progress)
            finally:
                progress.remove_task(task_id)

        yield from ordered_parallel_map(
            enumerate(books, start=1),
            prepare,
            workers=workers,
            capacity=capacity,
        )


def catalog_progress(books: Iterable[Book]) -> Iterable[Book]:
    count = 0
    for count, book in enumerate(books, start=1):
        if count == 1 or count % 1000 == 0:
            logger.info("Queued %s LibriVox catalog books", count)
        yield book
    logger.info("Finished catalog scan with %s books", count)


def report_resolution(
    context: typer.Context,
    archive: InternetArchiveClient,
    book: Book,
) -> None:
    try:
        resolved = archive.resolve_book(book)
    except QuarantinedBookError as error:
        emit(
            context,
            {
                "book_id": book.id,
                "status": "quarantined",
                "code": error.record.code,
                "detail": error.record.detail,
            },
            f"Book {book.id} would be quarantined: {error.record.detail}",
        )
        return
    source_bytes = sum(section.archive_file.size for section in resolved.sections)
    emit(
        context,
        {
            "book_id": book.id,
            "status": "resolved",
            "sections": len(resolved.sections),
            "source_bytes": source_bytes,
        },
        f"Book {book.id}: {len(resolved.sections)} original MP3s, {source_bytes:,} bytes",
    )


def deduplicate(books) -> list[Book]:
    return sorted({book.id: book for book in books}.values(), key=lambda book: book.id)


def take(books, count: int) -> list[Book]:
    selected = []
    for book in books:
        selected.append(book)
        if len(selected) == count:
            break
    return selected


def book_summary(book: Book) -> dict[str, object]:
    return {
        "book_id": book.id,
        "title": book.title,
        "language": book.language,
        "sections": len(book.sections),
        "source_fingerprint": book.source_fingerprint,
    }


def outcome_summary(outcome: BookOutcome) -> dict[str, object]:
    if outcome.error:
        status = "failed"
    elif outcome.skipped:
        status = "unchanged"
    elif outcome.quarantine:
        status = "quarantined"
    elif outcome.artifact:
        status = "packed"
    else:
        status = "unknown"
    payload: dict[str, object] = {"book_id": outcome.book.id, "status": status}
    if outcome.error:
        payload["error"] = outcome.error
    if outcome.quarantine:
        payload.update(code=outcome.quarantine.code, detail=outcome.quarantine.detail)
    if outcome.artifact:
        payload.update(
            artifact=str(outcome.artifact.path),
            sha256=outcome.artifact.sha256,
            bytes=outcome.artifact.size,
        )
    return payload


def human_outcome(outcome: BookOutcome, revision: str | None) -> str:
    if outcome.error:
        return f"Deferred book {outcome.book.id}: {outcome.error}"
    if outcome.skipped:
        return f"Book {outcome.book.id} is already current"
    if outcome.quarantine:
        return f"Quarantined book {outcome.book.id}: {outcome.quarantine.detail}"
    if revision:
        return f"Published book {outcome.book.id} in revision {revision}"
    if outcome.artifact:
        return f"Packed book {outcome.book.id} at {outcome.artifact.path}"
    return f"Processed book {outcome.book.id}"


def validate_range(start_id: int | None, end_id: int | None) -> None:
    if start_id is not None and end_id is not None and start_id > end_id:
        raise typer.BadParameter("--start-id must be less than or equal to --end-id")


def app_settings(context: typer.Context) -> AppSettings:
    return cast(AppSettings, context.obj)


def emit(context: typer.Context, payload: object, human: str) -> None:
    settings = app_settings(context)
    if settings.json_output:
        typer.echo(canonical_metadata_json(payload))
    else:
        settings.output.print(human)


def main() -> None:
    try:
        app()
    except (BookNotFoundError, OSError, ValueError) as error:
        logging.getLogger(__name__).error("%s", error)
        raise SystemExit(1) from error
