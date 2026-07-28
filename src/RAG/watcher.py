"""File-system watcher — auto-indexes new and modified research files.

Uses ``watchdog`` to monitor the configured watch directories.  When a file
creation or modification event fires the watcher calls
``LocalFileSource.ingest_file()`` so the document lands in PGVector within
seconds of hitting the filesystem.

The watcher runs as a daemon thread; it starts automatically when this module
is imported by ``graph.py`` (at LangGraph server startup).  No external process
or scheduler is needed.
"""

import logging
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)

# Extensions we care about
_WATCHED_EXTENSIONS = {".csv", ".pdf", ".txt"}

# Debounce window in seconds — ignore duplicate events within this window
_DEBOUNCE_SECONDS = 1.5

# Module-level observer (singleton)
_observer: Observer | None = None
_observer_lock = threading.Lock()


class _ResearchFileHandler(FileSystemEventHandler):
    """Handles watchdog events and triggers incremental ingestion."""

    def __init__(self, local_source):
        super().__init__()
        self._local_source = local_source
        self._last_event: dict[str, float] = {}  # path → last event timestamp
        self._lock = threading.Lock()

    def _should_process(self, path_str: str) -> bool:
        """Return True if the event is outside the debounce window."""
        now = time.monotonic()
        with self._lock:
            last = self._last_event.get(path_str, 0.0)
            if now - last < _DEBOUNCE_SECONDS:
                return False
            self._last_event[path_str] = now
        return True

    def _handle(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() not in _WATCHED_EXTENSIONS:
            return
        if not self._should_process(str(path)):
            return

        logger.info("[Watcher] Detected change: %s", path.name)
        try:
            result = self._local_source.ingest_file(path)
            logger.info("[Watcher] Ingested %s → %s", path.name, result)
        except Exception as exc:
            logger.error("[Watcher] Failed to ingest %s: %s", path.name, exc)

    def on_created(self, event: FileSystemEvent) -> None:
        self._handle(event)

    def on_modified(self, event: FileSystemEvent) -> None:
        self._handle(event)

    def on_moved(self, event: FileSystemEvent) -> None:
        # Treat a move/rename into a watched dir as a creation
        dest = Path(getattr(event, "dest_path", ""))
        if dest.suffix.lower() in _WATCHED_EXTENSIONS:
            if self._should_process(str(dest)):
                logger.info("[Watcher] File moved in: %s", dest.name)
                try:
                    result = self._local_source.ingest_file(dest)
                    logger.info("[Watcher] Ingested %s → %s", dest.name, result)
                except Exception as exc:
                    logger.error("[Watcher] Failed to ingest %s: %s", dest.name, exc)


def start_watching(
    directories: list[str | Path],
    local_source,
) -> None:
    """
    Start the watchdog observer in a daemon thread.

    Safe to call multiple times — subsequent calls are no-ops if the observer
    is already running.

    Args:
        directories: List of directory paths to monitor (relative or absolute).
        local_source: A ``LocalFileSource`` instance whose ``ingest_file``
                      method will be called on each event.
    """
    global _observer

    with _observer_lock:
        if _observer is not None and _observer.is_alive():
            logger.debug("[Watcher] Already running — skipping start.")
            return

        observer = Observer()
        handler = _ResearchFileHandler(local_source)

        watched_count = 0
        for raw_dir in directories:
            dir_path = Path(raw_dir)
            # Create the directory if it doesn't exist so watchdog can schedule it
            dir_path.mkdir(parents=True, exist_ok=True)
            observer.schedule(handler, str(dir_path), recursive=True)
            logger.info("[Watcher] Watching: %s", dir_path.resolve())
            watched_count += 1

        if watched_count == 0:
            logger.warning("[Watcher] No valid directories to watch.")
            return

        observer.daemon = True
        observer.start()
        _observer = observer
        logger.info("[Watcher] Started — monitoring %d directories.", watched_count)


def stop_watching() -> None:
    """Gracefully stop the watchdog observer (useful for testing)."""
    global _observer
    with _observer_lock:
        if _observer is not None and _observer.is_alive():
            _observer.stop()
            _observer.join(timeout=5)
            logger.info("[Watcher] Stopped.")
        _observer = None
