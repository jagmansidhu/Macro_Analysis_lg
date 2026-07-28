"""Ingest state tracker — persists SHA-256 hashes of ingested files.

Stored as a JSON sidecar ``ingest_state.json`` at the project root so that
incremental ingestion only processes new or modified files.
"""

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Default location of the sidecar file (project root)
_DEFAULT_STATE_PATH = Path(__file__).resolve().parent.parent.parent / "ingest_state.json"


def _load_state(state_path: Path) -> dict[str, str]:
    """Load the persisted hash map, returning an empty dict if missing/corrupt."""
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read ingest state file: %s — starting fresh.", exc)
        return {}


def _save_state(state: dict[str, str], state_path: Path) -> None:
    """Persist the hash map to disk."""
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def file_hash(file_path: Path) -> str:
    """Return the SHA-256 hex digest of a file's contents."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def is_file_new_or_changed(
    file_path: Path,
    state_path: Path = _DEFAULT_STATE_PATH,
) -> bool:
    """Return True if the file has not been indexed yet or its content has changed."""
    state = _load_state(state_path)
    key = str(file_path.resolve())
    current_hash = file_hash(file_path)
    return state.get(key) != current_hash


def mark_file_ingested(
    file_path: Path,
    state_path: Path = _DEFAULT_STATE_PATH,
) -> None:
    """Record the current hash of a successfully ingested file."""
    state = _load_state(state_path)
    key = str(file_path.resolve())
    state[key] = file_hash(file_path)
    _save_state(state, state_path)
    logger.debug("Marked ingested: %s", file_path.name)


def get_new_or_changed_files(
    directory: Path,
    glob_pattern: str = "**/*",
    state_path: Path = _DEFAULT_STATE_PATH,
) -> list[Path]:
    """Return all files under *directory* whose hashes differ from the stored state."""
    changed: list[Path] = []
    for path in sorted(directory.glob(glob_pattern)):
        if path.is_file() and is_file_new_or_changed(path, state_path):
            changed.append(path)
    return changed
