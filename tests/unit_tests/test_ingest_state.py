"""Unit tests for ingest_state — SHA-256 hash tracking."""

import json
import tempfile
from pathlib import Path

import pytest

from RAG.ingest_state import (
    file_hash,
    is_file_new_or_changed,
    mark_file_ingested,
    get_new_or_changed_files,
)


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def state_file(tmp_path):
    return tmp_path / "ingest_state.json"


def make_file(directory: Path, name: str, content: str) -> Path:
    p = directory / name
    p.write_text(content, encoding="utf-8")
    return p


class TestFileHash:
    def test_same_content_same_hash(self, tmp_dir):
        f1 = make_file(tmp_dir, "a.txt", "hello world")
        f2 = make_file(tmp_dir, "b.txt", "hello world")
        assert file_hash(f1) == file_hash(f2)

    def test_different_content_different_hash(self, tmp_dir):
        f1 = make_file(tmp_dir, "a.txt", "hello")
        f2 = make_file(tmp_dir, "b.txt", "world")
        assert file_hash(f1) != file_hash(f2)


class TestIsFileNewOrChanged:
    def test_new_file_is_detected(self, tmp_dir, state_file):
        f = make_file(tmp_dir, "data.csv", "date,value\n2025-01-01,5.0")
        assert is_file_new_or_changed(f, state_path=state_file) is True

    def test_marked_file_is_skipped(self, tmp_dir, state_file):
        f = make_file(tmp_dir, "data.csv", "date,value\n2025-01-01,5.0")
        mark_file_ingested(f, state_path=state_file)
        assert is_file_new_or_changed(f, state_path=state_file) is False

    def test_modified_file_is_detected(self, tmp_dir, state_file):
        f = make_file(tmp_dir, "data.csv", "date,value\n2025-01-01,5.0")
        mark_file_ingested(f, state_path=state_file)
        # Modify the file
        f.write_text("date,value\n2025-02-01,6.0", encoding="utf-8")
        assert is_file_new_or_changed(f, state_path=state_file) is True


class TestGetNewOrChangedFiles:
    def test_returns_only_changed_files(self, tmp_dir, state_file):
        old = make_file(tmp_dir, "old.csv", "date,value\n2024-01-01,1.0")
        new = make_file(tmp_dir, "new.csv", "date,value\n2025-01-01,2.0")

        # Mark old as already ingested
        mark_file_ingested(old, state_path=state_file)

        changed = get_new_or_changed_files(tmp_dir, glob_pattern="*.csv", state_path=state_file)
        assert new in changed
        assert old not in changed

    def test_empty_dir_returns_empty_list(self, tmp_dir, state_file):
        result = get_new_or_changed_files(tmp_dir, state_path=state_file)
        assert result == []

    def test_corrupt_state_file_returns_all_files(self, tmp_dir, state_file):
        state_file.write_text("not valid json", encoding="utf-8")
        f = make_file(tmp_dir, "data.csv", "x,y\n1,2")
        changed = get_new_or_changed_files(tmp_dir, glob_pattern="*.csv", state_path=state_file)
        assert f in changed
