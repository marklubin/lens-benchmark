from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """Return a path to a temporary SQLite database file."""
    return tmp_path / "test_cache.db"
