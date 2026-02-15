from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Generator


class LensError(Exception):
    """Base exception for all LENS benchmark errors."""


class ConfigError(LensError):
    """Invalid or missing configuration."""


class AdapterError(LensError):
    """Error in adapter execution."""


class BudgetExceededError(LensError):
    """Adapter exceeded its LLM budget (calls or tokens)."""

    def __init__(self, phase: str, method: str, limit_kind: str, limit: int, actual: int) -> None:
        self.phase = phase
        self.method = method
        self.limit_kind = limit_kind
        self.limit = limit
        self.actual = actual
        super().__init__(
            f"Budget exceeded in {phase}/{method}: "
            f"{limit_kind} limit={limit}, actual={actual}"
        )


class LatencyExceededError(LensError):
    """Adapter exceeded latency cap for a method."""

    def __init__(self, method: str, limit_ms: float, actual_ms: float) -> None:
        self.method = method
        self.limit_ms = limit_ms
        self.actual_ms = actual_ms
        super().__init__(
            f"Latency exceeded in {method}: limit={limit_ms}ms, actual={actual_ms:.1f}ms"
        )


class ValidationError(LensError):
    """Output schema or evidence validation failure."""


class EvidenceError(ValidationError):
    """Evidence quote not found as exact substring in episode."""

    def __init__(self, episode_id: str, quote: str) -> None:
        self.episode_id = episode_id
        self.quote = quote
        super().__init__(
            f"Evidence quote not found in episode {episode_id}: {quote[:80]!r}..."
        )


class AntiCheatError(LensError):
    """Adapter attempted to access episode text at query time."""


class DatasetError(LensError):
    """Invalid or missing dataset."""


class ScoringError(LensError):
    """Error during scoring."""


class PluginError(LensError):
    """Error loading adapter or metric plugin."""


@contextlib.contextmanager
def atomic_write(path: Path | str) -> Generator[Path, None, None]:
    """Write to a temp file then atomically rename to target path.

    Usage:
        with atomic_write("output.json") as tmp:
            tmp.write_text(json.dumps(data))
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    tmp = Path(tmp_path)
    try:
        os.close(fd)
        yield tmp
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
