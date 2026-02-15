from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import IO

from rich.console import Console


class Verbosity(IntEnum):
    QUIET = 0
    NORMAL = 1
    VERBOSE = 2
    DEBUG = 3


@dataclass
class StepLog:
    """A single logged step in a run."""

    step: str
    scope_id: str | None = None
    checkpoint: int | None = None
    message: str = ""
    elapsed_ms: float = 0.0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d: dict = {"step": self.step, "message": self.message, "elapsed_ms": self.elapsed_ms}
        if self.scope_id:
            d["scope_id"] = self.scope_id
        if self.checkpoint is not None:
            d["checkpoint"] = self.checkpoint
        if self.extra:
            d["extra"] = self.extra
        return d


class LensLogger:
    """Structured logger with Rich console output and step recording."""

    def __init__(
        self,
        verbosity: Verbosity = Verbosity.NORMAL,
        console: Console | None = None,
        log_file: IO[str] | None = None,
    ) -> None:
        self.verbosity = verbosity
        self.console = console or Console()
        self.log_file = log_file
        self.steps: list[StepLog] = []
        self._timer_stack: list[tuple[str, float]] = []

    def info(self, msg: str, **kwargs: object) -> None:
        if self.verbosity >= Verbosity.NORMAL:
            self.console.print(f"[bold blue]lens[/] {msg}", **kwargs)

    def success(self, msg: str, **kwargs: object) -> None:
        if self.verbosity >= Verbosity.NORMAL:
            self.console.print(f"[bold green]lens[/] {msg}", **kwargs)

    def warn(self, msg: str, **kwargs: object) -> None:
        if self.verbosity >= Verbosity.QUIET:
            self.console.print(f"[bold yellow]lens[/] {msg}", **kwargs)

    def error(self, msg: str, **kwargs: object) -> None:
        self.console.print(f"[bold red]lens[/] {msg}", **kwargs)

    def debug(self, msg: str, **kwargs: object) -> None:
        if self.verbosity >= Verbosity.DEBUG:
            self.console.print(f"[dim]lens[/] {msg}", **kwargs)

    def verbose(self, msg: str, **kwargs: object) -> None:
        if self.verbosity >= Verbosity.VERBOSE:
            self.console.print(f"[blue]lens[/] {msg}", **kwargs)

    def start_step(self, step: str) -> None:
        self._timer_stack.append((step, time.monotonic()))

    def end_step(
        self,
        message: str = "",
        scope_id: str | None = None,
        checkpoint: int | None = None,
        **extra: object,
    ) -> StepLog:
        if not self._timer_stack:
            msg = "end_step called without matching start_step"
            raise RuntimeError(msg)
        step, start = self._timer_stack.pop()
        elapsed = (time.monotonic() - start) * 1000
        log = StepLog(
            step=step,
            scope_id=scope_id,
            checkpoint=checkpoint,
            message=message,
            elapsed_ms=elapsed,
            extra=dict(extra),
        )
        self.steps.append(log)
        self.debug(f"{step}: {message} ({elapsed:.1f}ms)")
        return log

    def save_log(self, path: Path) -> None:
        """Write all recorded steps as JSON Lines to *path*."""
        import json

        path = Path(path) if not isinstance(path, Path) else path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for step in self.steps:
                f.write(json.dumps(step.to_dict()) + "\n")
