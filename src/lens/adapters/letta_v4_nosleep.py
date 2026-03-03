"""Letta V4 No-Sleep — Ablation adapter without active renewal.

Subclass of LettaV4Adapter that skips the sleep/consolidation phase.
Tests directed compression + timescale partition WITHOUT active renewal.

If V4-nosleep scores significantly below V4, renewal is doing real work.
If scores are similar, the gain comes from compression + partition alone.
"""
from __future__ import annotations

import logging

from lens.adapters.letta_v4 import LettaV4Adapter
from lens.adapters.registry import register_adapter

log = logging.getLogger(__name__)


@register_adapter("letta-v4-nosleep")
class LettaV4NoSleepAdapter(LettaV4Adapter):
    """V4 without sleep consolidation — tests directed compression + partition only."""

    def prepare(self, scope_id: str, checkpoint: int) -> None:
        """Skip consolidation entirely. No sleep agent runs."""
        log.info(
            "V4-nosleep: skipping consolidation at checkpoint %d (scope %s)",
            checkpoint, scope_id,
        )
