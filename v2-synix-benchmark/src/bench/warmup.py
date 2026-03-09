"""Modal endpoint readiness poller.

Blocks until both LLM and embedding endpoints are responding, with
configurable timeout and poll interval. Use this before benchmark runs
to avoid cold-start failures.

Usage:
    from bench.warmup import wait_for_modal

    wait_for_modal(
        llm_base_url="https://synix--lens-llm-llm-serve.modal.run/v1",
        embed_base_url="https://synix--lens-embed-serve.modal.run",
    )
    # Both endpoints are now warm — safe to start benchmark
"""
from __future__ import annotations

import logging
import time

import httpx

logger = logging.getLogger(__name__)


class ModalNotReady(TimeoutError):
    """Raised when Modal endpoints don't become ready within the timeout."""


def wait_for_modal(
    *,
    llm_base_url: str,
    embed_base_url: str,
    timeout: float = 300.0,
    poll_interval: float = 5.0,
    llm_model: str = "Qwen/Qwen3.5-35B-A3B",
    embed_model: str = "Alibaba-NLP/gte-modernbert-base",
) -> dict[str, float]:
    """Block until both Modal endpoints respond successfully.

    Returns dict with warmup latencies: {"llm_ms": ..., "embed_ms": ..., "total_s": ...}
    Raises ModalNotReady if timeout is exceeded.
    """
    deadline = time.monotonic() + timeout
    llm_ready = False
    embed_ready = False
    llm_ms = 0.0
    embed_ms = 0.0

    logger.info("Waiting for Modal endpoints (timeout=%.0fs)...", timeout)

    while time.monotonic() < deadline:
        if not llm_ready:
            llm_ready, llm_ms = _check_llm(llm_base_url, llm_model)
            if llm_ready:
                logger.info("LLM endpoint ready (%.0fms)", llm_ms)

        if not embed_ready:
            embed_ready, embed_ms = _check_embed(embed_base_url, embed_model)
            if embed_ready:
                logger.info("Embed endpoint ready (%.0fms)", embed_ms)

        if llm_ready and embed_ready:
            total_s = timeout - (deadline - time.monotonic())
            logger.info("Both endpoints ready in %.1fs", total_s)
            return {"llm_ms": llm_ms, "embed_ms": embed_ms, "total_s": total_s}

        remaining = deadline - time.monotonic()
        if remaining > 0:
            wait = min(poll_interval, remaining)
            status = []
            if not llm_ready:
                status.append("LLM")
            if not embed_ready:
                status.append("embed")
            logger.info("Waiting for %s... (%.0fs remaining)", " + ".join(status), remaining)
            time.sleep(wait)

    not_ready = []
    if not llm_ready:
        not_ready.append("LLM")
    if not embed_ready:
        not_ready.append("embed")
    raise ModalNotReady(f"Endpoints not ready after {timeout}s: {', '.join(not_ready)}")


def _check_llm(base_url: str, model: str) -> tuple[bool, float]:
    """Send a minimal completion request to the LLM endpoint."""
    try:
        t0 = time.monotonic()
        resp = httpx.post(
            f"{base_url}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Say ok"}],
                "max_tokens": 4,
                "temperature": 0.0,
            },
            timeout=30.0,
        )
        ms = (time.monotonic() - t0) * 1000
        if resp.status_code == 200:
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if content:
                return True, ms
        logger.debug("LLM check: status=%d", resp.status_code)
        return False, ms
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
        logger.debug("LLM check failed: %s", e)
        return False, 0.0


def _check_embed(base_url: str, model: str) -> tuple[bool, float]:
    """Send a minimal embedding request to the embed endpoint."""
    try:
        t0 = time.monotonic()
        resp = httpx.post(
            f"{base_url}/embeddings",
            json={"model": model, "input": ["test"]},
            timeout=30.0,
        )
        ms = (time.monotonic() - t0) * 1000
        if resp.status_code == 200:
            data = resp.json()
            if data.get("data") and len(data["data"][0].get("embedding", [])) > 0:
                return True, ms
        logger.debug("Embed check: status=%d", resp.status_code)
        return False, ms
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
        logger.debug("Embed check failed: %s", e)
        return False, 0.0
