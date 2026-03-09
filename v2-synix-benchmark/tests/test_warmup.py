"""Tests for bench.warmup — Modal endpoint readiness poller."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from bench.warmup import ModalNotReady, wait_for_modal


class TestWarmupSuccess:
    def test_returns_immediately_when_both_ready(self) -> None:
        with (
            patch("bench.warmup.httpx") as mock_httpx,
            patch("bench.warmup.time") as mock_time,
        ):
            # Both endpoints respond on first try
            llm_resp = MagicMock()
            llm_resp.status_code = 200
            llm_resp.json.return_value = {
                "choices": [{"message": {"content": "ok"}}],
            }

            embed_resp = MagicMock()
            embed_resp.status_code = 200
            embed_resp.json.return_value = {
                "data": [{"embedding": [0.1, 0.2]}],
            }

            mock_httpx.post.side_effect = [llm_resp, embed_resp]
            mock_httpx.ConnectError = ConnectionError
            mock_httpx.ReadTimeout = TimeoutError
            mock_httpx.ConnectTimeout = TimeoutError

            mock_time.monotonic.side_effect = [
                0.0,   # deadline calc
                1.0,   # loop start
                1.0, 1.1,  # llm check timing
                1.1, 1.2,  # embed check timing
                1.2,   # total_s calc
            ]
            mock_time.sleep = MagicMock()

            result = wait_for_modal(
                llm_base_url="http://fake-llm/v1",
                embed_base_url="http://fake-embed",
                timeout=60.0,
            )

        assert result["llm_ms"] > 0
        assert result["embed_ms"] > 0
        mock_time.sleep.assert_not_called()  # no polling needed


class TestWarmupRetry:
    def test_polls_until_ready(self) -> None:
        with (
            patch("bench.warmup.httpx") as mock_httpx,
            patch("bench.warmup.time") as mock_time,
        ):
            # First LLM call fails, second succeeds
            llm_fail = MagicMock()
            llm_fail.status_code = 503

            llm_ok = MagicMock()
            llm_ok.status_code = 200
            llm_ok.json.return_value = {"choices": [{"message": {"content": "ok"}}]}

            embed_ok = MagicMock()
            embed_ok.status_code = 200
            embed_ok.json.return_value = {"data": [{"embedding": [0.1]}]}

            mock_httpx.post.side_effect = [llm_fail, embed_ok, llm_ok]
            mock_httpx.ConnectError = ConnectionError
            mock_httpx.ReadTimeout = TimeoutError
            mock_httpx.ConnectTimeout = TimeoutError

            # Timeline: start=0, deadline=60, poll once at t=5, ready at t=10
            mock_time.monotonic.side_effect = [
                0.0,    # deadline
                5.0,    # loop 1 start
                5.0, 5.1,  # llm check 1 (fail)
                5.1, 5.2,  # embed check 1 (ok)
                5.2,    # remaining calc
                10.0,   # loop 2 start
                10.0, 10.1,  # llm check 2 (ok)
                10.1,   # total_s calc
            ]
            mock_time.sleep = MagicMock()

            result = wait_for_modal(
                llm_base_url="http://fake-llm/v1",
                embed_base_url="http://fake-embed",
                timeout=60.0,
                poll_interval=5.0,
            )

        assert result["llm_ms"] > 0
        assert result["embed_ms"] > 0
        mock_time.sleep.assert_called_once()  # polled once


class TestWarmupTimeout:
    def test_raises_after_timeout(self) -> None:
        with (
            patch("bench.warmup.httpx") as mock_httpx,
            patch("bench.warmup.time") as mock_time,
        ):
            # Always fails
            fail_resp = MagicMock()
            fail_resp.status_code = 503
            mock_httpx.post.return_value = fail_resp
            mock_httpx.ConnectError = ConnectionError
            mock_httpx.ReadTimeout = TimeoutError
            mock_httpx.ConnectTimeout = TimeoutError

            # Time passes deadline
            mock_time.monotonic.side_effect = [
                0.0,     # deadline = 10
                5.0,     # loop 1
                5.0, 5.1,  # llm check
                5.1, 5.2,  # embed check
                5.2,     # remaining
                11.0,    # loop 2 — past deadline
            ]
            mock_time.sleep = MagicMock()

            with pytest.raises(ModalNotReady, match="LLM, embed"):
                wait_for_modal(
                    llm_base_url="http://fake-llm/v1",
                    embed_base_url="http://fake-embed",
                    timeout=10.0,
                )

    def test_connection_error_retries(self) -> None:
        with (
            patch("bench.warmup.httpx") as mock_httpx,
            patch("bench.warmup.time") as mock_time,
        ):
            # Connection error then success
            mock_httpx.ConnectError = type("ConnectError", (Exception,), {})
            mock_httpx.ReadTimeout = type("ReadTimeout", (Exception,), {})
            mock_httpx.ConnectTimeout = type("ConnectTimeout", (Exception,), {})

            llm_ok = MagicMock()
            llm_ok.status_code = 200
            llm_ok.json.return_value = {"choices": [{"message": {"content": "ok"}}]}

            embed_ok = MagicMock()
            embed_ok.status_code = 200
            embed_ok.json.return_value = {"data": [{"embedding": [0.1]}]}

            mock_httpx.post.side_effect = [
                mock_httpx.ConnectError("refused"),  # llm fail
                mock_httpx.ConnectError("refused"),  # embed fail
                llm_ok,   # llm ok
                embed_ok,  # embed ok
            ]

            mock_time.monotonic.side_effect = [
                0.0,     # deadline
                1.0,     # loop 1
                1.0,     # t0 in _check_llm (before ConnectError)
                1.0,     # t0 in _check_embed (before ConnectError)
                1.0,     # remaining
                6.0,     # loop 2
                6.0, 6.1,  # llm check
                6.1, 6.2,  # embed check
                6.2,     # total_s
            ]
            mock_time.sleep = MagicMock()

            result = wait_for_modal(
                llm_base_url="http://fake-llm/v1",
                embed_base_url="http://fake-embed",
                timeout=60.0,
            )

        assert result["llm_ms"] > 0
