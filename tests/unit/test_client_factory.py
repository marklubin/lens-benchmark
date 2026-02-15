from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from lens.agent.client_factory import create_llm_client
from lens.agent.llm_client import MockLLMClient
from lens.core.config import LLMConfig
from lens.core.errors import ConfigError


def _install_fake_openai():
    """Install a fake 'openai' module so OpenAIClient can be imported."""
    fake = types.ModuleType("openai")
    fake.OpenAI = MagicMock()
    sys.modules["openai"] = fake
    return fake


class TestClientFactory:
    def test_mock_provider(self):
        config = LLMConfig(provider="mock")
        client = create_llm_client(config)
        assert isinstance(client, MockLLMClient)

    def test_openai_missing_key_raises(self):
        config = LLMConfig(provider="openai", api_key=None)
        with pytest.raises(ConfigError, match="API key"):
            create_llm_client(config)

    def test_openai_with_key(self):
        fake = _install_fake_openai()
        try:
            # Clear cached module so it re-imports with our fake
            sys.modules.pop("lens.agent.openai_client", None)

            config = LLMConfig(provider="openai", api_key="sk-test", model="gpt-4o-mini")
            client = create_llm_client(config)
            from lens.agent.openai_client import OpenAIClient

            assert isinstance(client, OpenAIClient)
            fake.OpenAI.assert_called_once()
        finally:
            sys.modules.pop("openai", None)
            sys.modules.pop("lens.agent.openai_client", None)

    def test_unknown_provider_raises(self):
        config = LLMConfig(provider="unknown_xyz")
        with pytest.raises(ConfigError, match="Unknown LLM provider"):
            create_llm_client(config)

    def test_env_override(self):
        config = LLMConfig(provider="mock")
        with patch.dict("os.environ", {"LENS_LLM_PROVIDER": "mock"}):
            client = create_llm_client(config)
            assert isinstance(client, MockLLMClient)
