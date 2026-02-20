from __future__ import annotations

from lens.agent.llm_client import BaseLLMClient, MockLLMClient
from lens.core.config import LLMConfig
from lens.core.errors import ConfigError


def create_llm_client(config: LLMConfig) -> BaseLLMClient:
    """Create an LLM client from configuration.

    Args:
        config: LLM configuration with provider, model, api_key, etc.

    Returns:
        A BaseLLMClient instance.

    Raises:
        ConfigError: If the provider is unknown or required config is missing.
    """
    config = config.resolve_env()

    if config.provider == "mock":
        return MockLLMClient()

    if config.provider == "openai":
        if not config.api_key:
            raise ConfigError(
                "OpenAI provider requires an API key. "
                "Set LENS_LLM_API_KEY or pass --api-key."
            )
        from lens.agent.openai_client import OpenAIClient

        return OpenAIClient(
            api_key=config.api_key,
            model=config.model,
            base_url=config.api_base,
            temperature=config.temperature,
            seed=config.seed,
            max_tokens=4096,
        )

    raise ConfigError(
        f"Unknown LLM provider: {config.provider!r}. "
        f"Available providers: mock, openai"
    )
