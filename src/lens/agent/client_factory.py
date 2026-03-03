from __future__ import annotations

from lens.agent.llm_client import BaseLLMClient, MockLLMClient
from lens.core.config import LLMConfig
from lens.core.errors import ConfigError


def create_llm_client(
    config: LLMConfig,
    cache_dir: str | None = None,
) -> BaseLLMClient:
    """Create an LLM client from configuration.

    Args:
        config: LLM configuration with provider, model, api_key, etc.
        cache_dir: Optional directory for LLM response caching.
            Also reads from LENS_LLM_CACHE_DIR env var.

    Returns:
        A BaseLLMClient instance.

    Raises:
        ConfigError: If the provider is unknown or required config is missing.
    """
    import os

    config = config.resolve_env()
    cache_dir = cache_dir or os.environ.get("LENS_LLM_CACHE_DIR")

    if config.provider == "mock":
        return MockLLMClient()

    if config.provider == "static":
        if not config.api_key:
            raise ConfigError(
                "Static provider requires an API key for synthesis. "
                "Set LENS_LLM_API_KEY or pass --api-key."
            )
        import json
        from pathlib import Path

        from lens.agent.static_driver import StaticLLMClient

        plans: dict = {}
        if config.query_plan:
            plan_path = Path(config.query_plan)
            if plan_path.exists():
                raw = json.loads(plan_path.read_text())
                plans = raw.get("plans", raw)
            else:
                import logging
                logging.getLogger(__name__).warning(
                    "Query plan file not found: %s — using default plans",
                    config.query_plan,
                )

        return StaticLLMClient(
            plans=plans,
            api_key=config.api_key,
            model=config.model,
            base_url=config.api_base,
            temperature=config.temperature,
            seed=config.seed,
        )

    if config.provider in ("modal", "openai"):
        if not config.api_key:
            raise ConfigError(
                "Modal provider requires an API key. "
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
            cache_dir=cache_dir,
        )

    if config.provider == "anthropic":
        api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ConfigError(
                "Anthropic provider requires an API key. "
                "Set ANTHROPIC_API_KEY or pass --api-key."
            )
        from lens.agent.anthropic_client import AnthropicClient

        return AnthropicClient(
            api_key=api_key,
            model=config.model,
            temperature=config.temperature,
            max_tokens=4096,
            cache_dir=cache_dir,
        )

    raise ConfigError(
        f"Unknown LLM provider: {config.provider!r}. "
        f"Available providers: mock, modal, anthropic, static"
    )
