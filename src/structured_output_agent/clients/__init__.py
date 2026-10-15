"""Generation client interfaces and offline implementations."""
from .base import GenerationRequest, LLMClient
from .fake_client import FakeLLMClient

__all__ = ["GenerationRequest", "LLMClient", "FakeLLMClient"]


def create_live_client(config):
    """Select a live adapter; fake scenarios keep explicit scripted clients."""
    from structured_output_agent.exceptions import ConfigurationError
    if config.mode != "live":
        raise ConfigurationError("Live client factory requires live mode.")
    if config.llm_provider == "codex":
        try:
            from .codex_client import CodexClient
        except ImportError:
            raise ConfigurationError("Install the codex optional dependency to use Codex.") from None
        return CodexClient.from_config(config)
    if config.llm_provider == "gemini":
        from .gemini_client import GeminiClient
        return GeminiClient.from_config(config)
    from .openai_client import OpenAIClient
    return OpenAIClient.from_config(config)
