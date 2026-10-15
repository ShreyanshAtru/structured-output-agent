"""Validated environment settings; instantiation has no network side effects."""
from typing import Literal, Self
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentConfig(BaseSettings):
    # Environment strings intentionally use settings coercion, unlike output models.
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_ignore_empty=True,
        extra="forbid", populate_by_name=True, frozen=True, hide_input_in_errors=True,
    )
    mode: Literal["fake", "live"] = Field(default="fake", validation_alias="AGENT_MODE")
    llm_provider: Literal["gemini", "openai", "codex"] = Field(default="gemini", validation_alias="LLM_PROVIDER")
    codex_model: str | None = Field(default=None, min_length=1, validation_alias="CODEX_MODEL")
    codex_timeout_seconds: float = Field(default=90.0, gt=0, le=300, allow_inf_nan=False, validation_alias="CODEX_TIMEOUT_SECONDS")
    codex_work_dir: str = Field(default="work/codex-agent", min_length=1, validation_alias="CODEX_WORK_DIR")
    gemini_api_key: SecretStr | None = Field(default=None, validation_alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-3.8-flash", min_length=1, validation_alias="GEMINI_MODEL")
    gemini_timeout_seconds: float = Field(default=30.0, gt=0, allow_inf_nan=False, validation_alias="GEMINI_TIMEOUT_SECONDS")
    gemini_max_output_tokens: int = Field(default=4096, ge=1, validation_alias="GEMINI_MAX_OUTPUT_TOKENS")
    max_attempts: int = Field(default=3, ge=1, le=10, validation_alias="AGENT_MAX_ATTEMPTS")
    max_tool_calls: int = Field(default=5, ge=0, le=20, validation_alias="AGENT_MAX_TOOL_CALLS")
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_model: str | None = Field(default=None, validation_alias="OPENAI_MODEL")
    openai_timeout_seconds: float = Field(default=30.0, gt=0, allow_inf_nan=False, validation_alias="OPENAI_TIMEOUT_SECONDS")
    openai_max_output_tokens: int = Field(default=2000, ge=1, validation_alias="OPENAI_MAX_OUTPUT_TOKENS")
    observability_backend: Literal["local", "logfire"] = Field(
        default="local", validation_alias="OBSERVABILITY_BACKEND"
    )
    observability_log_path: str = Field(default="logs/events.jsonl", min_length=1, validation_alias="OBSERVABILITY_LOG_PATH")
    logfire_token: SecretStr | None = Field(default=None, validation_alias="LOGFIRE_TOKEN")

    @model_validator(mode="after")
    def check_live_configuration(self) -> Self:
        if self.mode == "live" and self.llm_provider == "gemini":
            if self.gemini_api_key is None or not self.gemini_api_key.get_secret_value().strip():
                raise ValueError("live Gemini mode requires GEMINI_API_KEY")
            if not self.gemini_model.strip():
                raise ValueError("live Gemini mode requires GEMINI_MODEL")
        if self.mode == "live" and self.llm_provider == "openai":
            if self.openai_api_key is None or not self.openai_api_key.get_secret_value().strip():
                raise ValueError("live mode requires OPENAI_API_KEY")
            if self.openai_model is None or not self.openai_model.strip():
                raise ValueError("live mode requires OPENAI_MODEL")
        if self.codex_model is not None and not self.codex_model.strip():
            raise ValueError("CODEX_MODEL must be nonblank or omitted")
        if self.observability_backend == "logfire":
            if self.logfire_token is None or not self.logfire_token.get_secret_value().strip():
                raise ValueError("logfire backend requires LOGFIRE_TOKEN")
        return self
