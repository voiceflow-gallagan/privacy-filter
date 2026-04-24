from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    model_name: str = "openai/privacy-filter"
    device: str = "cpu"
    default_mode: str = "balanced"

    max_text_length: int = 524_288
    chunk_size_tokens: int = 120_000
    chunk_overlap_tokens: int = 512
    smart_split: bool = True

    max_concurrent_inferences: int = 2

    max_batch_size: int = 32
    max_batch_total_tokens: int = 200_000

    rate_limit_enabled: bool = True
    rate_limit_per_ip: str = "30/10minutes"

    log_level: str = "info"


def get_settings() -> Settings:
    return Settings()
