from functools import lru_cache
from models import AppConfig, ClickHouseConfig, OpenSearchConfig

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=True
    )

    clickhouse: ClickHouseConfig = Field(default_factory=ClickHouseConfig)
    opensearch: OpenSearchConfig = Field(default_factory=OpenSearchConfig)
    app: AppConfig = Field(default_factory=AppConfig)

    @property
    def log_level(self) -> str:
        """Convenience property for log level."""
        return self.app.log_level

    @property
    def environment(self) -> str:
        """Convenience property for environment."""
        return self.app.env


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are only loaded once.
    Call get_settings.cache_clear() to reload settings if needed.

    Returns:
        Settings instance with all configuration loaded.
    """
    return Settings()


# Convenience singleton for direct import
settings = get_settings()
