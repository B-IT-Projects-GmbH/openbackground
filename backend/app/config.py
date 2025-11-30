"""Configuration settings for OpenBackground."""

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Configuration
    api_keys: str = "dev-key-change-me"
    api_title: str = "OpenBackground API"
    api_version: str = "1.0.0"

    # Model Configuration
    default_model: str = ""
    model_cache_dir: str = "./models"
    available_models: str = ""

    # Processing Configuration
    max_image_size: int = 4096
    model_input_size: int = 1024

    # Statistics Configuration
    enable_stats_persistence: bool = False
    stats_db_path: str = "./data/stats.db"

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def api_keys_list(self) -> List[str]:
        """Parse comma-separated API keys into a list."""
        return [key.strip() for key in self.api_keys.split(",") if key.strip()]

    @property
    def available_models_list(self) -> List[str]:
        """Parse comma-separated model names into a list."""
        return [model.strip() for model in self.available_models.split(",") if model.strip()]


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

