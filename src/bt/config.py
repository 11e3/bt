"""Application configuration using Pydantic Settings.

Environment variables can override default values.
Use .env file for local development.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Environment variables should be prefixed with BT_
    Example: BT_DATA_DIR=/path/to/data
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="BT_",
        case_sensitive=False,
    )

    # Data directory
    data_dir: Path = Field(
        default=Path("data"),
        description="Base directory for storing market data",
    )

    # Logging configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    log_format: str = Field(
        default="json",
        description="Log format (json or text)",
    )

    # API configuration
    api_timeout: int = Field(
        default=30,
        ge=1,
        description="API request timeout in seconds",
    )
    api_rate_limit: float = Field(
        default=0.1,
        ge=0,
        description="Minimum seconds between API calls",
    )

    # Backtest defaults
    default_initial_cash: int = Field(
        default=10_000_000,
        gt=0,
        description="Default initial cash in KRW",
    )
    default_fee: float = Field(
        default=0.0005,
        ge=0,
        lt=1,
        description="Default trading fee (0.0005 = 0.05%)",
    )
    default_slippage: float = Field(
        default=0.0005,
        ge=0,
        lt=1,
        description="Default slippage (0.0005 = 0.05%)",
    )


# Global settings instance
settings = Settings()
