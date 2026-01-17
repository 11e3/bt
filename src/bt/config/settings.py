"""Centralized configuration management with Pydantic settings.

Provides environment-aware configuration with validation and type safety.
"""

from enum import Enum
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from bt.exceptions import ConfigurationError


class Environment(str, Enum):
    """Environment types for configuration management."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class BacktestBaseConfig(BaseSettings):
    """Base configuration with common settings."""

    # Environment configuration
    environment: Environment = Field(
        default=Environment.DEVELOPMENT, description="Application environment"
    )

    # Database configuration
    database_url: str = Field(default="sqlite:///bt_data.db", description="Database connection URL")

    # API configuration
    api_timeout: int = Field(default=30, ge=1, le=300, description="API request timeout in seconds")

    api_rate_limit: float = Field(
        default=0.1, ge=0.001, le=10.0, description="API rate limit requests per second"
    )

    # Cache configuration
    cache_ttl: int = Field(
        default=3600, ge=60, le=86400, description="Cache time-to-live in seconds"
    )

    cache_max_size: int = Field(default=1000, ge=100, le=10000, description="Maximum cache size")

    # Logging configuration
    log_level: str = Field(
        default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )

    log_format: str = Field(default="json", description="Log format (json, text)")

    log_file: str | None = Field(default=None, description="Log file path")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="BT_",
        case_sensitive=False,
    )


class StrategyConfig(BaseSettings):
    """Strategy-specific configuration."""

    # VBO strategy defaults
    vbo_lookback_period: int = Field(
        default=5, ge=1, le=100, description="VBO strategy lookback period"
    )

    vbo_multiplier: int = Field(
        default=2, ge=1, le=10, description="VBO strategy multiplier for long-term MA"
    )

    vbo_k_factor: float = Field(
        default=0.5, ge=0.1, le=3.0, description="VBO strategy K factor for breakout threshold"
    )

    # Momentum strategy defaults
    momentum_lookback_period: int = Field(
        default=20, ge=5, le=252, description="Momentum strategy lookback period"
    )

    momentum_top_n: int = Field(
        default=3, ge=1, le=20, description="Number of top assets for momentum allocation"
    )

    # General strategy defaults
    max_positions_per_symbol: int = Field(
        default=1, ge=1, le=10, description="Maximum positions per symbol"
    )

    min_allocation_pct: float = Field(
        default=0.01, ge=0.001, le=0.5, description="Minimum allocation percentage"
    )

    max_allocation_pct: float = Field(
        default=0.5, ge=0.1, le=1.0, description="Maximum allocation percentage"
    )

    model_config = SettingsConfigDict(
        env_prefix="BT_STRATEGY_",
        case_sensitive=False,
    )


class BacktestConfig(BaseSettings):
    """Main backtest configuration."""

    # Financial defaults
    initial_cash: float = Field(
        default=1000000.0, gt=0, description="Initial cash amount in base currency"
    )

    fee_rate: float = Field(
        default=0.0005, ge=0.0, le=0.01, description="Trading fee rate (0.0005 = 0.05%)"
    )

    slippage_rate: float = Field(
        default=0.0005, ge=0.0, le=0.01, description="Slippage rate (0.0005 = 0.05%)"
    )

    # Data configuration
    data_directory: str = Field(default="data", description="Directory for market data")

    default_symbols: list[str] = Field(
        default=["BTC", "ETH"], description="Default symbols to backtest"
    )

    start_date: str | None = Field(default=None, description="Backtest start date (YYYY-MM-DD)")

    end_date: str | None = Field(default=None, description="Backtest end date (YYYY-MM-DD)")

    # Performance configuration
    max_bars_per_strategy: int = Field(
        default=10000, ge=1000, le=100000, description="Maximum bars to process per strategy"
    )

    memory_limit_mb: int = Field(default=2048, ge=512, le=16384, description="Memory limit in MB")

    # Validation configuration
    validate_strategies: bool = Field(
        default=True, description="Validate strategy configurations before backtest"
    )

    skip_invalid_trades: bool = Field(default=False, description="Skip trades with invalid data")

    model_config = SettingsConfigDict(
        env_prefix="BT_",
        case_sensitive=False,
    )


class ReportingConfig(BaseSettings):
    """Reporting and visualization configuration."""

    # Chart configuration
    chart_style: str = Field(default="default", description="Chart style (default, dark, minimal)")

    chart_dpi: int = Field(default=100, ge=72, le=300, description="Chart DPI for image quality")

    chart_format: str = Field(default="png", description="Default chart format (png, pdf, svg)")

    # Report configuration
    report_directory: str = Field(default="output", description="Directory for generated reports")

    include_charts: bool = Field(default=True, description="Include charts in generated reports")

    include_raw_data: bool = Field(default=False, description="Include raw data in report exports")

    # Metrics configuration
    risk_free_rate: float = Field(
        default=0.02, ge=0.0, le=0.1, description="Risk-free rate for Sharpe ratio calculation"
    )

    var_confidence_level: float = Field(
        default=0.05, ge=0.01, le=0.1, description="Value at Risk confidence level (0.05 = 95% VaR)"
    )

    model_config = SettingsConfigDict(
        env_prefix="BT_REPORTING_",
        case_sensitive=False,
    )


class ConfigurationManager:
    """Centralized configuration manager."""

    def __init__(self, environment: str | None = None):
        """Initialize configuration manager.

        Args:
            environment: Target environment (development, testing, staging, production)
        """
        self._environment = environment
        self._configs: dict[str, BaseSettings] = {}
        self._load_configs()

    def _load_configs(self) -> None:
        """Load all configuration objects."""
        # Load all configuration objects
        # Environment is determined via env vars, not constructor params
        self._configs["base"] = BacktestBaseConfig()
        self._configs["strategy"] = StrategyConfig()
        self._configs["backtest"] = BacktestConfig()
        self._configs["reporting"] = ReportingConfig()

    def _get_environment(self) -> str:
        """Get current environment."""
        if self._environment:
            return self._environment

        # Try to get from environment variable
        import os

        return os.getenv("BT_ENV", Environment.DEVELOPMENT)

    def get_config(self, config_name: str) -> BaseSettings:
        """Get specific configuration object.

        Args:
            config_name: Name of configuration (base, strategy, backtest, reporting)

        Returns:
            Configuration object

        Raises:
            ConfigurationError: If config not found
        """
        if config_name not in self._configs:
            available = ", ".join(self._configs.keys())
            raise ConfigurationError(
                f"Unknown configuration: {config_name}. Available: {available}"
            )

        return self._configs[config_name]

    def get_base_config(self) -> BacktestBaseConfig:
        """Get base configuration."""
        config = self.get_config("base")
        assert isinstance(config, BacktestBaseConfig)
        return config

    def get_strategy_config(self) -> StrategyConfig:
        """Get strategy configuration."""
        config = self.get_config("strategy")
        assert isinstance(config, StrategyConfig)
        return config

    def get_backtest_config(self) -> BacktestConfig:
        """Get backtest configuration."""
        config = self.get_config("backtest")
        assert isinstance(config, BacktestConfig)
        return config

    def get_reporting_config(self) -> ReportingConfig:
        """Get reporting configuration."""
        config = self.get_config("reporting")
        assert isinstance(config, ReportingConfig)
        return config

    def get_env_value(self, key: str, default: Any = None) -> Any:
        """Get environment-specific value from configuration."""
        try:
            config = self.get_config(self._determine_config_type(key))
            return getattr(config, key, default)
        except Exception as e:
            raise ConfigurationError(f"Error getting config value {key}: {e}") from e

    def _determine_config_type(self, key: str) -> str:
        """Determine which config section contains a key."""
        # Strategy-related keys
        if any(
            key.startswith(prefix)
            for prefix in ["vbo_", "momentum_", "max_positions", "min_allocation"]
        ):
            return "strategy"

        # Reporting-related keys
        if any(key.startswith(prefix) for prefix in ["chart_", "report_", "risk_free_", "var_"]):
            return "reporting"

        # Default to backtest config
        return "backtest"

    def validate_configuration(self) -> list[str]:
        """Validate all configurations and return errors."""
        errors = []

        for config_name, config in self._configs.items():
            try:
                # Pydantic automatically validates on instantiation
                # Trigger validation by accessing a field
                config.model_dump()
            except Exception as e:
                errors.append(f"Configuration validation failed for {config_name}: {e}")

        return errors

    def set_config(self, config: dict[str, Any]) -> None:
        """Set runtime configuration overrides.

        Args:
            config: Dictionary of configuration overrides
        """
        # Store runtime overrides for later access
        self._runtime_config = config

    def get_runtime_config(self) -> dict[str, Any]:
        """Get runtime configuration overrides."""
        return getattr(self, "_runtime_config", {})

    def get_all_configs_dict(self) -> dict[str, dict[str, Any]]:
        """Get all configurations as nested dictionary."""
        return {name: config.model_dump() for name, config in self._configs.items()}

    def validate_backtest_parameters(self, params: dict[str, Any]) -> list[str]:
        """Validate backtest parameters against current configuration."""
        config = self.get_backtest_config()
        errors = []

        # Validate initial cash
        if "initial_cash" in params:
            cash = params["initial_cash"]
            if cash < config.initial_cash:
                errors.append(f"initial_cash must be >= {config.initial_cash}")

        # Validate fee rate
        if "fee_rate" in params:
            fee = params["fee_rate"]
            if not (0.0 <= fee <= config.fee_rate):
                errors.append(f"fee_rate must be between 0.0 and {config.fee_rate}")

        return errors


# Global configuration manager instance
_config_manager = None


def get_config_manager() -> ConfigurationManager:
    """Get the global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager


def set_config_manager(manager: ConfigurationManager) -> None:
    """Set the global configuration manager."""
    global _config_manager
    _config_manager = manager


# Environment-specific factory functions


def create_development_config() -> ConfigurationManager:
    """Create configuration for development environment."""
    return ConfigurationManager("development")


def create_testing_config() -> ConfigurationManager:
    """Create configuration for testing environment."""
    return ConfigurationManager("testing")


def create_staging_config() -> ConfigurationManager:
    """Create configuration for staging environment."""
    return ConfigurationManager("staging")


def create_production_config() -> ConfigurationManager:
    """Create configuration for production environment."""
    return ConfigurationManager("production")


# Configuration validation functions


def validate_strategy_parameters(params: dict[str, Any]) -> list[str]:
    """Validate strategy parameters against current configuration."""
    config = get_config_manager().get_strategy_config()
    errors = []

    # Validate VBO parameters
    if "vbo_lookback_period" in params:
        lookback = params["vbo_lookback_period"]
        if not (1 <= lookback <= config.vbo_lookback_period):
            errors.append(f"vbo_lookback_period must be between 1 and {config.vbo_lookback_period}")

    # Validate momentum parameters
    if "momentum_lookback_period" in params:
        lookback = params["momentum_lookback_period"]
        if not (5 <= lookback <= config.momentum_lookback_period):
            errors.append(
                f"momentum_lookback_period must be between 5 and {config.momentum_lookback_period}"
            )

    return errors


def validate_backtest_parameters(params: dict[str, Any]) -> list[str]:
    """Validate backtest parameters against current configuration."""
    config = get_config_manager().get_backtest_config()
    errors = []

    # Validate initial cash
    if "initial_cash" in params:
        cash = params["initial_cash"]
        if cash < config.initial_cash:
            errors.append(f"initial_cash must be >= {config.initial_cash}")

    # Validate fee rate
    if "fee_rate" in params:
        fee = params["fee_rate"]
        if not (0.0 <= fee <= config.fee_rate):
            errors.append(f"fee_rate must be between 0.0 and {config.fee_rate}")

    return errors
