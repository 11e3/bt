"""Comprehensive test configuration and fixtures for pytest."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from bt.config import ConfigurationManager
from bt.core.container import Container, get_default_container
from bt.core.registry import get_strategy_registry
from bt.framework import BacktestFramework

# Test data generators
from tests.fixtures.data_generators import (
    BacktestResultGenerator,
    MarketDataGenerator,
    PerformanceScenarioGenerator,
    create_performance_test_scenarios,
    create_test_backtest_result,
    create_test_market_data,
)

# === PYTEST CONFIGURATION ===


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "performance: mark test as performance test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "flaky: mark test as potentially flaky")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on path."""
    for item in items:
        # Mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Mark performance tests
        if "performance" in str(item.fspath) or "benchmark" in item.name:
            item.add_marker(pytest.mark.performance)


# === BASIC FIXTURES ===


@pytest.fixture(scope="session")
def test_config_manager() -> ConfigurationManager:
    """Provide test configuration manager."""
    manager = ConfigurationManager("testing")
    yield manager


@pytest.fixture(scope="session")
def test_container(test_config_manager) -> Container:
    """Provide test dependency injection container."""
    container = Container("test")

    # Register test services
    container.register_singleton(type(test_config_manager), test_config_manager)

    # Mock logger for testing
    mock_logger = MagicMock()
    container.register_singleton(type(mock_logger), mock_logger)

    yield container


@pytest.fixture
def framework(test_container) -> BacktestFramework:
    """Provide BacktestFramework instance for testing."""
    return BacktestFramework(container=test_container)


# === MARKET DATA FIXTURES ===


@pytest.fixture(scope="session")
def market_data_generator() -> MarketDataGenerator:
    """Provide market data generator with fixed seed for reproducibility."""
    return MarketDataGenerator(seed=42)


@pytest.fixture
def sample_market_data(market_data_generator) -> dict[str, pd.DataFrame]:
    """Provide sample market data for testing."""
    return create_test_market_data(
        symbols=["BTC", "ETH", "ADA"], periods=100, start_date="2020-01-01"
    )


@pytest.fixture
def single_symbol_data(market_data_generator) -> pd.DataFrame:
    """Provide single symbol market data."""
    return market_data_generator.generate_ohlcv_data(
        symbol="BTC",
        start_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
        periods=50,
        base_price=50000.0,
        volatility=0.03,
    )


@pytest.fixture
def correlated_market_data(market_data_generator) -> dict[str, pd.DataFrame]:
    """Provide correlated multi-symbol market data."""
    # Create correlation matrix (BTC and ETH correlated, ADA uncorrelated)
    correlation_matrix = np.array(
        [
            [1.0, 0.7, 0.2],  # BTC correlations
            [0.7, 1.0, 0.3],  # ETH correlations
            [0.2, 0.3, 1.0],  # ADA correlations
        ]
    )

    return market_data_generator.generate_multi_symbol_data(
        symbols=["BTC", "ETH", "ADA"],
        start_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
        periods=100,
        correlation_matrix=correlation_matrix,
        volatility=0.025,
    )


@pytest.fixture
def high_volatility_data(market_data_generator) -> pd.DataFrame:
    """Provide high volatility market data for stress testing."""
    return market_data_generator.generate_ohlcv_data(
        symbol="VOLATILE",
        start_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
        periods=50,
        base_price=100.0,
        volatility=0.1,  # 10% daily volatility
        trend=0.0,  # No trend
    )


# === BACKTEST RESULT FIXTURES ===


@pytest.fixture(scope="session")
def backtest_result_generator() -> BacktestResultGenerator:
    """Provide backtest result generator."""
    return BacktestResultGenerator(seed=42)


@pytest.fixture
def sample_backtest_result(backtest_result_generator) -> dict[str, Any]:
    """Provide sample backtest result."""
    return create_test_backtest_result(
        strategy_name="test_strategy",
        total_return=0.15,
        sharpe_ratio=1.5,
        max_drawdown=-0.12,
        win_rate=0.55,
        num_trades=25,
    )


@pytest.fixture
def profitable_backtest_result(backtest_result_generator) -> dict[str, Any]:
    """Provide highly profitable backtest result."""
    return backtest_result_generator.generate_backtest_result(
        strategy_name="profitable_strategy",
        total_return=0.85,  # 85% return
        sharpe_ratio=2.1,
        max_drawdown=-0.08,
        win_rate=0.65,
        num_trades=40,
    )


@pytest.fixture
def losing_backtest_result(backtest_result_generator) -> dict[str, Any]:
    """Provide losing backtest result."""
    return backtest_result_generator.generate_backtest_result(
        strategy_name="losing_strategy",
        total_return=-0.25,  # -25% return
        sharpe_ratio=-0.8,
        max_drawdown=-0.35,
        win_rate=0.35,
        num_trades=30,
    )


# === PERFORMANCE SCENARIO FIXTURES ===


@pytest.fixture(scope="session")
def performance_scenario_generator() -> PerformanceScenarioGenerator:
    """Provide performance scenario generator."""
    return PerformanceScenarioGenerator(seed=42)


@pytest.fixture
def crash_scenario(performance_scenario_generator) -> dict[str, Any]:
    """Provide market crash scenario data."""
    return performance_scenario_generator.generate_crash_scenario()


@pytest.fixture
def bull_market_scenario(performance_scenario_generator) -> dict[str, Any]:
    """Provide bull market scenario data."""
    return performance_scenario_generator.generate_bull_market_scenario()


@pytest.fixture
def sideways_market_scenario(performance_scenario_generator) -> dict[str, Any]:
    """Provide sideways market scenario data."""
    return performance_scenario_generator.generate_sideways_market_scenario()


@pytest.fixture
def all_performance_scenarios(performance_scenario_generator) -> dict[str, dict[str, Any]]:
    """Provide all performance test scenarios."""
    return create_performance_test_scenarios()


# === INTEGRATION TEST FIXTURES ===


@pytest.fixture
def integration_container() -> Container:
    """Provide container configured for integration testing."""
    container = Container("integration")

    # Register real services for integration testing
    # Note: In a real implementation, these would be the actual service implementations

    # Mock services that require external dependencies
    mock_logger = MagicMock()
    container.register_singleton(type(mock_logger), mock_logger)

    yield container


@pytest.fixture
def integration_framework(integration_container) -> BacktestFramework:
    """Provide framework configured for integration testing."""
    return BacktestFramework(container=integration_container)


@pytest.fixture
def full_backtest_setup(integration_framework, sample_market_data) -> dict[str, Any]:
    """Provide complete backtest setup for integration testing."""
    return {
        "framework": integration_framework,
        "market_data": sample_market_data,
        "config": {
            "initial_cash": 1000000,
            "fee_rate": 0.0005,
            "slippage_rate": 0.0005,
        },
    }


# === PERFORMANCE TESTING FIXTURES ===


@pytest.fixture
def benchmark_data(market_data_generator) -> dict[str, pd.DataFrame]:
    """Provide benchmark-sized data for performance testing."""
    return market_data_generator.generate_multi_symbol_data(
        symbols=["BTC", "ETH", "ADA", "SOL", "DOT", "LINK", "UNI", "AAVE"],
        start_date=datetime(2018, 1, 1, tzinfo=timezone.utc),
        periods=1000,  # ~3 years of daily data
        volatility=0.03,
    )


@pytest.fixture
def large_portfolio_data(market_data_generator) -> dict[str, pd.DataFrame]:
    """Provide data for testing large portfolios."""
    return market_data_generator.generate_multi_symbol_data(
        symbols=[f"ASSET_{i:03d}" for i in range(50)],  # 50 assets
        start_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
        periods=252,
        volatility=0.02,
    )


# === STRATEGY TESTING FIXTURES ===


@pytest.fixture
def mock_strategy() -> MagicMock:
    """Provide mock strategy for testing."""
    strategy = MagicMock()
    strategy.get_name.return_value = "mock_strategy"
    strategy.get_buy_conditions.return_value = {}
    strategy.get_sell_conditions.return_value = {}
    strategy.get_buy_price_func.return_value = lambda e, s: 100.0
    strategy.get_sell_price_func.return_value = lambda e, s: 100.0
    strategy.get_allocation_func.return_value = lambda e, s, p: 100
    strategy.validate.return_value = []
    return strategy


@pytest.fixture
def vbo_strategy_config() -> dict[str, Any]:
    """Provide VBO strategy configuration."""
    return {
        "lookback": 5,
        "multiplier": 2,
        "k_factor": 0.5,
        "top_n": 3,
        "mom_lookback": 20,
    }


@pytest.fixture
def momentum_strategy_config() -> dict[str, Any]:
    """Provide momentum strategy configuration."""
    return {
        "lookback": 20,
    }


# === UTILITY FIXTURES ===


@pytest.fixture
def temp_data_dir(tmp_path) -> Path:
    """Provide temporary directory for test data."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def clean_registry():
    """Ensure strategy registry is clean between tests."""
    registry = get_strategy_registry()
    original_strategies = registry._strategies.copy()

    yield

    # Restore original registry state
    registry._strategies = original_strategies


@pytest.fixture(autouse=True)
def reset_container():
    """Reset default container between tests."""
    original_container = get_default_container()
    test_container = Container("test_reset")

    # Temporarily replace default container
    import bt.core.container

    bt.core.container._default_container = test_container

    yield

    # Restore original container
    bt.core.container._default_container = original_container


# === ASYNC TESTING SUPPORT ===


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    import asyncio

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# === CUSTOM ASSERTION HELPERS ===


class BacktestAssertions:
    """Custom assertions for backtest result validation."""

    @staticmethod
    def assert_valid_backtest_result(result: dict[str, Any]):
        """Assert that a backtest result has all required fields."""
        required_keys = ["strategy", "performance", "trades", "equity_curve"]
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

        # Check performance metrics
        perf = result["performance"]
        required_metrics = ["total_return", "sharpe", "mdd", "win_rate", "num_trades"]
        for metric in required_metrics:
            assert metric in perf, f"Missing performance metric: {metric}"

        # Check equity curve
        equity = result["equity_curve"]
        assert "dates" in equity and "values" in equity
        assert len(equity["dates"]) == len(equity["values"])
        assert len(equity["dates"]) > 0

    @staticmethod
    def assert_reasonable_performance(
        result: dict[str, Any], expected_return_range: tuple = (-1.0, 2.0)
    ):
        """Assert that performance metrics are within reasonable bounds."""
        perf = result["performance"]

        # Check total return is reasonable
        total_return = perf["total_return"] / 100.0  # Convert from percentage
        assert expected_return_range[0] <= total_return <= expected_return_range[1], (
            f"Total return {total_return} outside expected range {expected_return_range}"
        )

        # Check Sharpe ratio is reasonable
        sharpe = perf["sharpe"]
        assert -5.0 <= sharpe <= 5.0, f"Sharpe ratio {sharpe} seems unreasonable"

        # Check win rate is between 0 and 100
        win_rate = perf["win_rate"]
        assert 0 <= win_rate <= 100, f"Win rate {win_rate} must be between 0 and 100"


@pytest.fixture
def backtest_assertions() -> BacktestAssertions:
    """Provide backtest assertion helpers."""
    return BacktestAssertions()
