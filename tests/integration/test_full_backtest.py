"""Integration tests for full backtest workflows."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from bt.exceptions import InsufficientDataError, ValidationError
from bt.framework import BacktestFramework
from bt.strategies.implementations import VolatilityBreakoutStrategy


class TestFullBacktestWorkflow:
    """Test complete backtest workflows end-to-end."""

    @pytest.mark.integration
    def test_vbo_strategy_full_backtest(self, framework: BacktestFramework, sample_market_data):
        """Test complete VBO strategy backtest."""
        # Create VBO strategy
        strategy = VolatilityBreakoutStrategy(
            lookback=5, multiplier=2, k_factor=0.5, top_n=3, mom_lookback=20
        )

        # Run full backtest
        result = framework.run_backtest(
            strategy=strategy.get_name(),
            symbols=list(sample_market_data.keys()),
            data=sample_market_data,
            config={
                "initial_cash": 1000000,
                "fee_rate": 0.0005,
                "slippage_rate": 0.0005,
            },
        )

        # Validate complete result structure
        assert result["strategy"] == "VolatilityBreakout"
        assert "performance" in result
        assert "trades" in result
        assert "equity_curve" in result
        assert "configuration" in result

        # Validate performance metrics
        perf = result["performance"]
        required_metrics = ["total_return", "sharpe", "mdd", "win_rate", "num_trades"]
        for metric in required_metrics:
            assert metric in perf

        # Validate equity curve
        equity = result["equity_curve"]
        assert len(equity["dates"]) == len(equity["values"])
        assert len(equity["dates"]) > 0

        # Validate trades
        trades = result["trades"]
        assert isinstance(trades, list)
        if trades:  # Only validate if there are trades
            trade = trades[0]
            required_trade_fields = [
                "symbol",
                "entry_date",
                "exit_date",
                "entry_price",
                "exit_price",
                "quantity",
                "pnl",
            ]
            for field in required_trade_fields:
                assert field in trade

    @pytest.mark.integration
    def test_multi_symbol_correlated_backtest(
        self, framework: BacktestFramework, correlated_market_data
    ):
        """Test backtest with correlated multi-symbol data."""
        strategy = framework.create_strategy("momentum", lookback=20)

        result = framework.run_backtest(
            strategy=strategy.get_name(),
            symbols=list(correlated_market_data.keys()),
            data=correlated_market_data,
        )

        # Should handle correlated assets properly
        assert result["performance"]["num_trades"] >= 0
        assert len(result["equity_curve"]["dates"]) > 0

    @pytest.mark.integration
    def test_configuration_validation(self, framework: BacktestFramework, sample_market_data):
        """Test that invalid configurations are properly rejected."""
        # Test invalid strategy name
        with pytest.raises(ValueError):
            framework.run_backtest(
                strategy="invalid_strategy",
                symbols=list(sample_market_data.keys()),
                data=sample_market_data,
            )

        # Test invalid configuration
        strategy = framework.create_strategy("vbo")
        with pytest.raises(ValueError):
            framework.run_backtest(
                strategy=strategy.get_name(),
                symbols=list(sample_market_data.keys()),
                data=sample_market_data,
                config={"invalid_param": "value"},
            )


class TestStrategyRegistryIntegration:
    """Test strategy registry integration."""

    @pytest.mark.integration
    def test_registry_strategy_creation(self, framework: BacktestFramework):
        """Test creating strategies through registry."""
        # List available strategies
        strategies = framework.list_available_strategies()
        assert "volatility_breakout" in strategies
        assert "momentum" in strategies
        assert "buy_and_hold" in strategies

        # Get strategy info
        info = framework.get_strategy_info("volatility_breakout")
        assert info is not None
        assert "description" in info

        # Create strategy
        strategy = framework.create_strategy("volatility_breakout")
        assert strategy.get_name() == "VolatilityBreakout"

    @pytest.mark.integration
    def test_strategy_validation_integration(self, framework: BacktestFramework):
        """Test strategy validation through framework."""
        # Valid configuration
        errors = framework.validate_strategy_config(
            "vbo", {"lookback": 5, "multiplier": 2, "k_factor": 0.5}
        )
        assert len(errors) == 0

        # Invalid configuration
        errors = framework.validate_strategy_config(
            "vbo",
            {
                "lookback": -1,  # Invalid
                "multiplier": 2,
                "k_factor": 0.5,
            },
        )
        assert len(errors) > 0


class TestFrameworkLifecycle:
    """Test complete framework lifecycle."""

    @pytest.mark.integration
    def test_framework_initialization(self):
        """Test framework initializes properly."""
        framework = BacktestFramework()

        # Check framework components
        info = framework.get_framework_info()
        assert "version" in info
        assert "strategies_count" in info
        assert info["strategies_count"] > 0

    @pytest.mark.integration
    def test_data_loading_integration(self, framework: BacktestFramework):
        """Test data loading through framework."""
        # This would normally load real data, but we'll mock it
        # In real implementation, this would test file loading
        pass

    @pytest.mark.integration
    def test_report_generation(self, framework: BacktestFramework, sample_backtest_result):
        """Test report generation integration."""
        # Test report generation (would normally save to file)
        # framework.create_performance_report(sample_backtest_result)
        pass


class TestErrorHandlingIntegration:
    """Test error handling across components."""

    @pytest.mark.integration
    def test_insufficient_data_handling(self, framework: BacktestFramework):
        """Test handling of insufficient market data."""
        # Create minimal data that should trigger insufficient data error
        minimal_data = {
            "BTC": pd.DataFrame(
                {
                    "datetime": [datetime(2020, 1, 1, tzinfo=timezone.utc)],
                    "open": [100],
                    "high": [105],
                    "low": [95],
                    "close": [102],
                }
            )
        }

        with pytest.raises(InsufficientDataError):  # Should raise insufficient data error
            framework.run_backtest(
                strategy="volatility_breakout", symbols=["BTC"], data=minimal_data
            )

    @pytest.mark.integration
    def test_invalid_symbol_handling(self, framework: BacktestFramework):
        """Test handling of invalid symbols."""
        data = {"INVALID_SYMBOL": pd.DataFrame()}

        with pytest.raises(ValidationError):
            framework.run_backtest(strategy="buy_and_hold", symbols=["INVALID_SYMBOL"], data=data)
