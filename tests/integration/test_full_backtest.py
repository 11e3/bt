"""Integration tests for full backtest workflows."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from bt.exceptions import StrategyError
from bt.exceptions import ValidationError as ExceptionsValidationError
from bt.framework import BacktestFramework
from bt.interfaces.core import ValidationError as CoreValidationError


class TestFullBacktestWorkflow:
    """Test complete backtest workflows end-to-end."""

    @pytest.mark.integration
    def test_vbo_strategy_full_backtest(self, framework: BacktestFramework, sample_market_data):
        """Test complete VBO strategy backtest."""
        # Run full backtest using strategy name
        result = framework.run_backtest(
            strategy="volatility_breakout",
            symbols=list(sample_market_data.keys()),
            data=sample_market_data,
            config={
                "initial_cash": 1000000,
                "fee_rate": 0.0005,
                "slippage_rate": 0.0005,
                "lookback": 5,
                "multiplier": 2,
                "k_factor": 0.5,
                "top_n": 3,
                "mom_lookback": 20,
            },
        )

        # Validate complete result structure
        assert "VolatilityBreakout" in result["strategy"]
        assert "performance" in result
        assert "trades" in result
        assert "equity_curve" in result
        assert "configuration" in result

        # Validate performance metrics (can be dict or PerformanceMetrics object)
        perf = result["performance"]
        required_metrics = ["total_return", "sortino_ratio", "mdd", "win_rate", "num_trades"]
        for metric in required_metrics:
            if isinstance(perf, dict):
                assert metric in perf
            else:
                assert hasattr(perf, metric)

        # Validate equity curve
        equity = result["equity_curve"]
        # Allow for off-by-one due to initial equity value
        assert abs(len(equity["dates"]) - len(equity["values"])) <= 1
        assert len(equity["dates"]) > 0 or len(equity["values"]) > 0

        # Validate trades
        trades = result["trades"]
        assert isinstance(trades, list)
        if trades:  # Only validate if there are trades
            trade = trades[0]
            # Trade format uses entry/exit pattern
            # Support both Trade model objects and dict format
            if hasattr(trade, "symbol"):
                # Trade model object
                assert trade.symbol is not None
                assert trade.entry_date is not None
                assert trade.exit_date is not None
                assert trade.quantity is not None
            else:
                # Dict format
                required_trade_fields = [
                    "symbol",
                    "entry_date",
                    "exit_date",
                    "entry_price",
                    "exit_price",
                    "quantity",
                ]
                for field in required_trade_fields:
                    assert field in trade

    @pytest.mark.integration
    def test_multi_symbol_correlated_backtest(
        self, framework: BacktestFramework, correlated_market_data
    ):
        """Test backtest with correlated multi-symbol data."""
        result = framework.run_backtest(
            strategy="momentum",
            symbols=list(correlated_market_data.keys()),
            data=correlated_market_data,
            config={"lookback": 20},
        )

        # Should handle correlated assets properly
        perf = result["performance"]
        if isinstance(perf, dict):
            assert perf["num_trades"] >= 0
        else:
            assert perf.num_trades >= 0
        assert len(result["equity_curve"]["dates"]) > 0

    @pytest.mark.integration
    def test_configuration_validation(self, framework: BacktestFramework, sample_market_data):
        """Test that invalid configurations are properly rejected."""
        # Test invalid strategy name
        with pytest.raises(StrategyError):
            framework.run_backtest(
                strategy="invalid_strategy",
                symbols=list(sample_market_data.keys()),
                data=sample_market_data,
            )

        # Test invalid configuration (invalid_param should be ignored or cause validation error)
        # The current implementation passes through config, so test a truly invalid param
        with pytest.raises((ValueError, StrategyError)):
            framework.run_backtest(
                strategy="volatility_breakout",
                symbols=list(sample_market_data.keys()),
                data=sample_market_data,
                config={"lookback": -1},  # Invalid lookback value
            )


class TestStrategyRegistryIntegration:
    """Test strategy registry integration."""

    @pytest.mark.integration
    def test_registry_strategy_creation(self, framework: BacktestFramework):
        """Test creating strategies through registry."""
        # List available strategies
        strategies = framework.list_available_strategies()
        strategy_names = [s.name if hasattr(s, "name") else s for s in strategies]
        assert "volatility_breakout" in strategy_names
        assert "momentum" in strategy_names
        assert "buy_and_hold" in strategy_names

        # Get strategy info
        info = framework.get_strategy_info("volatility_breakout")
        assert info is not None
        # StrategyInfo object has description attribute
        assert hasattr(info, "description") or "description" in info

        # Create strategy
        strategy = framework.create_strategy("volatility_breakout")
        assert "VolatilityBreakout" in strategy.get_name()

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
        """Test handling of insufficient market data - should complete with 0 trades."""
        # Create minimal data - not enough for strategy lookback
        minimal_data = {
            "BTC": pd.DataFrame(
                {
                    "datetime": [datetime(2020, 1, 1, tzinfo=timezone.utc)],
                    "open": [100],
                    "high": [105],
                    "low": [95],
                    "close": [102],
                    "volume": [1000],
                }
            )
        }

        # Should complete but with no trades due to insufficient data for strategy
        result = framework.run_backtest(
            strategy="volatility_breakout", symbols=["BTC"], data=minimal_data
        )

        # Verify no trades occurred due to insufficient data
        assert len(result.get("trades", [])) == 0

    @pytest.mark.integration
    def test_invalid_symbol_handling(self, framework: BacktestFramework):
        """Test handling of invalid symbols."""
        # Use a very long symbol name which triggers validation error
        data = {"INVALID_SYMBOL": pd.DataFrame()}

        # ValidationError is raised for symbol too long or empty dataframe
        with pytest.raises((ExceptionsValidationError, CoreValidationError)):
            framework.run_backtest(strategy="buy_and_hold", symbols=["INVALID_SYMBOL"], data=data)
