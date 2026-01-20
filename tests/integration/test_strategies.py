"""Integration tests for strategy implementations."""

import pytest

from bt.framework import BacktestFramework
from bt.strategies.implementations import (
    BuyAndHoldStrategy,
    MomentumStrategy,
    StrategyFactory,
    VBOPortfolioStrategy,
    VolatilityBreakoutStrategy,
)


class TestVolatilityBreakoutStrategy:
    """Test VolatilityBreakoutStrategy end-to-end."""

    @pytest.mark.integration
    def test_vbo_strategy_backtest(self, framework: BacktestFramework, sample_market_data):
        """Test VBO strategy produces valid backtest results."""
        result = framework.run_backtest(
            strategy="volatility_breakout",
            symbols=list(sample_market_data.keys()),
            data=sample_market_data,
            config={
                "lookback": 5,
                "multiplier": 2,
                "k_factor": 0.5,
                "top_n": 2,
                "mom_lookback": 20,
            },
        )

        assert "VolatilityBreakout" in result["strategy"]
        assert "performance" in result
        assert "equity_curve" in result
        assert len(result["equity_curve"]["dates"]) > 0

    @pytest.mark.integration
    def test_vbo_strategy_validation(self):
        """Test VBO strategy validates configuration correctly."""
        # Valid config
        strategy = VolatilityBreakoutStrategy(
            lookback=5, multiplier=2, k_factor=0.5, top_n=3, mom_lookback=20
        )
        assert len(strategy.validate()) == 0

        # Invalid lookback
        strategy = VolatilityBreakoutStrategy(lookback=0)
        errors = strategy.validate()
        assert len(errors) > 0
        assert any("lookback" in e for e in errors)

        # Invalid k_factor
        strategy = VolatilityBreakoutStrategy(k_factor=5.0)
        errors = strategy.validate()
        assert len(errors) > 0
        assert any("k_factor" in e for e in errors)

    @pytest.mark.integration
    def test_vbo_strategy_conditions(self, framework: BacktestFramework, sample_market_data):
        """Test VBO strategy buy/sell conditions are correctly configured."""
        strategy = VolatilityBreakoutStrategy(
            lookback=5, multiplier=2, k_factor=0.5, top_n=3, mom_lookback=20
        )

        buy_conditions = strategy.get_buy_conditions()
        assert "no_position" in buy_conditions
        assert "breakout" in buy_conditions
        assert "trend_short" in buy_conditions
        assert "trend_long" in buy_conditions

        sell_conditions = strategy.get_sell_conditions()
        assert "trend_exit" in sell_conditions


class TestVBOPortfolioStrategy:
    """Test VBOPortfolioStrategy end-to-end."""

    @pytest.mark.integration
    def test_vbo_portfolio_backtest(self, framework: BacktestFramework, sample_market_data):
        """Test VBO Portfolio strategy with multiple symbols."""
        # Ensure BTC is in the data for market filter
        if "BTC" not in sample_market_data:
            pytest.skip("BTC required for VBO Portfolio strategy")

        result = framework.run_backtest(
            strategy="vbo_portfolio",
            symbols=list(sample_market_data.keys()),
            data=sample_market_data,
            config={
                "ma_short": 5,
                "btc_ma": 20,
                "noise_ratio": 0.5,
                "btc_symbol": "BTC",
            },
        )

        assert "VBOPortfolio" in result["strategy"]
        assert "performance" in result
        assert "equity_curve" in result

    @pytest.mark.integration
    def test_vbo_portfolio_validation(self):
        """Test VBO Portfolio validates configuration."""
        # Valid config
        strategy = VBOPortfolioStrategy(ma_short=5, btc_ma=20, noise_ratio=0.5, btc_symbol="BTC")
        assert len(strategy.validate()) == 0

        # Invalid ma_short
        strategy = VBOPortfolioStrategy(ma_short=100)
        errors = strategy.validate()
        assert len(errors) > 0
        assert any("ma_short" in e for e in errors)

        # Invalid noise_ratio
        strategy = VBOPortfolioStrategy(noise_ratio=3.0)
        errors = strategy.validate()
        assert len(errors) > 0
        assert any("noise_ratio" in e for e in errors)


class TestMomentumStrategy:
    """Test MomentumStrategy end-to-end."""

    @pytest.mark.integration
    def test_momentum_strategy_backtest(self, framework: BacktestFramework, sample_market_data):
        """Test Momentum strategy produces valid results."""
        result = framework.run_backtest(
            strategy="momentum",
            symbols=list(sample_market_data.keys()),
            data=sample_market_data,
            config={"lookback": 20},
        )

        assert "Momentum" in result["strategy"]
        assert "performance" in result
        assert "equity_curve" in result

    @pytest.mark.integration
    def test_momentum_strategy_validation(self):
        """Test Momentum strategy validates configuration."""
        # Valid config
        strategy = MomentumStrategy(lookback=20)
        assert len(strategy.validate()) == 0

        # Invalid lookback (too small)
        strategy = MomentumStrategy(lookback=3)
        errors = strategy.validate()
        assert len(errors) > 0
        assert any("lookback" in e for e in errors)

        # Invalid lookback (too large)
        strategy = MomentumStrategy(lookback=300)
        errors = strategy.validate()
        assert len(errors) > 0


class TestBuyAndHoldStrategy:
    """Test BuyAndHoldStrategy end-to-end."""

    @pytest.mark.integration
    def test_buy_and_hold_backtest(self, framework: BacktestFramework, sample_market_data):
        """Test Buy and Hold strategy produces valid results."""
        result = framework.run_backtest(
            strategy="buy_and_hold",
            symbols=list(sample_market_data.keys()),
            data=sample_market_data,
        )

        assert result["strategy"] == "BuyAndHold"
        assert "performance" in result
        assert "equity_curve" in result

    @pytest.mark.integration
    def test_buy_and_hold_no_sell_conditions(self):
        """Test Buy and Hold has no sell conditions."""
        strategy = BuyAndHoldStrategy()

        buy_conditions = strategy.get_buy_conditions()
        sell_conditions = strategy.get_sell_conditions()

        assert "no_position" in buy_conditions
        assert len(sell_conditions) == 0  # No sell conditions


class TestStrategyFactory:
    """Test StrategyFactory integration."""

    @pytest.mark.integration
    def test_create_all_strategies(self):
        """Test factory can create all registered strategies."""
        available = StrategyFactory.list_strategies()

        for strategy_type in available:
            if strategy_type == "vbo_regime":
                # Skip regime strategy as it requires model_path
                continue

            strategy = StrategyFactory.create_strategy(strategy_type)
            assert strategy is not None
            assert hasattr(strategy, "get_buy_conditions")
            assert hasattr(strategy, "get_sell_conditions")

    @pytest.mark.integration
    def test_factory_invalid_strategy(self):
        """Test factory raises error for unknown strategy."""
        with pytest.raises(ValueError) as exc_info:
            StrategyFactory.create_strategy("nonexistent_strategy")

        assert "Unknown strategy type" in str(exc_info.value)

    @pytest.mark.integration
    def test_factory_strategy_info(self):
        """Test factory provides strategy information."""
        info = StrategyFactory.get_strategy_info("volatility_breakout")

        assert info is not None
        assert "name" in info
        assert "description" in info
        assert "parameters" in info
        assert "default_config" in info

    @pytest.mark.integration
    def test_factory_with_custom_config(self):
        """Test factory creates strategy with custom config."""
        strategy = StrategyFactory.create_strategy(
            "volatility_breakout",
            lookback=10,
            multiplier=3,
            k_factor=0.7,
            top_n=5,
            mom_lookback=30,
        )

        assert strategy.config["lookback"] == 10
        assert strategy.config["multiplier"] == 3
        assert strategy.config["k_factor"] == 0.7


class TestStrategyWithEdgeCases:
    """Test strategies with edge cases and boundary conditions."""

    @pytest.mark.integration
    def test_strategy_with_minimal_data(self, framework: BacktestFramework, sample_market_data):
        """Test strategy handles minimal data gracefully."""
        # Use sample market data with only one symbol
        single_symbol_data = {"BTC": sample_market_data["BTC"]}

        result = framework.run_backtest(
            strategy="buy_and_hold",
            symbols=["BTC"],
            data=single_symbol_data,
        )

        # Should complete without error
        assert "performance" in result

    @pytest.mark.integration
    def test_strategy_with_high_volatility_data(
        self, framework: BacktestFramework, high_volatility_data
    ):
        """Test strategy handles high volatility data."""
        data = {"VOLATILE": high_volatility_data}

        result = framework.run_backtest(
            strategy="volatility_breakout",
            symbols=["VOLATILE"],
            data=data,
            config={
                "lookback": 5,
                "multiplier": 2,
                "k_factor": 0.5,
                "top_n": 1,
                "mom_lookback": 10,
            },
        )

        assert "performance" in result
        # High volatility should potentially generate more trades
        assert isinstance(result["trades"], list)
