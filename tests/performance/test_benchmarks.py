"""Performance benchmarks for critical backtest operations."""

from datetime import datetime, timezone

import numpy as np
import pytest

from bt.framework import BacktestFramework
from bt.reporting.metrics import calculate_performance_metrics
from bt.utils.indicator_cache import get_indicator_cache


class TestBacktestPerformance:
    """Performance benchmarks for backtest operations."""

    @pytest.mark.performance
    def test_vbo_backtest_performance(self, benchmark, benchmark_data):
        """Benchmark VBO strategy backtest performance."""
        framework = BacktestFramework()

        def run_backtest():
            return framework.run_backtest(
                strategy="volatility_breakout",
                symbols=list(benchmark_data.keys())[:5],  # First 5 symbols
                data={k: benchmark_data[k] for k in list(benchmark_data.keys())[:5]},
                config={
                    "initial_cash": 1000000,
                    "fee_rate": 0.0005,
                    "slippage_rate": 0.0005,
                },
            )

        result = benchmark(run_backtest)

        # Validate result structure (benchmark will still run even if assertions fail)
        assert result["performance"]["num_trades"] >= 0
        assert len(result["equity_curve"]["dates"]) > 100

    @pytest.mark.performance
    def test_large_portfolio_performance(self, benchmark, large_portfolio_data):
        """Benchmark performance with large number of symbols."""
        framework = BacktestFramework()

        def run_large_portfolio():
            return framework.run_backtest(
                strategy="momentum",
                symbols=list(large_portfolio_data.keys())[:20],  # 20 symbols
                data={k: large_portfolio_data[k] for k in list(large_portfolio_data.keys())[:20]},
            )

        result = benchmark(run_large_portfolio)

        # Should handle large portfolios without excessive slowdown
        assert result["performance"]["num_trades"] >= 0

    @pytest.mark.performance
    def test_metrics_calculation_performance(self, benchmark, all_performance_scenarios):
        """Benchmark performance metrics calculation."""
        crash_scenario = all_performance_scenarios["crash"]
        equity_values = crash_scenario["equity_curve"]["values"]
        dates = crash_scenario["equity_curve"]["dates"]

        def calculate_metrics():
            return calculate_performance_metrics(
                equity_curve=equity_values,
                dates=[datetime.fromisoformat(d) for d in dates],
                trades=[],
                initial_cash=1000000,
            )

        result = benchmark(calculate_metrics)

        # Validate metrics structure
        assert "total_return" in result
        assert "sharpe" in result
        assert "mdd" in result


class TestIndicatorPerformance:
    """Performance benchmarks for technical indicators."""

    @pytest.mark.performance
    def test_sma_calculation_performance(self, benchmark, benchmark_data):
        """Benchmark SMA calculation performance."""
        cache = get_indicator_cache()
        btc_data = benchmark_data["BTC"]
        close_prices = btc_data["close"].values

        def calculate_sma_batch():
            results = []
            for i in range(20, len(close_prices)):
                # Calculate SMA for different lookbacks
                for lookback in [5, 10, 20, 50]:
                    if i >= lookback:
                        sma = cache.calculate_indicator(
                            "BTC", "sma", lookback, close_prices[: i + 1]
                        )
                        results.append(sma)
            return results

        results = benchmark(calculate_sma_batch)
        assert len(results) > 0

    @pytest.mark.performance
    def test_indicator_caching_performance(self, benchmark, benchmark_data):
        """Benchmark indicator caching effectiveness."""
        cache = get_indicator_cache()
        btc_data = benchmark_data["BTC"]
        close_prices = btc_data["close"].values

        def calculate_with_caching():
            results = []
            # Same calculations multiple times to test cache hits
            for _ in range(10):
                for i in range(50, len(close_prices), 10):
                    sma = cache.calculate_indicator("BTC", "sma", 20, close_prices[: i + 1])
                    results.append(sma)
            return results

        results = benchmark(calculate_with_caching)
        assert len(results) > 0

        # Check cache performance
        stats = cache.get_cache_stats()
        assert stats["cache_hits"] > 0  # Should have cache hits


class TestDataProcessingPerformance:
    """Performance benchmarks for data processing operations."""

    @pytest.mark.performance
    def test_market_data_generation_performance(self, benchmark, market_data_generator):
        """Benchmark synthetic market data generation."""

        def generate_data():
            return market_data_generator.generate_multi_symbol_data(
                symbols=[f"SYMBOL_{i:03d}" for i in range(10)],
                start_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
                periods=500,
                volatility=0.02,
            )

        data = benchmark(generate_data)
        assert len(data) == 10
        assert len(data["SYMBOL_001"]) == 500

    @pytest.mark.performance
    def test_correlated_data_generation_performance(self, benchmark, market_data_generator):
        """Benchmark correlated data generation."""
        correlation_matrix = np.array(
            [
                [1.0, 0.8, 0.6, 0.4, 0.2],
                [0.8, 1.0, 0.7, 0.5, 0.3],
                [0.6, 0.7, 1.0, 0.6, 0.4],
                [0.4, 0.5, 0.6, 1.0, 0.5],
                [0.2, 0.3, 0.4, 0.5, 1.0],
            ]
        )

        def generate_correlated_data():
            return market_data_generator.generate_multi_symbol_data(
                symbols=["A", "B", "C", "D", "E"],
                start_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
                periods=252,
                correlation_matrix=correlation_matrix,
            )

        data = benchmark(generate_correlated_data)
        assert len(data) == 5


class TestMemoryPerformance:
    """Memory usage benchmarks."""

    @pytest.mark.performance
    def test_memory_usage_large_backtest(self, benchmark, benchmark_data):
        """Benchmark memory usage for large backtests."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        def run_memory_intensive_backtest():
            framework = BacktestFramework()
            return framework.run_backtest(
                strategy="volatility_breakout",
                symbols=list(benchmark_data.keys()),
                data=benchmark_data,
                config={
                    "initial_cash": 10000000,  # Large portfolio
                    "fee_rate": 0.0005,
                    "slippage_rate": 0.0005,
                },
            )

        result = benchmark(run_memory_intensive_backtest)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory

        # Log memory usage for analysis
        print(f"Memory used: {memory_used:.2f} MB")

        # Should complete without excessive memory usage
        assert memory_used < 500  # Less than 500MB additional usage
        assert result["performance"]["num_trades"] >= 0


class TestConcurrentPerformance:
    """Performance benchmarks for concurrent operations."""

    @pytest.mark.performance
    def test_multiple_backtests_concurrent(self, benchmark, sample_market_data):
        """Benchmark running multiple backtests concurrently."""
        import concurrent.futures

        def run_single_backtest(strategy_name):
            framework = BacktestFramework()
            return framework.run_backtest(
                strategy=strategy_name,
                symbols=list(sample_market_data.keys()),
                data=sample_market_data,
            )

        strategies = ["volatility_breakout", "momentum", "buy_and_hold"]

        def run_concurrent_backtests():
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(run_single_backtest, strategy) for strategy in strategies
                ]
                return [future.result() for future in concurrent.futures.as_completed(futures)]

        results = benchmark(run_concurrent_backtests)
        assert len(results) == 3
        for result in results:
            assert result["performance"]["num_trades"] >= 0


# Performance regression detection

PERFORMANCE_THRESHOLDS = {
    "vbo_backtest_time": 5.0,  # seconds
    "large_portfolio_time": 10.0,  # seconds
    "metrics_calculation_time": 0.1,  # seconds
    "sma_calculation_time": 1.0,  # seconds
    "data_generation_time": 2.0,  # seconds
}


@pytest.fixture(scope="session", autouse=True)
def performance_regression_check(request):
    """Check for performance regressions after test runs."""
    yield

    # This would run after all performance tests
    # In a real implementation, you'd collect benchmark results
    # and compare against historical baselines

    if hasattr(request.config, "_benchmark_data"):
        # Analyze benchmark results for regressions
        benchmark_data = request.config._benchmark_data

        for benchmark_name, stats in benchmark_data.items():
            if benchmark_name in PERFORMANCE_THRESHOLDS:
                threshold = PERFORMANCE_THRESHOLDS[benchmark_name]
                mean_time = stats.get("mean", 0)

                if mean_time > threshold:
                    pytest.fail(
                        f"Performance regression detected in {benchmark_name}: "
                        f"{mean_time:.3f}s > {threshold:.3f}s threshold"
                    )
