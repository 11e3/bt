"""Performance monitoring decorators."""

from __future__ import annotations

import time
from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


def monitor_performance(operation_name: str | None = None, log_result: bool = True):
    """Decorator to monitor function performance."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Import at runtime to avoid circular imports
            from bt.monitoring import get_metrics_collector, get_structured_logger

            start_time = time.time()
            success = True
            exception = None

            try:
                return func(*args, **kwargs)
            except Exception as e:
                success = False
                exception = e
                raise
            finally:
                duration = time.time() - start_time

                # Record metrics
                metrics = get_metrics_collector()
                metrics.record_metric(
                    f"function.duration.{func.__name__}",
                    duration,
                    {"success": str(success)},
                    unit="seconds",
                )

                # Log performance if requested
                if log_result:
                    logger = get_structured_logger()
                    logger.log_performance(
                        operation=operation_name or func.__name__,
                        duration=duration,
                        success=success,
                        function=func.__name__,
                        args_count=len(args),
                        kwargs_count=len(kwargs),
                        exception=str(exception) if exception else None,
                    )

        return wrapper

    return decorator


def monitor_backtest(backtest_id_param: str = "backtest_id"):
    """Decorator to monitor backtest operations."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Import at runtime to avoid circular imports
            from bt.monitoring import get_performance_monitor, get_structured_logger
            from bt.utils.logging import get_logger

            get_logger(__name__)
            start_time = time.time()

            # Extract backtest ID from parameters
            backtest_id = "unknown"
            if backtest_id_param in kwargs:
                backtest_id = kwargs[backtest_id_param]
            elif len(args) > 0 and hasattr(args[0], backtest_id_param):
                backtest_id = getattr(args[0], backtest_id_param)

            monitor = get_performance_monitor()

            try:
                result = func(*args, **kwargs)

                # Record backtest metrics if result contains relevant data
                if isinstance(result, dict) and "performance" in result:
                    perf = result["performance"]
                    duration = time.time() - start_time

                    symbols_count = len(result.get("symbols", []))
                    bars_count = sum(len(data) for data in result.get("market_data", {}).values())
                    trades_count = len(result.get("trades", []))
                    total_return = perf.get("total_return", 0)

                    monitor.record_backtest_metrics(
                        backtest_id=backtest_id,
                        duration=duration,
                        symbols_count=symbols_count,
                        bars_count=bars_count,
                        trades_count=trades_count,
                        total_return=total_return,
                    )

                return result

            except Exception as e:
                duration = time.time() - start_time
                structured_logger = get_structured_logger()
                structured_logger.log_performance(
                    operation=f"backtest.{func.__name__}",
                    duration=duration,
                    success=False,
                    backtest_id=backtest_id,
                    exception=str(e),
                )
                raise

        return wrapper

    return decorator
