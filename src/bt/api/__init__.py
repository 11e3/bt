"""
REST API layer for remote backtest management and execution.

Provides HTTP endpoints for:
- Running backtests remotely
- Managing backtest configurations
- Retrieving results and reports
- Health monitoring and metrics
"""

from bt.api.jobs import BacktestJob
from bt.api.models import (
    BacktestRequest,
    BacktestResponse,
    HealthResponse,
    MetricsResponse,
)
from bt.api.server import BacktestAPI, create_api_app, run_api_server

__all__ = [
    # Models
    "BacktestRequest",
    "BacktestResponse",
    "HealthResponse",
    "MetricsResponse",
    # Jobs
    "BacktestJob",
    # Server
    "BacktestAPI",
    "create_api_app",
    "run_api_server",
]

if __name__ == "__main__":
    run_api_server()
