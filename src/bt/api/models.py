"""Pydantic models for API requests/responses."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from datetime import datetime


class BacktestRequest(BaseModel):
    """Request model for backtest execution."""

    strategy: str = Field(..., description="Strategy name to use")
    symbols: list[str] = Field(..., description="Trading symbols")
    config: dict[str, Any] | None = Field(default=None, description="Backtest configuration")
    data_source: str | None = Field(default=None, description="Data source identifier")

    class Config:
        schema_extra = {
            "example": {
                "strategy": "volatility_breakout",
                "symbols": ["BTC", "ETH"],
                "config": {"initial_cash": 100000, "fee_rate": 0.0005},
            }
        }


class BacktestResponse(BaseModel):
    """Response model for backtest results."""

    backtest_id: str
    status: str  # pending, running, completed, failed
    created_at: datetime
    completed_at: datetime | None = None
    results: dict[str, Any] | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: datetime
    version: str
    uptime_seconds: float
    active_backtests: int


class MetricsResponse(BaseModel):
    """System metrics response."""

    cpu_percent: float
    memory_mb: float
    disk_usage_mb: float
    active_backtests: int
    cache_stats: dict[str, Any]
