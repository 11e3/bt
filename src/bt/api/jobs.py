"""Backtest job management."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from bt.api.models import BacktestRequest, BacktestResponse

if TYPE_CHECKING:
    import asyncio


class BacktestJob:
    """Represents a background backtest job."""

    def __init__(self, request: BacktestRequest):
        self.id: str = f"bt_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{hash(str(request)) % 10000}"
        self.request: BacktestRequest = request
        self.status: str = "pending"
        self.created_at: datetime = datetime.now(timezone.utc)
        self.completed_at: datetime | None = None
        self.results: dict[str, Any] | None = None
        self.error: str | None = None
        self.task: asyncio.Task | None = None

    def to_response(self) -> BacktestResponse:
        """Convert to API response."""
        return BacktestResponse(
            backtest_id=self.id,
            status=self.status,
            created_at=self.created_at,
            completed_at=self.completed_at,
            results=self.results,
            error=self.error,
        )
