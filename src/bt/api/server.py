"""REST API server for backtest management."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

from bt.api.jobs import BacktestJob
from bt.api.models import (
    BacktestRequest,
    BacktestResponse,
    HealthResponse,
    MetricsResponse,
)
from bt.data.storage import get_data_manager
from bt.framework import BacktestFramework
from bt.monitoring import get_monitor
from bt.utils.logging import get_logger

logger = get_logger(__name__)


class BacktestAPI:
    """REST API for backtest management."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        self.host = host
        self.port = port
        self.app = FastAPI(
            title="BT Framework API",
            description="REST API for cryptocurrency backtesting",
            version="1.0.0",
        )

        # Initialize components
        self.framework = BacktestFramework()
        self.data_manager = get_data_manager()
        self.monitor = get_monitor()

        # Job management
        self.jobs: dict[str, BacktestJob] = {}
        self.start_time = datetime.now(timezone.utc)

        # Setup routes
        self._setup_routes()

        logger.info(f"BacktestAPI initialized on {host}:{port}")

    def _setup_routes(self):
        """Set up API routes."""

        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            return HealthResponse(
                status="healthy",
                timestamp=datetime.now(timezone.utc),
                version="1.0.0",
                uptime_seconds=(datetime.now(timezone.utc) - self.start_time).total_seconds(),
                active_backtests=len([j for j in self.jobs.values() if j.status == "running"]),
            )

        @self.app.get("/metrics", response_model=MetricsResponse)
        async def get_metrics():
            """Get system metrics."""
            import shutil

            import psutil

            process = psutil.Process()
            disk = shutil.disk_usage("/")

            return MetricsResponse(
                cpu_percent=process.cpu_percent(),
                memory_mb=process.memory_info().rss / (1024 * 1024),
                disk_usage_mb=disk.used / (1024 * 1024),
                active_backtests=len([j for j in self.jobs.values() if j.status == "running"]),
                cache_stats=self.data_manager.get_cache_stats()
                if hasattr(self.data_manager, "get_cache_stats")
                else {},
            )

        @self.app.post("/backtests", response_model=BacktestResponse)
        async def create_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
            """Create and start a new backtest."""
            job = BacktestJob(request)
            self.jobs[job.id] = job

            # Start background task
            background_tasks.add_task(self._run_backtest, job)

            logger.info(f"Created backtest job {job.id}")
            return job.to_response()

        @self.app.get("/backtests", response_model=list[BacktestResponse])
        async def list_backtests(
            status: str | None = Query(None, description="Filter by status"),
            limit: int = Query(50, description="Maximum number of results"),
        ):
            """List all backtests."""
            jobs = list(self.jobs.values())

            if status:
                jobs = [j for j in jobs if j.status == status]

            # Sort by creation time (newest first)
            jobs.sort(key=lambda x: x.created_at, reverse=True)

            return [job.to_response() for job in jobs[:limit]]

        @self.app.get("/backtests/{backtest_id}", response_model=BacktestResponse)
        async def get_backtest(backtest_id: str):
            """Get backtest details."""
            if backtest_id not in self.jobs:
                raise HTTPException(status_code=404, detail="Backtest not found")

            return self.jobs[backtest_id].to_response()

        @self.app.delete("/backtests/{backtest_id}")
        async def cancel_backtest(backtest_id: str):
            """Cancel a running backtest."""
            if backtest_id not in self.jobs:
                raise HTTPException(status_code=404, detail="Backtest not found")

            job = self.jobs[backtest_id]
            if job.status == "running" and job.task:
                job.task.cancel()
                job.status = "cancelled"
                job.completed_at = datetime.now(timezone.utc)

            return {"message": f"Backtest {backtest_id} cancelled"}

        @self.app.get("/strategies")
        async def list_strategies():
            """List available strategies."""
            try:
                strategies = self.framework.list_available_strategies()
                return {"strategies": strategies}
            except Exception as e:
                logger.error(f"Error listing strategies: {e}")
                raise HTTPException(status_code=500, detail=str(e)) from e

        @self.app.get("/backtests/{backtest_id}/results/download")
        async def download_results(backtest_id: str):
            """Download backtest results as file."""
            if backtest_id not in self.jobs:
                raise HTTPException(status_code=404, detail="Backtest not found")

            job = self.jobs[backtest_id]
            if job.status != "completed" or not job.results:
                raise HTTPException(
                    status_code=400, detail="Backtest not completed or no results available"
                )

            # Save results to temporary file
            temp_file = Path(f"/tmp/{backtest_id}_results.json")
            with temp_file.open("w") as f:
                json.dump(job.results, f, indent=2, default=str)

            return FileResponse(
                temp_file, media_type="application/json", filename=f"{backtest_id}_results.json"
            )

        # Error handlers
        @self.app.exception_handler(Exception)
        async def global_exception_handler(_request, exc):
            logger.error(f"Unhandled exception: {exc}", exc_info=True)
            return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    async def _run_backtest(self, job: BacktestJob):
        """Run backtest in background."""
        try:
            job.status = "running"
            logger.info(f"Starting backtest {job.id}")

            # Prepare data (simplified - in production, load from data source)
            if job.request.data_source:
                # Load data from configured source
                data = self.data_manager.retrieve(job.request.data_source)
                if data is None:
                    raise ValueError(f"Data source {job.request.data_source} not found")
            else:
                # Use sample data for demo
                data = self._generate_sample_data(job.request.symbols)

            # Run backtest
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                self.framework.run_backtest,
                job.request.strategy,
                job.request.symbols,
                data,
                job.request.config,
            )

            job.status = "completed"
            job.results = results
            job.completed_at = datetime.now(timezone.utc)

            logger.info(f"Completed backtest {job.id}")

        except asyncio.CancelledError:
            job.status = "cancelled"
            job.completed_at = datetime.now(timezone.utc)
            logger.info(f"Cancelled backtest {job.id}")

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            job.completed_at = datetime.now(timezone.utc)
            logger.error(f"Failed backtest {job.id}: {e}")

    def _generate_sample_data(self, symbols: list[str]) -> dict[str, Any]:
        """Generate sample data for demo purposes."""
        import numpy as np
        import pandas as pd

        data = {}
        dates = pd.date_range("2020-01-01", periods=365, freq="D")

        for symbol in symbols:
            np.random.seed(hash(symbol) % 2**32)
            returns = np.random.normal(0.001, 0.03, len(dates))
            prices = 50000 * np.exp(np.cumsum(returns))

            highs = prices * (1 + np.random.uniform(0, 0.02, len(dates)))
            lows = prices * (1 - np.random.uniform(0, 0.02, len(dates)))
            opens = prices * (1 + np.random.normal(0, 0.005, len(dates)))
            closes = prices
            volumes = np.random.lognormal(15, 1, len(dates)).astype(int)

            data[symbol] = pd.DataFrame(
                {
                    "datetime": dates,
                    "open": opens,
                    "high": highs,
                    "low": lows,
                    "close": closes,
                    "volume": volumes,
                }
            )

        return data

    def run(self):
        """Run the API server."""
        logger.info(f"Starting BT Framework API on {self.host}:{self.port}")
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")


def create_api_app() -> FastAPI:
    """Create FastAPI app instance."""
    api = BacktestAPI()
    return api.app


def run_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    api = BacktestAPI(host=host, port=port)
    api.run()
