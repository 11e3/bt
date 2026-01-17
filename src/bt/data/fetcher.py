"""Data fetching module with error handling and retry logic.

Fetches cryptocurrency data from Upbit exchange with proper
timeout handling, rate limiting, and error recovery.
"""

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyupbit

from bt.config.config import settings
from bt.data.storage import get_data_manager
from bt.utils.logging import get_logger

logger = get_logger(__name__)


class DataFetchError(Exception):
    """Base exception for data fetching errors."""


class RateLimitError(DataFetchError):
    """Raised when API rate limit is exceeded."""


class DataFetcher:
    """Fetches and stores cryptocurrency data from Upbit exchange.

    Features:
    - Automatic retry with exponential backoff
    - Rate limiting to respect API constraints
    - Incremental updates to minimize API calls
    - Proper timeout handling
    """

    def __init__(
        self,
        base_dir: Path | None = None,
        timeout: int | None = None,
        rate_limit: float | None = None,
    ) -> None:
        """Initialize DataFetcher.

        Args:
            base_dir: Base directory for storing data files
            timeout: Request timeout in seconds
            rate_limit: Minimum seconds between requests
        """
        self.base_dir = base_dir or settings.data_dir
        self.timeout = timeout or settings.api_timeout
        self.rate_limit = rate_limit or settings.api_rate_limit

        # Initialize data manager for caching
        self.data_manager = get_data_manager()

        logger.info(
            "DataFetcher initialized",
            extra={
                "base_dir": str(self.base_dir),
                "timeout": self.timeout,
                "rate_limit": self.rate_limit,
            },
        )

    def get_data_path(self, interval: str, symbol: str) -> Path:
        """Get the file path for storing data.

        Args:
            interval: Time interval
            symbol: Trading symbol

        Returns:
            Path to data file
        """
        interval_dir = self.base_dir / interval
        interval_dir.mkdir(parents=True, exist_ok=True)
        return interval_dir / f"{symbol}.parquet"

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        count: int = 10000,
        max_retries: int = 3,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Upbit with retry logic and caching."""
        ticker = f"KRW-{symbol.upper()}"

        # Create cache key based on parameters
        cache_key = f"ohlcv_{symbol}_{interval}_{start_date}_{end_date}_{count}"

        # Try to get from cache first
        if use_cache:
            cached_data = self.data_manager.retrieve(cache_key)
            if cached_data is not None:
                logger.info(f"Using cached data for {ticker}")
                return cached_data

        all_data = []
        current_end = end_date or datetime.now(tz=timezone.utc)

        # [추가] 무한 루프 방지용 변수
        prev_oldest_date = None

        logger.info(
            f"Fetching {ticker} data",
            extra={"interval": interval, "start_date": start_date, "end_date": end_date},
        )

        retry_count = 0

        while True:
            try:
                # Rate limiting
                time.sleep(self.rate_limit)

                # Fetch data with timeout
                df = pyupbit.get_ohlcv(
                    ticker=ticker,
                    interval=interval,
                    to=current_end.strftime("%Y%m%d%H%M%S") if current_end else None,
                    count=count,
                )

                if df is None or len(df) == 0:
                    logger.debug("No more data available")
                    break

                # Reset index to make datetime a column
                df = df.reset_index()
                df.rename(columns={"index": "datetime"}, inplace=True)

                all_data.append(df)

                # 현재 배치에서 가장 오래된 날짜 확인
                oldest_date = df["datetime"].min()

                # Ensure oldest_date is timezone-aware in UTC for consistent comparisons
                if pd.isna(oldest_date):
                    logger.debug("No valid dates in batch")
                    break

                if oldest_date.tzinfo is None:
                    oldest_date = oldest_date.replace(tzinfo=timezone.utc)

                # [수정 1] 진행 상황을 INFO로 변경하여 사용자에게 알림 (매 1000개 캔들마다 or 매번)
                logger.info(
                    f"Fetched {len(df)} rows ({ticker})",
                    extra={"oldest_date": oldest_date.isoformat()},
                )

                # [수정 2] 무한 루프 방지: 날짜가 더 이상 과거로 가지 않으면 중단
                if prev_oldest_date is not None and oldest_date >= prev_oldest_date:
                    logger.warning(
                        "Loop detected: oldest_date is not updating. Stopping fetch.",
                        extra={"symbol": symbol, "date": oldest_date},
                    )
                    break
                prev_oldest_date = oldest_date

                # Check if we've reached the start date
                if start_date and oldest_date <= start_date:
                    break

                # Move to earlier data
                current_end = oldest_date - timedelta(seconds=1)
                retry_count = 0  # Reset retry count on success

            except Exception as e:
                # (기존 예외 처리 로직 유지)
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error("Max retries exceeded", exc_info=True)
                    raise DataFetchError(f"Failed to fetch {symbol}") from e

                wait_time = 2**retry_count
                logger.warning(f"Retrying in {wait_time}s due to: {e}")
                time.sleep(wait_time)

        if not all_data:
            logger.warning(f"No data fetched for {symbol}")
            return pd.DataFrame()

        # (이하 데이터 병합 및 리턴 로직 기존과 동일)
        result_df = pd.concat(all_data, ignore_index=True)
        result_df = result_df.drop_duplicates(subset=["datetime"])
        result_df = result_df.sort_values("datetime").reset_index(drop=True)

        # Ensure datetime is timezone-aware in UTC for consistent comparisons
        if result_df["datetime"].dt.tz is None:
            result_df["datetime"] = result_df["datetime"].dt.tz_localize("UTC")
        else:
            result_df["datetime"] = result_df["datetime"].dt.tz_convert("UTC")

        if start_date:
            result_df = result_df[result_df["datetime"] >= start_date]
        if end_date:
            result_df = result_df[result_df["datetime"] <= end_date]

        result_df = pd.DataFrame(result_df)

        # Cache the result
        if use_cache and not result_df.empty:
            self.data_manager.store(cache_key, result_df, ttl=3600)  # Cache for 1 hour

        return result_df

    def load_existing_data(self, interval: str, symbol: str) -> pd.DataFrame | None:
        """Load existing data from parquet file.

        Args:
            interval: Time interval
            symbol: Trading symbol

        Returns:
            DataFrame with existing data, or None if file doesn't exist
        """
        file_path = self.get_data_path(interval, symbol)

        if not file_path.exists():
            return None

        try:
            df = pd.read_parquet(file_path)
            # Ensure datetime column is datetime type
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                # Ensure datetime is timezone-aware in UTC
                if df["datetime"].dt.tz is None:
                    df["datetime"] = df["datetime"].dt.tz_localize("UTC")
                else:
                    df["datetime"] = df["datetime"].dt.tz_convert("UTC")

            logger.debug(
                "Existing data loaded",
                extra={"symbol": symbol, "rows": len(df), "path": str(file_path)},
            )
            return df
        except Exception as e:
            logger.error(
                "Error loading existing data",
                extra={"symbol": symbol, "path": str(file_path), "error": str(e)},
                exc_info=True,
            )
            return None

    def save_data(self, df: pd.DataFrame, interval: str, symbol: str) -> None:
        """Save data to parquet file.

        Args:
            df: DataFrame to save
            interval: Time interval
            symbol: Trading symbol

        Raises:
            IOError: If save operation fails
        """
        file_path = self.get_data_path(interval, symbol)

        try:
            if "datetime" in df.columns:
                # 1. 먼저 datetime 객체로 변환 (아직 Naive 상태)
                df["datetime"] = pd.to_datetime(df["datetime"])

                # 2. 타임존 정보가 없는 경우(Naive)에만 KST 부여 후 UTC로 변환
                if df["datetime"].dt.tz is None:
                    df["datetime"] = (
                        df["datetime"]
                        .dt.tz_localize("Asia/Seoul")  # KST라고 명시
                        .dt.tz_convert("UTC")  # 그 값을 UTC로 변환
                    )
                else:
                    # 이미 타임존이 있다면 UTC로 통일
                    df["datetime"] = df["datetime"].dt.tz_convert("UTC")

            # Save to parquet
            df.to_parquet(file_path, index=False, engine="pyarrow")

            logger.info(
                "Data saved",
                extra={"symbol": symbol, "rows": len(df), "path": str(file_path)},
            )
        except Exception as e:
            logger.error(
                "Error saving data",
                extra={"symbol": symbol, "path": str(file_path), "error": str(e)},
                exc_info=True,
            )
            raise OSError(f"Failed to save data for {symbol}") from e

    def fetch_and_update(
        self,
        symbol: str,
        interval: str,
        full_history: bool = False,
    ) -> pd.DataFrame:
        """Fetch data and update existing file incrementally.

        Args:
            symbol: Trading pair symbol
            interval: Time interval
            full_history: If True, fetch all available history

        Returns:
            Updated DataFrame
        """
        # Load existing data
        existing_df = self.load_existing_data(interval, symbol)

        if existing_df is not None and not full_history:
            # Incremental update: fetch only new data
            last_date = existing_df["datetime"].max()
            logger.info(
                "Incremental update",
                extra={"symbol": symbol, "last_date": last_date.isoformat()},
            )

            # Fetch new data
            new_df = self.fetch_ohlcv(
                symbol=symbol, interval=interval, start_date=last_date, end_date=None
            )

            if new_df.empty:
                logger.info("No new data available", extra={"symbol": symbol})
                return existing_df

            # Combine and remove duplicates
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["datetime"])
            combined_df = combined_df.sort_values("datetime").reset_index(drop=True)

            # Save updated data
            self.save_data(combined_df, interval, symbol)

            logger.info(
                "Data updated",
                extra={
                    "symbol": symbol,
                    "new_rows": len(combined_df) - len(existing_df),
                    "total_rows": len(combined_df),
                },
            )

            return combined_df

        # Full history fetch
        logger.info("Fetching full history", extra={"symbol": symbol})

        df = self.fetch_ohlcv(symbol=symbol, interval=interval, start_date=None, end_date=None)

        if df.empty:
            logger.warning(f"No data available for {symbol}")
            return df

        # Save data
        self.save_data(df, interval, symbol)
        return df

    def fetch_multiple_symbols(
        self,
        symbols: list[str],
        intervals: list[str],
        full_history: bool = False,
    ) -> None:
        """Fetch data for multiple symbols and intervals.

        Args:
            symbols: List of symbols to fetch
            intervals: List of intervals to fetch
            full_history: If True, fetch all available history
        """
        total = len(symbols) * len(intervals)
        count = 0
        errors = []

        logger.info(
            "Starting batch fetch",
            extra={"symbols": symbols, "intervals": intervals, "total_tasks": total},
        )

        for symbol in symbols:
            for interval in intervals:
                count += 1
                logger.info(
                    f"Processing {count}/{total}",
                    extra={"symbol": symbol, "interval": interval},
                )

                try:
                    self.fetch_and_update(symbol, interval, full_history)
                except Exception as e:
                    error_msg = f"{symbol}-{interval}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(
                        "Failed to fetch data",
                        extra={"symbol": symbol, "interval": interval, "error": str(e)},
                        exc_info=True,
                    )
                    continue

        if errors:
            logger.warning(
                "Batch fetch completed with errors",
                extra={"total_errors": len(errors), "errors": errors},
            )
        else:
            logger.info("Batch fetch completed successfully", extra={"total": total})
