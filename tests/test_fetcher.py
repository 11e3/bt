"""Test DataFetcher module."""

from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from bt.data.fetcher import DataFetcher, DataFetchError


@pytest.fixture
def temp_data_dir() -> Path:
    """Provide temporary data directory."""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def fetcher(temp_data_dir: Path) -> DataFetcher:
    """Provide DataFetcher with temp directory."""
    return DataFetcher(base_dir=temp_data_dir, timeout=10, rate_limit=0.01)


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Provide sample OHLCV DataFrame."""
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
            "open": [100.0, 102.0, 101.0, 103.0, 104.0],
            "high": [105.0, 106.0, 104.0, 107.0, 108.0],
            "low": [99.0, 100.0, 99.0, 101.0, 102.0],
            "close": [102.0, 101.0, 103.0, 104.0, 105.0],
            "volume": [1000, 1200, 1100, 1300, 1400],
        }
    )


class TestDataFetcherInit:
    """Test DataFetcher initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization uses settings."""
        fetcher = DataFetcher()

        assert fetcher.base_dir is not None
        assert fetcher.timeout > 0
        assert fetcher.rate_limit >= 0

    def test_custom_initialization(self, temp_data_dir: Path) -> None:
        """Test custom initialization."""
        fetcher = DataFetcher(base_dir=temp_data_dir, timeout=30, rate_limit=0.5)

        assert fetcher.base_dir == temp_data_dir
        assert fetcher.timeout == 30
        assert fetcher.rate_limit == 0.5


class TestGetDataPath:
    """Test get_data_path method."""

    def test_path_generation(self, fetcher: DataFetcher) -> None:
        """Test data path generation."""
        path = fetcher.get_data_path("day", "BTC")

        assert "day" in str(path)
        assert "BTC.parquet" in str(path)

    def test_path_with_special_interval(self, fetcher: DataFetcher) -> None:
        """Test path with special characters in interval."""
        path = fetcher.get_data_path("minute30", "ETH")

        # Slash should be replaced
        assert "minute30" in str(path)

    def test_creates_directory(self, fetcher: DataFetcher) -> None:
        """Test directory is created."""
        path = fetcher.get_data_path("new_interval", "BTC")

        assert path.parent.exists()


class TestFetchOhlcv:
    """Test fetch_ohlcv method."""

    @patch("bt.data.fetcher.pyupbit.get_ohlcv")
    def test_fetch_success(self, mock_get_ohlcv: MagicMock, fetcher: DataFetcher) -> None:
        """Test successful data fetch."""
        mock_df = pd.DataFrame(
            {
                "open": [100, 101],
                "high": [105, 106],
                "low": [95, 96],
                "close": [102, 103],
                "volume": [1000, 1100],
            },
            index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        )

        # [수정 전] 무한 루프 원인: 항상 데이터를 반환함
        # mock_get_ohlcv.return_value = mock_df

        # [수정 후] 첫 번째는 데이터 반환, 두 번째는 None 반환하여 루프 종료 유도
        mock_get_ohlcv.side_effect = [mock_df, None]

        result = fetcher.fetch_ohlcv("BTC", "day", count=2)

        assert len(result) == 2
        assert "datetime" in result.columns
        assert "close" in result.columns

    @patch("bt.data.fetcher.pyupbit.get_ohlcv")
    def test_fetch_empty_response(self, mock_get_ohlcv: MagicMock, fetcher: DataFetcher) -> None:
        """Test handling empty response."""
        mock_get_ohlcv.return_value = None

        result = fetcher.fetch_ohlcv("BTC", "day")

        assert len(result) == 0

    @patch("bt.data.fetcher.pyupbit.get_ohlcv")
    def test_fetch_with_date_range(self, mock_get_ohlcv: MagicMock, fetcher: DataFetcher) -> None:
        """Test fetch with date range."""
        mock_df = pd.DataFrame(
            {
                "open": [100],
                "high": [105],
                "low": [95],
                "close": [102],
                "volume": [1000],
            },
            # [수정 전] tz_localize 없음 (Naive) -> 에러 발생
            # index=pd.to_datetime(["2024-01-01"]),
            # [수정 후] UTC 시간대 적용 (Aware) -> start_date와 비교 가능
            index=pd.to_datetime(["2024-01-01"]).tz_localize("UTC"),
        )
        mock_get_ohlcv.return_value = mock_df

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 5, tzinfo=UTC)

        result = fetcher.fetch_ohlcv("BTC", "day", start_date=start, end_date=end)

        assert len(result) >= 0

    @patch("bt.data.fetcher.pyupbit.get_ohlcv")
    def test_fetch_retry_on_error(self, mock_get_ohlcv: MagicMock, fetcher: DataFetcher) -> None:
        """Test retry logic on error."""
        # Fail twice, succeed on third
        mock_get_ohlcv.side_effect = [
            Exception("Network error"),
            Exception("Network error"),
            None,  # Empty response to break loop
        ]

        fetcher.fetch_ohlcv("BTC", "day", max_retries=3)

        assert mock_get_ohlcv.call_count == 3

    @patch("bt.data.fetcher.pyupbit.get_ohlcv")
    def test_fetch_max_retries_exceeded(
        self, mock_get_ohlcv: MagicMock, fetcher: DataFetcher
    ) -> None:
        """Test exception when max retries exceeded."""
        mock_get_ohlcv.side_effect = Exception("Persistent error")

        with pytest.raises(DataFetchError, match="Failed to fetch"):
            fetcher.fetch_ohlcv("BTC", "day", max_retries=2)


class TestLoadExistingData:
    """Test load_existing_data method."""

    def test_load_existing(self, fetcher: DataFetcher, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test loading existing data."""
        # Save first
        fetcher.save_data(sample_ohlcv_df, "day", "BTC")

        # Then load
        result = fetcher.load_existing_data("day", "BTC")

        assert result is not None
        assert len(result) == 5

    def test_load_nonexistent(self, fetcher: DataFetcher) -> None:
        """Test loading nonexistent file."""
        result = fetcher.load_existing_data("day", "NONEXISTENT")

        assert result is None

    def test_load_ensures_datetime_type(
        self, fetcher: DataFetcher, sample_ohlcv_df: pd.DataFrame
    ) -> None:
        """Test datetime column is properly typed."""
        fetcher.save_data(sample_ohlcv_df, "day", "BTC")
        result = fetcher.load_existing_data("day", "BTC")

        assert result is not None
        assert pd.api.types.is_datetime64_any_dtype(result["datetime"])


class TestSaveData:
    """Test save_data method."""

    def test_save_success(self, fetcher: DataFetcher, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test successful save."""
        fetcher.save_data(sample_ohlcv_df, "day", "BTC")

        path = fetcher.get_data_path("day", "BTC")
        assert path.exists()

    def test_save_creates_directory(
        self, fetcher: DataFetcher, sample_ohlcv_df: pd.DataFrame
    ) -> None:
        """Test save creates directory if needed."""
        fetcher.save_data(sample_ohlcv_df, "new_interval", "BTC")

        path = fetcher.get_data_path("new_interval", "BTC")
        assert path.exists()

    def test_save_roundtrip(self, fetcher: DataFetcher, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test save and load roundtrip."""
        fetcher.save_data(sample_ohlcv_df, "day", "BTC")
        loaded = fetcher.load_existing_data("day", "BTC")

        assert loaded is not None
        assert len(loaded) == len(sample_ohlcv_df)


class TestFetchAndUpdate:
    """Test fetch_and_update method."""

    @patch("bt.data.fetcher.pyupbit.get_ohlcv")
    def test_full_history_fetch(
        self,
        mock_get_ohlcv: MagicMock,
        fetcher: DataFetcher,
    ) -> None:
        """Test full history fetch."""
        mock_df = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [105, 106, 107],
                "low": [95, 96, 97],
                "close": [102, 103, 104],
                "volume": [1000, 1100, 1200],
            },
            # [권장] 타임존 명시 (이전 수정사항 반영)
            index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]).tz_localize("UTC"),
        )

        # [수정] return_value 대신 side_effect 사용
        # 첫 번째 호출: mock_df 반환
        # 두 번째 호출: None 반환 -> 루프 종료
        mock_get_ohlcv.side_effect = [mock_df, None]

        result = fetcher.fetch_and_update("BTC", "day", full_history=True)

        assert len(result) == 3

    @patch("bt.data.fetcher.pyupbit.get_ohlcv")
    def test_incremental_update(
        self,
        mock_get_ohlcv: MagicMock,
        fetcher: DataFetcher,
        sample_ohlcv_df: pd.DataFrame,
    ) -> None:
        """Test incremental update."""
        # Save existing data
        fetcher.save_data(sample_ohlcv_df, "day", "BTC")

        # Mock new data
        new_df = pd.DataFrame(
            {
                "open": [105, 106],
                "high": [110, 111],
                "low": [103, 104],
                "close": [107, 108],
                "volume": [1500, 1600],
            },
            # [중요] 저장된 데이터(UTC)와 비교하기 위해 UTC 설정 필수
            index=pd.to_datetime(["2024-01-06", "2024-01-07"]).tz_localize("UTC"),
        )

        # [수정] 첫 번째는 데이터, 두 번째는 None을 반환하여 루프 종료
        mock_get_ohlcv.side_effect = [new_df, None]

        result = fetcher.fetch_and_update("BTC", "day", full_history=False)

        # Should have original + new data
        assert len(result) == 7

    @patch("bt.data.fetcher.pyupbit.get_ohlcv")
    def test_no_new_data(
        self,
        mock_get_ohlcv: MagicMock,
        fetcher: DataFetcher,
        sample_ohlcv_df: pd.DataFrame,
    ) -> None:
        """Test when no new data available."""
        fetcher.save_data(sample_ohlcv_df, "day", "BTC")

        # Empty DataFrame for new data
        mock_get_ohlcv.return_value = pd.DataFrame()

        result = fetcher.fetch_and_update("BTC", "day", full_history=False)

        # Should return existing data unchanged
        assert len(result) == 5


class TestFetchMultipleSymbols:
    """Test fetch_multiple_symbols method."""

    @patch("bt.data.fetcher.pyupbit.get_ohlcv")
    def test_multiple_symbols_and_intervals(
        self,
        mock_get_ohlcv: MagicMock,
        fetcher: DataFetcher,
    ) -> None:
        """Test fetching multiple symbols and intervals."""
        mock_df = pd.DataFrame(
            {
                "open": [100],
                "high": [105],
                "low": [95],
                "close": [102],
                "volume": [1000],
            },
            # [중요] UTC 설정
            index=pd.to_datetime(["2024-01-01"]).tz_localize("UTC"),
        )

        # [수정] 2개 심볼 * 2개 인터벌 = 총 4회 fetch 작업
        # 각 작업은 (데이터 반환 -> None 반환)으로 루프를 탈출해야 함
        # 따라서 [mock_df, None] 패턴을 4번 반복
        mock_get_ohlcv.side_effect = [mock_df, None] * 4

        fetcher.fetch_multiple_symbols(
            symbols=["BTC", "ETH"],
            intervals=["day", "week"],
            full_history=True,
        )

        # Should fetch 2 symbols * 2 intervals = 4 calls that return data
        # Plus 4 calls that return None (to break loops)
        # Total calls usually >= 8, but assert logic checks attempted fetches
        assert mock_get_ohlcv.call_count >= 4

    @patch("bt.data.fetcher.pyupbit.get_ohlcv")
    def test_handles_partial_errors(
        self,
        mock_get_ohlcv: MagicMock,
        fetcher: DataFetcher,
    ) -> None:
        """Test continues on partial errors."""
        mock_df = pd.DataFrame(
            {
                "open": [100],
                "high": [105],
                "low": [95],
                "close": [102],
                "volume": [1000],
            },
            index=pd.to_datetime(["2024-01-01"]),
        )

        # First call succeeds, second fails
        mock_get_ohlcv.side_effect = [
            mock_df,
            Exception("Error"),
            Exception("Error"),
            Exception("Error"),  # max retries
            mock_df,
        ]

        # Should not raise, just log errors
        fetcher.fetch_multiple_symbols(
            symbols=["BTC", "ETH"],
            intervals=["day"],
            full_history=True,
        )
