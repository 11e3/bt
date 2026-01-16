"""Market data loading functionality.

Responsible for loading market data from various sources.
Follows Single Responsibility Principle.
"""

from typing import Any

from bt.interfaces.protocols import ILogger
from bt.utils.logging import get_logger


class DataLoader:
    """Loads market data for backtesting.

    Responsibilities:
    - Load data from files
    - Validate loaded data
    - Return formatted data

    Does NOT handle:
    - Backtest execution
    - Strategy management
    - Report generation
    """

    def __init__(self, logger: ILogger | None = None):
        """Initialize data loader.

        Args:
            logger: Logger instance
        """
        self.logger = logger or get_logger(__name__)

    def load_from_directory(self, data_directory: str, symbols: list[str]) -> dict[str, Any]:
        """Load market data from directory.

        Args:
            data_directory: Directory containing market data files
            symbols: Symbols to load

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        import pandas as pd

        data = {}

        for symbol in symbols:
            try:
                file_path = f"{data_directory}/{symbol.lower()}.parquet"
                df = pd.read_parquet(file_path)

                if df is not None and not df.empty:
                    data[symbol] = df
                    self.logger.info(f"Loaded {len(df)} bars for {symbol}")
                else:
                    self.logger.warning(f"No data found for {symbol}")

            except Exception as e:
                self.logger.error(f"Error loading data for {symbol}: {e}")

        return data

    def load_from_file(self, file_path: str, symbol: str) -> dict[str, Any]:
        """Load market data for single symbol from file.

        Args:
            file_path: Path to data file
            symbol: Symbol name

        Returns:
            Dictionary with single symbol data
        """
        import pandas as pd

        try:
            # Determine file type from extension
            if file_path.endswith(".parquet"):
                df = pd.read_parquet(file_path)
            elif file_path.endswith(".csv"):
                df = pd.read_csv(file_path, parse_dates=["datetime"])
            else:
                raise ValueError(f"Unsupported file format: {file_path}")

            if df is not None and not df.empty:
                self.logger.info(f"Loaded {len(df)} bars for {symbol} from {file_path}")
                return {symbol: df}
            self.logger.warning(f"No data found in {file_path}")
            return {}

        except Exception as e:
            self.logger.error(f"Error loading data from {file_path}: {e}")
            return {}

    def validate_data(self, data: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate loaded market data.

        Args:
            data: Market data dictionary

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        if not data:
            errors.append("No data loaded")
            return False, errors

        required_columns = ["open", "high", "low", "close", "volume"]

        for symbol, df in data.items():
            # Check if DataFrame is empty
            if df.empty:
                errors.append(f"{symbol}: DataFrame is empty")
                continue

            # Check required columns
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                errors.append(f"{symbol}: Missing columns {missing_cols}")

            # Check for NaN values
            if df[required_columns].isnull().any().any():
                errors.append(f"{symbol}: Contains NaN values")

        is_valid = len(errors) == 0
        return is_valid, errors
