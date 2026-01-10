"""Example: Fetch cryptocurrency data from Upbit.

Demonstrates:
- Using DataFetcher with error handling
- Configuring logging
- Incremental vs full history updates
"""

from bt.data.fetcher import DataFetcher
from bt.logging import setup_logging


def main() -> None:
    """Fetch market data for multiple symbols and intervals."""
    # Setup logging
    setup_logging(level="INFO", log_format="text")

    # Initialize fetcher
    fetcher = DataFetcher()

    # Define symbols and intervals
    symbols = ["BTC", "ETH", "XRP", "TRX"]
    intervals = [
        "minute60",
        "minute240",
        "day",
        "week",
        "month",
    ]

    # Fetch data for all symbols and intervals
    # Set full_history=True for initial download
    # Set full_history=False for incremental updates
    fetcher.fetch_multiple_symbols(
        symbols=symbols,
        intervals=intervals,
        full_history=True,  # Change to False for updates
    )


if __name__ == "__main__":
    main()
