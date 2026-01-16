"""Strategy management and creation.

Responsible for managing strategy registry and creating strategy instances.
Follows Single Responsibility Principle.
"""

from typing import Any

from bt.core.registry import StrategyFactory, get_strategy_registry
from bt.interfaces.protocols import ILogger, IStrategy
from bt.utils.logging import get_logger


class StrategyManager:
    """Manages strategy registry and creation.

    Responsibilities:
    - List available strategies
    - Get strategy information
    - Create strategy instances
    - Validate strategy configurations

    Does NOT handle:
    - Backtest execution
    - Data loading
    - Report generation
    """

    def __init__(self, logger: ILogger | None = None):
        """Initialize strategy manager.

        Args:
            logger: Logger instance
        """
        self._registry = get_strategy_registry()
        self._factory = StrategyFactory()
        self.logger = logger or get_logger(__name__)

    def list_strategies(self, category: str | None = None) -> list[str]:
        """List all available strategies.

        Args:
            category: Optional category filter

        Returns:
            List of strategy names
        """
        return self._registry.list_strategies(category)

    def get_strategy_info(self, strategy_name: str) -> dict[str, Any] | None:
        """Get information about a strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Strategy information dictionary or None if not found
        """
        return self._registry.get_strategy_info(strategy_name)

    def create_strategy(
        self, strategy_name: str, config: dict[str, Any] | None = None, **kwargs
    ) -> IStrategy:
        """Create strategy instance with configuration.

        Args:
            strategy_name: Name of the strategy
            config: Strategy configuration
            **kwargs: Additional keyword arguments

        Returns:
            Strategy instance

        Raises:
            ValueError: If strategy not found or configuration invalid
        """
        try:
            strategy = self._registry.get_strategy(strategy_name, **(config or {}), **kwargs)
            self.logger.info(f"Created strategy: {strategy_name}")
            return strategy
        except Exception as e:
            self.logger.error(f"Error creating strategy '{strategy_name}': {e}")
            raise

    def validate_config(self, strategy_name: str, config: dict[str, Any]) -> list[str]:
        """Validate strategy configuration.

        Args:
            strategy_name: Name of the strategy
            config: Configuration to validate

        Returns:
            List of validation errors (empty if valid)
        """
        return self._registry.validate_strategy_config(strategy_name, config)

    def get_categories(self) -> list[str]:
        """Get all available strategy categories.

        Returns:
            List of category names
        """
        return self._registry.get_available_categories()
