"""Strategy-related protocols following Interface Segregation Principle.

Smaller, more focused strategy interfaces for better flexibility.
"""

from typing import Any, Protocol


class IStrategyConditions(Protocol):
    """Interface for strategy conditions only.

    Follows Interface Segregation Principle - only condition-related methods.
    """

    def get_buy_conditions(self) -> dict[str, Any]:
        """Get buy condition functions.

        Returns:
            Dictionary of condition name -> function
        """
        ...

    def get_sell_conditions(self) -> dict[str, Any]:
        """Get sell condition functions.

        Returns:
            Dictionary of condition name -> function
        """
        ...


class IStrategyPricing(Protocol):
    """Interface for strategy pricing only.

    Follows Interface Segregation Principle - only pricing-related methods.
    """

    def get_buy_price_func(self) -> Any:
        """Get buy price function.

        Returns:
            Function to calculate buy price
        """
        ...

    def get_sell_price_func(self) -> Any:
        """Get sell price function.

        Returns:
            Function to calculate sell price
        """
        ...


class IStrategyAllocation(Protocol):
    """Interface for strategy allocation only.

    Follows Interface Segregation Principle - only allocation-related methods.
    """

    def get_allocation_func(self) -> Any:
        """Get position sizing function.

        Returns:
            Function to calculate quantity
        """
        ...


class IStrategyMetadata(Protocol):
    """Interface for strategy metadata only.

    Follows Interface Segregation Principle - only metadata methods.
    """

    def get_name(self) -> str:
        """Get strategy name.

        Returns:
            Strategy name
        """
        ...

    def get_description(self) -> str:
        """Get strategy description.

        Returns:
            Strategy description
        """
        ...

    def get_category(self) -> str:
        """Get strategy category.

        Returns:
            Strategy category
        """
        ...


class IStrategyConfiguration(Protocol):
    """Interface for strategy configuration only.

    Follows Interface Segregation Principle - only configuration methods.
    """

    def get_config(self) -> dict[str, Any]:
        """Get strategy configuration.

        Returns:
            Configuration dictionary
        """
        ...

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        """Validate strategy configuration.

        Args:
            config: Configuration to validate

        Returns:
            List of validation errors (empty if valid)
        """
        ...


# Composite interface for full strategy functionality


class IFullStrategy(
    IStrategyConditions,
    IStrategyPricing,
    IStrategyAllocation,
    IStrategyMetadata,
    IStrategyConfiguration,
    Protocol,
):
    """Complete strategy interface composing all strategy capabilities.

    Use this when you need full strategy functionality.
    Use individual interfaces when you only need specific capabilities.

    Example:
    - Backtester needs: IStrategyConditions, IStrategyPricing, IStrategyAllocation
    - Strategy registry needs: IStrategyMetadata
    - Configuration validator needs: IStrategyConfiguration

    This follows Interface Segregation Principle by:
    - Providing small, focused interfaces
    - Allowing clients to depend on only what they need
    - Composing interfaces for full functionality
    """

    pass


# Minimal strategy interface for simple strategies


class ISimpleStrategy(IStrategyConditions, IStrategyPricing, IStrategyAllocation, Protocol):
    """Minimal strategy interface for simple strategies.

    Only includes execution-related interfaces, not metadata or configuration.
    Use this for lightweight strategy implementations.
    """

    pass
