"""Strategy-specific protocols and type definitions.

Provides clean type definitions for strategy components
without circular dependencies.
"""

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from bt.interfaces.protocols import IBacktestEngine


class ConditionFunc(Protocol):
    """Protocol for trading condition functions."""

    def __call__(self, engine: "IBacktestEngine", symbol: str) -> bool:
        """Evaluate condition for given symbol."""
        ...


class PriceFunc(Protocol):
    """Protocol for price calculation functions."""

    def __call__(self, engine: "IBacktestEngine", symbol: str) -> float:
        """Calculate price for given symbol."""
        ...


class AllocationFunc(Protocol):
    """Protocol for position sizing functions."""

    def __call__(self, engine: "IBacktestEngine", symbol: str, price: float) -> float:
        """Calculate allocation quantity for given symbol."""
        ...


# Type aliases for better readability

ConditionDict = dict[str, ConditionFunc]
StrategyConfig = dict[str, Any]


class IStrategyComponent(Protocol):
    """Protocol for individual strategy components."""

    def validate(self) -> bool:
        """Validate component configuration."""
        ...

    def get_description(self) -> str:
        """Get component description."""
        ...


class ICondition(IStrategyComponent, Protocol):
    """Protocol for trading conditions."""

    def evaluate(self, engine: "IBacktestEngine", symbol: str) -> bool:
        """Evaluate if condition is met."""
        ...


class IAllocation(IStrategyComponent, Protocol):
    """Protocol for allocation strategies."""

    def calculate_quantity(self, engine: "IBacktestEngine", symbol: str, price: float) -> float:
        """Calculate position quantity."""
        ...


class IPricing(IStrategyComponent, Protocol):
    """Protocol for pricing strategies."""

    def calculate_price(self, engine: "IBacktestEngine", symbol: str) -> float:
        """Calculate execution price."""
        ...


class IStrategy(Protocol):
    """Protocol for complete trading strategies."""

    def get_name(self) -> str:
        """Get strategy name."""
        ...

    def get_buy_conditions(self) -> ConditionDict:
        """Get buy condition functions."""
        ...

    def get_sell_conditions(self) -> ConditionDict:
        """Get sell condition functions."""
        ...

    def get_buy_price_func(self) -> PriceFunc:
        """Get buy price function."""
        ...

    def get_sell_price_func(self) -> PriceFunc:
        """Get sell price function."""
        ...

    def get_allocation_func(self) -> AllocationFunc:
        """Get allocation function."""
        ...

    def validate(self) -> list[str]:
        """Validate strategy configuration.

        Returns:
            List of validation errors, empty if valid
        """
        ...
