"""Strategy registry for standardized strategy management.

Provides a centralized system for strategy registration, discovery,
and metadata management.
"""

from dataclasses import dataclass
from typing import Any

from bt.exceptions import StrategyError
from bt.interfaces.strategy_types import IStrategy
from bt.strategies.implementations import (
    BuyAndHoldStrategy,
    MomentumStrategy,
    StrategyFactory,
    VBOPortfolioStrategy,
    VolatilityBreakoutStrategy,
)


@dataclass
class StrategyInfo:
    """Information about a registered strategy."""

    name: str
    description: str
    author: str = "Framework Team"
    version: str = "1.0.0"
    category: str = "General"
    tags: list[str] = None
    parameters: dict[str, Any] = None
    examples: list[str] = None
    dependencies: list[str] = None
    performance_notes: str = None


@dataclass
class StrategyRegistration:
    """Represents a registered strategy."""

    strategy_class: type[IStrategy]
    info: StrategyInfo
    factory: StrategyFactory | None = None


class StrategyRegistry:
    """Centralized registry for managing strategy lifecycle."""

    def __init__(self):
        """Initialize the strategy registry."""
        self._strategies: dict[str, StrategyRegistration] = {}
        self._factories: dict[str, StrategyFactory] = {}
        self._categories: dict[str, list[str]] = {}

    def register_strategy(
        self,
        name: str,
        strategy_class: type[IStrategy],
        category: str = "Custom",
        description: str = None,
        author: str = None,
        version: str = "1.0.0",
        tags: list[str] = None,
        factory: StrategyFactory | None = None,
    ) -> None:
        """Register a strategy class.

        Args:
            name: Unique strategy name
            strategy_class: Strategy implementation class
            category: Strategy category
            description: Strategy description
            author: Strategy author
            version: Strategy version
            tags: List of tags for categorization
            factory: Optional factory for strategy creation

        Raises:
            StrategyError: If strategy already exists
        """
        if name in self._strategies:
            existing = self._strategies[name]
            raise StrategyError(
                f"Strategy '{name}' already registered by {existing.info.author}",
                validation_errors=[f"Strategy '{name}' conflicts with existing registration"],
            )

        # Create strategy info
        info = StrategyInfo(
            name=name,
            description=description or f"{name} strategy implementation",
            category=category,
            author=author,
            version=version,
            tags=tags or [],
        )

        # Register the strategy
        self._strategies[name] = StrategyRegistration(
            strategy_class=strategy_class, info=info, factory=factory
        )

        # Update categories
        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(name)

    def register_factory(self, name: str, factory: StrategyFactory) -> None:
        """Register a strategy factory.

        Args:
            name: Factory name
            factory: Factory function
        """
        self._factories[name] = factory

    def get_strategy(self, name: str, **config) -> IStrategy:
        """Create strategy instance by name.

        Args:
            name: Strategy name
            **config: Strategy configuration

        Returns:
            Strategy instance

        Raises:
            StrategyError: If strategy not found or configuration invalid
        """
        if name not in self._strategies:
            available = ", ".join(self._strategies.keys())
            raise StrategyError(
                f"Strategy '{name}' not found. Available: {available}",
                validation_errors=[f"Unknown strategy: {name}"],
            )

        # Get registration
        registration = self._strategies[name]

        # Create strategy instance
        try:
            if registration.factory:
                strategy = registration.factory(**config)
            else:
                strategy = registration.strategy_class(**config)

            # Validate strategy configuration
            errors = strategy.validate()
            if errors:
                raise StrategyError(
                    f"Strategy '{name}' configuration validation failed: {', '.join(errors)}",
                    validation_errors=errors,
                    strategy=name,
                    config=config,
                )

            return strategy

        except Exception as e:
            raise StrategyError(f"Error creating strategy '{name}': {str(e)}", strategy=name) from e

    def list_strategies(self, category: str | None = None) -> list[StrategyInfo]:
        """List all registered strategies.

        Args:
            category: Optional category filter

        Returns:
            List of strategy information
        """
        strategies = []

        for registration in self._strategies.values():
            info = registration.info
            if category is None or info.category == category:
                strategies.append(info)

        return strategies

    def get_strategy_info(self, name: str) -> StrategyInfo | None:
        """Get information about a strategy.

        Args:
            name: Strategy name

        Returns:
            Strategy information or None if not found
        """
        registration = self._strategies.get(name)
        return registration.info if registration else None

    def is_registered(self, name: str) -> bool:
        """Check if a strategy is already registered.

        Args:
            name: Strategy name to check

        Returns:
            True if strategy is registered, False otherwise
        """
        return name in self._strategies

    def get_available_factories(self) -> list[str]:
        """Get list of available factory names."""
        return list(self._factories.keys())

    def get_available_categories(self) -> list[str]:
        """Get list of available strategy categories."""
        return sorted(self._categories.keys())

    def validate_strategy_config(self, name: str, config: dict[str, Any]) -> list[str]:
        """Validate strategy configuration using strategy's own validation.

        Args:
            name: Strategy name
            config: Configuration parameters

        Returns:
            List of validation errors
        """
        try:
            strategy = self.get_strategy(name, **config)
            return strategy.validate()
        except Exception as e:
            return [f"Validation error: {str(e)}"]

    def create_default_strategies(self) -> None:
        """Register all default framework strategies."""
        # Register VBO strategy
        self.register_strategy(
            name="volatility_breakout",
            strategy_class=VolatilityBreakoutStrategy,
            category="Breakout",
            description="Volatility breakout with trend confirmation and momentum allocation",
            author="Framework Team",
        )

        # Register alias
        self.register_strategy(
            name="vbo",
            strategy_class=VolatilityBreakoutStrategy,
            category="Breakout",
            description="Alias for volatility breakout strategy",
            author="Framework Team",
        )

        # Register momentum strategy
        self.register_strategy(
            name="momentum",
            strategy_class=MomentumStrategy,
            category="Momentum",
            description="Pure momentum strategy with equal-weight allocation",
            author="Framework Team",
        )

        # Register buy and hold strategy
        self.register_strategy(
            name="buy_and_hold",
            strategy_class=BuyAndHoldStrategy,
            category="Simple",
            description="Simple buy and hold strategy",
            author="Framework Team",
        )

        # Register VBO Portfolio strategy
        self.register_strategy(
            name="vbo_portfolio",
            strategy_class=VBOPortfolioStrategy,
            category="Portfolio",
            description="Multi-asset VBO strategy with BTC market filter and 1/N allocation",
            author="Framework Team",
        )


# Global strategy registry instance
_strategy_registry: StrategyRegistry | None = None


def get_strategy_registry() -> StrategyRegistry:
    """Get the global strategy registry."""
    global _strategy_registry
    if _strategy_registry is None:
        _strategy_registry = StrategyRegistry()
        _strategy_registry.create_default_strategies()
    return _strategy_registry


def set_strategy_registry(registry: StrategyRegistry) -> None:
    """Set the global strategy registry."""
    global _strategy_registry
    _strategy_registry = registry


# Decorator for strategy registration


def register_strategy(
    name: str,
    category: str = "Custom",
    description: str = None,
    author: str = None,
    version: str = "1.0.0",
    tags: list[str] = None,
    factory: StrategyFactory | None = None,
):
    """Decorator for automatic strategy registration."""

    def decorator(strategy_class: type[IStrategy]):
        original_init = strategy_class.__init__

        def _init_wrapper(self, *args, **kwargs):
            original_init(self, *args, **kwargs)

            # Register the strategy if not already registered
            registry = get_strategy_registry()
            if not registry.is_registered(name):
                registry.register_strategy(
                    name=name,
                    strategy_class=strategy_class,
                    category=category,
                    description=description,
                    author=author,
                    version=version,
                    tags=tags,
                    factory=factory,
                )

        strategy_class.__init__ = _init_wrapper
        return strategy_class

    return decorator


# Factory for creating strategies with configuration validation


class ConfigurableStrategyFactory:
    """Factory that creates strategies with configuration validation."""

    def __init__(self, default_config: dict[str, Any] = None):
        """Initialize factory with default configuration."""
        self.default_config = default_config or {}

    def create_strategy(
        self, strategy_name: str, config_overrides: dict[str, Any] = None
    ) -> IStrategy:
        """Create strategy instance with validation.

        Args:
            strategy_name: Name of strategy to create
            config_overrides: Configuration overrides

        Returns:
            Configured strategy instance
        """
        # Merge configurations
        final_config = {**self.default_config, **config_overrides}

        # Create strategy with validation
        registry = get_strategy_registry()

        # Validate final configuration
        errors = registry.validate_strategy_config(strategy_name, final_config)
        if errors:
            raise StrategyError(
                f"Invalid configuration for {strategy_name}: {', '.join(errors)}",
                strategy=strategy_name,
                config=final_config,
            )

        return registry.get_strategy(strategy_name, **final_config)


# Metadata and introspection utilities


def get_all_strategy_metadata() -> dict[str, dict[str, Any]]:
    """Get metadata for all registered strategies."""
    registry = get_strategy_registry()
    metadata = {}

    for name, registration in registry._strategies.items():
        metadata[name] = {
            "name": registration.info.name,
            "description": registration.info.description,
            "category": registration.info.category,
            "author": registration.info.author,
            "version": registration.info.version,
            "tags": registration.info.tags,
            "class_name": registration.strategy_class.__name__,
            "has_factory": registration.factory is not None,
            "dependencies": _get_strategy_dependencies(registration.strategy_class),
        }

    return metadata


def _get_strategy_dependencies(strategy_class: type[IStrategy]) -> list[str]:
    """Analyze strategy dependencies."""
    dependencies = []

    # Check import statements for dependencies
    import inspect

    source = inspect.getsource(strategy_class)

    # Look for imports from strategy components
    if "bt.strategies.components" in source:
        dependencies.append("bt.strategies.components")
    if "bt.strategies.implementations" in source:
        dependencies.append("bt.strategies.implementations")

    # Check for common external dependencies
    common_dependencies = ["numpy", "pandas", "matplotlib"]

    for dep in common_dependencies:
        if dep in source:
            dependencies.append(dep)

    return list(set(dependencies))
