"""Core module for backtesting framework."""

from .bootstrap import register_core_services
from .container import (
    Container,
    ContainerError,
    CoreServices,
    IContainer,
    ServiceDescriptor,
    ServiceLifetime,
    get_default_container,
    set_default_container,
)
from .orchestrator import BacktestOrchestrator
from .registry import StrategyFactory, StrategyInfo, StrategyRegistry, get_strategy_registry

__all__ = [
    "Container",
    "ContainerError",
    "CoreServices",
    "IContainer",
    "ServiceDescriptor",
    "ServiceLifetime",
    "get_default_container",
    "set_default_container",
    "register_core_services",
    "BacktestOrchestrator",
    "StrategyFactory",
    "StrategyInfo",
    "StrategyRegistry",
    "get_strategy_registry",
]
