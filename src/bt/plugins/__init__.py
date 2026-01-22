"""Plugin system for extensibility and third-party integrations.

Provides a framework for loading and managing plugins that extend
the backtesting framework with custom strategies, data providers,
and reporting components.
"""

from bt.plugins.interfaces import (
    DataProviderPlugin,
    PluginInterface,
    ReporterPlugin,
    StrategyPlugin,
)
from bt.plugins.manager import PluginManager
from bt.plugins.metadata import PluginMetadata
from bt.plugins.registry import PluginRegistry
from bt.plugins.templates import (
    create_plugin_template,
    get_plugin_registry,
    load_plugin_from_entry_point,
    load_plugin_from_file,
    set_plugin_registry,
)

__all__ = [
    # Metadata
    "PluginMetadata",
    # Interfaces
    "PluginInterface",
    "StrategyPlugin",
    "DataProviderPlugin",
    "ReporterPlugin",
    # Manager
    "PluginManager",
    # Registry
    "PluginRegistry",
    # Utility functions
    "get_plugin_registry",
    "set_plugin_registry",
    "load_plugin_from_entry_point",
    "load_plugin_from_file",
    "create_plugin_template",
]
