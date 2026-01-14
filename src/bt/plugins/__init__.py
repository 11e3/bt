"""Plugin system for extensibility and third-party integrations.

Provides a framework for loading and managing plugins that extend
the backtesting framework with custom strategies, data providers,
and reporting components.
"""

import importlib
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import pkg_resources

from bt.exceptions import StrategyError
from bt.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""

    name: str
    version: str
    description: str
    author: str
    entry_points: dict[str, str]
    dependencies: list[str] = None
    homepage: str | None = None
    license: str | None = None

    @classmethod
    def from_distribution(cls, dist: pkg_resources.Distribution) -> "PluginMetadata":
        """Create metadata from a pkg_resources distribution."""
        dist.get_metadata("METADATA")
        return cls(
            name=dist.project_name,
            version=dist.version,
            description=dist.get_metadata("DESCRIPTION") or "",
            author=dist.get_metadata("AUTHOR") or "",
            entry_points={},  # Will be filled by entry point scanning
            dependencies=list(dist.requires()) if hasattr(dist, "requires") else [],
        )


class PluginInterface(ABC):
    """Base interface for all plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version."""
        pass

    @abstractmethod
    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the plugin with configuration."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the plugin and clean up resources."""
        pass


class StrategyPlugin(PluginInterface):
    """Plugin interface for trading strategies."""

    @abstractmethod
    def get_strategy_class(self) -> type:
        """Return the strategy class to register."""
        pass

    @abstractmethod
    def get_strategy_config_schema(self) -> dict[str, Any]:
        """Return JSON schema for strategy configuration."""
        pass


class DataProviderPlugin(PluginInterface):
    """Plugin interface for data providers."""

    @abstractmethod
    def get_data_provider_class(self) -> type:
        """Return the data provider class to register."""
        pass

    @abstractmethod
    def get_supported_symbols(self) -> list[str]:
        """Return list of supported symbols."""
        pass


class ReporterPlugin(PluginInterface):
    """Plugin interface for reporting components."""

    @abstractmethod
    def get_reporter_class(self) -> type:
        """Return the reporter class to register."""
        pass

    @abstractmethod
    def get_supported_formats(self) -> list[str]:
        """Return list of supported output formats."""
        pass


class PluginManager:
    """Central manager for plugin lifecycle and registration."""

    def __init__(self):
        self._plugins: dict[str, PluginInterface] = {}
        self._metadata: dict[str, PluginMetadata] = {}
        self._loaded_plugins: dict[str, Any] = {}
        self._plugin_paths: list[Path] = []

    def register_plugin_path(self, path: str | Path) -> None:
        """Register a path to search for plugins."""
        plugin_path = Path(path)
        if plugin_path.exists():
            self._plugin_paths.append(plugin_path)
            logger.info(f"Registered plugin path: {plugin_path}")

    def discover_plugins(self) -> list[str]:
        """Discover and return list of available plugins."""
        discovered = []

        # Discover via setuptools entry points
        for entry_point in pkg_resources.iter_entry_points("bt.plugins"):
            try:
                plugin_class = entry_point.load()
                plugin_instance = plugin_class()

                plugin_name = plugin_instance.name
                self._plugins[plugin_name] = plugin_instance

                # Create metadata
                metadata = PluginMetadata(
                    name=plugin_name,
                    version=plugin_instance.version,
                    description=getattr(plugin_instance, "description", ""),
                    author=getattr(plugin_instance, "author", "Unknown"),
                    entry_points={entry_point.name: entry_point.module_name},
                )

                self._metadata[plugin_name] = metadata
                discovered.append(plugin_name)

                logger.info(f"Discovered plugin: {plugin_name} v{plugin_instance.version}")

            except Exception as e:
                logger.error(f"Failed to load plugin {entry_point}: {e}")

        # Discover via plugin paths (for development)
        for plugin_path in self._plugin_paths:
            if plugin_path.is_dir():
                for plugin_file in plugin_path.glob("*.py"):
                    if plugin_file.name != "__init__.py":
                        try:
                            self._load_plugin_from_file(plugin_file)
                            discovered.append(plugin_file.stem)
                        except Exception as e:
                            logger.error(f"Failed to load plugin from {plugin_file}: {e}")

        return discovered

    def _load_plugin_from_file(self, plugin_file: Path) -> None:
        """Load a plugin from a Python file."""
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location(plugin_file.stem, plugin_file)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[plugin_file.stem] = module
            spec.loader.exec_module(module)

            # Find plugin class
            for _name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, PluginInterface)
                    and obj != PluginInterface
                ):
                    plugin_instance = obj()
                    plugin_name = plugin_instance.name

                    self._plugins[plugin_name] = plugin_instance
                    logger.info(f"Loaded plugin from file: {plugin_name}")

                    # Create basic metadata
                    self._metadata[plugin_name] = PluginMetadata(
                        name=plugin_name,
                        version=plugin_instance.version,
                        description=getattr(plugin_instance, "description", ""),
                        author=getattr(plugin_instance, "author", "File Plugin"),
                        entry_points={"file": str(plugin_file)},
                    )
                    break

    def load_plugin(self, name: str, config: dict[str, Any] | None = None) -> PluginInterface:
        """Load and initialize a plugin."""
        if name not in self._plugins:
            raise ValueError(f"Plugin '{name}' not found. Available: {list(self._plugins.keys())}")

        plugin = self._plugins[name]

        if name in self._loaded_plugins:
            logger.warning(f"Plugin '{name}' already loaded")
            return self._loaded_plugins[name]

        try:
            plugin.initialize(config or {})
            self._loaded_plugins[name] = plugin
            logger.info(f"Loaded plugin: {name}")
            return plugin

        except Exception as e:
            logger.error(f"Failed to load plugin '{name}': {e}")
            raise

    def unload_plugin(self, name: str) -> None:
        """Unload a plugin."""
        if name in self._loaded_plugins:
            plugin = self._loaded_plugins[name]
            try:
                plugin.shutdown()
                del self._loaded_plugins[name]
                logger.info(f"Unloaded plugin: {name}")
            except Exception as e:
                logger.error(f"Error unloading plugin '{name}': {e}")

    def get_plugin(self, name: str) -> PluginInterface | None:
        """Get a loaded plugin instance."""
        return self._loaded_plugins.get(name)

    def list_plugins(self) -> dict[str, PluginMetadata]:
        """List all discovered plugins with metadata."""
        return self._metadata.copy()

    def get_loaded_plugins(self) -> dict[str, PluginInterface]:
        """Get all currently loaded plugins."""
        return self._loaded_plugins.copy()

    def is_plugin_loaded(self, name: str) -> bool:
        """Check if a plugin is currently loaded."""
        return name in self._loaded_plugins


class PluginRegistry:
    """Registry for managing different types of plugins."""

    def __init__(self):
        self._strategies: dict[str, type] = {}
        self._data_providers: dict[str, type] = {}
        self._reporters: dict[str, type] = {}
        self._manager = PluginManager()

    def register_strategy_plugin(self, plugin: StrategyPlugin) -> None:
        """Register a strategy plugin."""
        strategy_class = plugin.get_strategy_class()
        strategy_name = getattr(strategy_class, "get_name", lambda: plugin.name)()

        self._strategies[strategy_name] = strategy_class
        logger.info(f"Registered strategy plugin: {strategy_name}")

    def register_data_provider_plugin(self, plugin: DataProviderPlugin) -> None:
        """Register a data provider plugin."""
        provider_class = plugin.get_data_provider_class()
        provider_name = plugin.name

        self._data_providers[provider_name] = provider_class
        logger.info(f"Registered data provider plugin: {provider_name}")

    def register_reporter_plugin(self, plugin: ReporterPlugin) -> None:
        """Register a reporter plugin."""
        reporter_class = plugin.get_reporter_class()
        reporter_name = plugin.name

        self._reporters[reporter_name] = reporter_class
        logger.info(f"Registered reporter plugin: {reporter_name}")

    def get_strategy_class(self, name: str) -> type:
        """Get a strategy class by name."""
        if name not in self._strategies:
            raise StrategyError(f"Strategy plugin '{name}' not found")
        return self._strategies[name]

    def get_data_provider_class(self, name: str) -> type:
        """Get a data provider class by name."""
        if name not in self._data_providers:
            raise ValueError(f"Data provider plugin '{name}' not found")
        return self._data_providers[name]

    def get_reporter_class(self, name: str) -> type:
        """Get a reporter class by name."""
        if name not in self._reporters:
            raise ValueError(f"Reporter plugin '{name}' not found")
        return self._reporters[name]

    def list_strategies(self) -> list[str]:
        """List all registered strategy plugins."""
        return list(self._strategies.keys())

    def list_data_providers(self) -> list[str]:
        """List all registered data provider plugins."""
        return list(self._data_providers.keys())

    def list_reporters(self) -> list[str]:
        """List all registered reporter plugins."""
        return list(self._reporters.keys())

    def discover_and_load_plugins(self, plugin_paths: list[Path] | None = None) -> None:
        """Discover and load all available plugins."""
        if plugin_paths:
            for path in plugin_paths:
                self._manager.register_plugin_path(path)

        discovered = self._manager.discover_plugins()

        for plugin_name in discovered:
            plugin = self._manager.load_plugin(plugin_name)

            # Register with appropriate registry based on type
            if isinstance(plugin, StrategyPlugin):
                self.register_strategy_plugin(plugin)
            elif isinstance(plugin, DataProviderPlugin):
                self.register_data_provider_plugin(plugin)
            elif isinstance(plugin, ReporterPlugin):
                self.register_reporter_plugin(plugin)


# Global plugin registry instance
_plugin_registry = PluginRegistry()


def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry."""
    return _plugin_registry


def set_plugin_registry(registry: PluginRegistry) -> None:
    """Set the global plugin registry."""
    global _plugin_registry
    _plugin_registry = registry


# Plugin loading utilities


def load_plugin_from_entry_point(group: str, name: str) -> PluginInterface | None:
    """Load a plugin from setuptools entry points."""
    try:
        for entry_point in pkg_resources.iter_entry_points(group, name):
            plugin_class = entry_point.load()
            return plugin_class()
    except Exception as e:
        logger.error(f"Failed to load plugin '{name}' from entry points: {e}")

    return None


def load_plugin_from_file(filepath: str | Path) -> PluginInterface | None:
    """Load a plugin from a Python file."""
    try:
        plugin_path = Path(filepath)
        plugin_manager = PluginManager()
        plugin_manager.register_plugin_path(plugin_path.parent)

        # Extract plugin name from filename
        plugin_name = plugin_path.stem
        return plugin_manager.load_plugin(plugin_name)

    except Exception as e:
        logger.error(f"Failed to load plugin from file '{filepath}': {e}")

    return None


# Plugin development utilities


def create_plugin_template(plugin_type: str, name: str, output_dir: str | Path = ".") -> Path:
    """Create a template plugin file."""

    templates = {
        "strategy": f'''"""{{name}} trading strategy plugin."""

from bt.plugins import StrategyPlugin
from bt.strategies.implementations import BaseStrategy


class {name}Strategy(BaseStrategy):
    """Custom {name} trading strategy."""

    def validate(self) -> list[str]:
        """Validate strategy configuration."""
        errors = []
        # Add validation logic here
        return errors

    def get_buy_conditions(self) -> dict:
        """Define buy conditions."""
        return {{
            "custom_condition": lambda engine, symbol: True  # Replace with actual logic
        }}

    def get_allocation_func(self) -> callable:
        """Define position sizing."""
        return lambda engine, symbol, price: 1000  # Replace with actual logic


class {name}StrategyPlugin(StrategyPlugin):
    """Plugin wrapper for {name} strategy."""

    @property
    def name(self) -> str:
        return "{name}"

    @property
    def version(self) -> str:
        return "1.0.0"

    def get_strategy_class(self):
        return {name}Strategy

    def get_strategy_config_schema(self) -> dict:
        return {{
            "type": "object",
            "properties": {{
                "param1": {{"type": "number", "default": 1.0}}
            }}
        }}

    def initialize(self, config: dict) -> None:
        """Initialize the plugin."""
        pass

    def shutdown(self) -> None:
        """Shutdown the plugin."""
        pass
''',
        "data_provider": f'''"""{{name}} data provider plugin."""

from bt.plugins import DataProviderPlugin
from bt.interfaces.protocols import IDataProvider


class {name}DataProvider(IDataProvider):
    """Custom {name} data provider."""

    def __init__(self):
        self._data = {{}}

    @property
    def symbols(self) -> list[str]:
        return list(self._data.keys())

    def load_data(self, symbol: str, df) -> None:
        self._data[symbol] = df

    def get_bar(self, symbol: str, offset: int = 0):
        if symbol in self._data:
            df = self._data[symbol]
            idx = len(df) - 1 - offset
            if idx >= 0:
                return df.iloc[idx]
        return None

    def get_bars(self, symbol: str, count: int):
        if symbol in self._data:
            return self._data[symbol].tail(count)
        return None

    def has_more_data(self) -> bool:
        return any(len(df) > 0 for df in self._data.values())

    def next_bar(self) -> None:
        # Implementation for data iteration
        pass

    def set_current_bar(self, symbol: str, index: int) -> None:
        # Implementation for bar positioning
        pass

    def get_prices_batch(self, symbols: list[str]) -> dict[str, float]:
        return {{symbol: 100.0 for symbol in symbols}}  # Replace with actual logic

    def get_current_datetime_batch(self, symbols: list[str]):
        from datetime import datetime
        return datetime.now()


class {name}DataProviderPlugin(DataProviderPlugin):
    """Plugin wrapper for {name} data provider."""

    @property
    def name(self) -> str:
        return "{name}"

    @property
    def version(self) -> str:
        return "1.0.0"

    def get_data_provider_class(self):
        return {name}DataProvider

    def get_supported_symbols(self) -> list[str]:
        return ["SYMBOL1", "SYMBOL2"]  # Replace with actual symbols

    def initialize(self, config: dict) -> None:
        """Initialize the plugin."""
        pass

    def shutdown(self) -> None:
        """Shutdown the plugin."""
        pass
''',
    }

    if plugin_type not in templates:
        raise ValueError(f"Unknown plugin type: {plugin_type}")

    output_path = Path(output_dir) / f"{name}_plugin.py"
    output_path.write_text(templates[plugin_type].format(name=name))

    return output_path
