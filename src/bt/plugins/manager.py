"""Plugin manager for plugin lifecycle and registration."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pkg_resources

from bt.plugins.interfaces import PluginInterface
from bt.plugins.metadata import PluginMetadata
from bt.utils.logging import get_logger

logger = get_logger(__name__)


class PluginManager:
    """Central manager for plugin lifecycle and registration."""

    def __init__(self) -> None:
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
