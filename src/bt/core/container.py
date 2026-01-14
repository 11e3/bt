"""Dependency injection container for service management.

Provides IoC (Inversion of Control) pattern for better testability,
loose coupling, and centralized service configuration.
"""

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..interfaces import (
    IChartGenerator,
    IDataProvider,
    ILogger,
    IMetricsGenerator,
    IPortfolio,
)


class ServiceLifetime(Enum):
    """Service lifetime management."""

    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


@dataclass
class ServiceDescriptor:
    """Describes a service dependency."""

    interface: type
    factory: Callable[[], Any] | None = None
    implementation: type | None = None
    lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT
    dependencies: list[type] | None = None
    config_key: str | None = None


class ContainerError(Exception):
    """Container-related errors."""

    pass


class IContainer(ABC):
    """Interface for dependency injection containers."""

    @abstractmethod
    def register_singleton(self, interface: type, implementation: type) -> None:
        """Register singleton service."""
        pass

    @abstractmethod
    def register_factory(self, interface: type, factory: Callable[[], Any]) -> None:
        """Register factory for service creation."""
        pass

    @abstractmethod
    def register_transient(self, interface: type, implementation: type) -> None:
        """Register transient service."""
        pass

    @abstractmethod
    def register_scoped(self, interface: type, implementation: type) -> None:
        """Register scoped service."""
        pass

    @abstractmethod
    def get(self, interface: type) -> Any:
        """Resolve service dependency."""
        pass

    @abstractmethod
    def is_registered(self, interface: type) -> bool:
        """Check if interface is registered."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all registrations."""
        pass


class ServiceRegistry:
    """Registry for service descriptors and implementations."""

    def __init__(self):
        self._services: dict[type, ServiceDescriptor] = {}
        self._instances: dict[type, Any] = {}
        self._lock = threading.RLock()

    def register_service(self, descriptor: ServiceDescriptor) -> None:
        """Register a service with its descriptor."""
        with self._lock:
            self._services[descriptor.interface] = descriptor

    def register_implementation(self, interface: type, implementation: type) -> None:
        """Register an implementation for an interface."""
        with self._lock:
            if interface not in self._services:
                self._services[interface] = ServiceDescriptor(interface=interface)

            descriptor = self._services[interface]
            descriptor.implementation = implementation

    def register_factory(self, interface: type, factory: Callable[[], Any]) -> None:
        """Register a factory function for an interface."""
        with self._lock:
            if interface not in self._services:
                self._services[interface] = ServiceDescriptor(interface=interface)

            descriptor = self._services[interface]
            descriptor.factory = factory

    def get_descriptor(self, interface: type) -> ServiceDescriptor | None:
        """Get service descriptor for interface."""
        with self._lock:
            return self._services.get(interface)

    def create_instance(self, interface: type, container: "IContainer") -> Any:
        """Create instance of registered service."""
        with self._lock:
            descriptor = self._services.get(interface)

            if descriptor is None:
                raise ContainerError(f"No service registered for interface: {interface}")

            # Handle singleton instances
            if descriptor.lifetime == ServiceLifetime.SINGLETON:
                if interface in self._instances:
                    return self._instances[interface]

                instance = self._create_instance(descriptor, container)
                self._instances[interface] = instance
                return instance

            # Create new instance for transient/scoped
            return self._create_instance(descriptor, container)

    def _create_instance(self, descriptor: ServiceDescriptor, container: "IContainer") -> Any:
        """Create instance using factory or implementation."""
        if descriptor.factory:
            return descriptor.factory()

        if descriptor.implementation:
            # Inject dependencies
            return self._create_with_dependencies(
                descriptor.implementation, container, descriptor.dependencies or []
            )

        raise ContainerError(f"No factory or implementation for {descriptor.interface}")

    def _create_with_dependencies(
        self, implementation: type, container: "IContainer", dependencies: list[type]
    ) -> Any:
        """Create instance with dependency injection."""
        if not dependencies:
            return implementation()

        # Resolve dependencies
        resolved_deps = {}
        for dep in dependencies:
            resolved_deps[dep.__name__] = container.get(dep)

        return implementation(**resolved_deps)

    def list_services(self) -> dict[type, ServiceDescriptor]:
        """List all registered services."""
        with self._lock:
            return self._services.copy()

    def clear(self) -> None:
        """Clear all registrations and instances."""
        with self._lock:
            self._services.clear()
            self._instances.clear()


class Container(IContainer):
    """Dependency injection container implementation."""

    def __init__(self, name: str = "default"):
        self.name = name
        self._registry = ServiceRegistry()
        self._config = {}
        self._parent = None
        self._scoped_instances = {}  # For scoped lifetime

    def register_singleton(self, interface: type, implementation: type) -> None:
        """Register singleton service."""
        descriptor = ServiceDescriptor(
            interface=interface, implementation=implementation, lifetime=ServiceLifetime.SINGLETON
        )
        self._registry.register_service(descriptor)

    def register_factory(self, interface: type, factory: Callable[[], Any]) -> None:
        """Register factory for service creation."""
        descriptor = ServiceDescriptor(
            interface=interface, factory=factory, lifetime=ServiceLifetime.TRANSIENT
        )
        self._registry.register_service(descriptor)

    def register_transient(self, interface: type, implementation: type) -> None:
        """Register transient service."""
        descriptor = ServiceDescriptor(
            interface=interface, implementation=implementation, lifetime=ServiceLifetime.TRANSIENT
        )
        self._registry.register_service(descriptor)

    def register_scoped(self, interface: type, implementation: type) -> None:
        """Register scoped service."""
        descriptor = ServiceDescriptor(
            interface=interface, implementation=implementation, lifetime=ServiceLifetime.SCOPED
        )
        self._registry.register_service(descriptor)

    def register_by_descriptor(self, descriptor: ServiceDescriptor) -> None:
        """Register service using descriptor."""
        self._registry.register_service(descriptor)

    def get(self, interface: type) -> Any:
        """Resolve service dependency."""
        return self._registry.create_instance(interface, self)

    def is_registered(self, interface: type) -> bool:
        """Check if interface is registered."""
        return self._registry.get_descriptor(interface) is not None

    def clear(self) -> None:
        """Clear all registrations."""
        self._registry.clear()

    def set_config(self, config: dict[str, Any]) -> None:
        """Set configuration for service creation."""
        self._config = config.copy()

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)

    def create_child_container(self, name: str) -> "Container":
        """Create child container with parent relationship."""
        child = Container(name)
        child._parent = self
        return child

    def list_services(self) -> dict[type, ServiceDescriptor]:
        """List all registered services."""
        return self._registry.list_services()


# Global default container
_default_container = Container("default")


def get_default_container() -> Container:
    """Get the default global container."""
    return _default_container


def set_default_container(container: Container) -> None:
    """Set the default global container."""
    global _default_container
    _default_container = container


# Decorators for automatic dependency injection


def inject(interface: type, name: str | None = None):
    """Decorator for automatic dependency injection in constructors."""

    def decorator(cls):
        original_init = cls.__init__

        def __init__(self, *args, **kwargs):
            # Get container (first from kwargs, then from global)
            container = kwargs.pop("_container", None)
            if container is None:
                container = get_default_container()

            # Store injected dependencies
            if not hasattr(self, "_injected_deps"):
                self._injected_deps = {}

            dependency_name = name or interface.__name__
            self._injected_deps[dependency_name] = container.get(interface)

            # Call original init
            original_init(self, *args, **kwargs)

        cls.__init__ = __init__
        return cls

    return decorator


def auto_wire(interface_map: dict[type, str]):
    """Decorator to automatically wire all dependencies for a class."""

    def decorator(cls):
        original_init = cls.__init__

        def __init__(self, *args, **kwargs):
            container = kwargs.pop("_container", None)
            if container is None:
                container = get_default_container()

            # Auto-wire all dependencies
            for interface, param_name in interface_map.items():
                if not hasattr(self, "_injected_deps"):
                    self._injected_deps = {}

                self._injected_deps[param_name] = container.get(interface)

            original_init(self, *args, **kwargs)

        cls.__init__ = __init__
        return cls

    return decorator


# Service descriptors for common framework components


class CoreServices:
    """Descriptor definitions for core framework services."""

    DATA_PROVIDER = ServiceDescriptor(interface=IDataProvider, lifetime=ServiceLifetime.SINGLETON)

    PORTFOLIO = ServiceDescriptor(interface=IPortfolio, lifetime=ServiceLifetime.SINGLETON)

    LOGGER = ServiceDescriptor(interface=ILogger, lifetime=ServiceLifetime.SINGLETON)

    METRICS_GENERATOR = ServiceDescriptor(
        interface=IMetricsGenerator, lifetime=ServiceLifetime.SINGLETON
    )

    CHART_GENERATOR = ServiceDescriptor(
        interface=IChartGenerator, lifetime=ServiceLifetime.SINGLETON
    )
