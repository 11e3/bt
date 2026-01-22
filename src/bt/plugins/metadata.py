"""Plugin metadata definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pkg_resources


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""

    name: str
    version: str
    description: str
    author: str
    entry_points: dict[str, str]
    dependencies: list[str] = field(default_factory=list)
    homepage: str | None = None
    license: str | None = None

    @classmethod
    def from_distribution(cls, dist: pkg_resources.Distribution) -> PluginMetadata:
        """Create metadata from a pkg_resources distribution."""
        dist.get_metadata("METADATA")
        requires = dist.requires() if hasattr(dist, "requires") else []
        deps: list[str] = [str(r) for r in requires] if requires else []
        return cls(
            name=dist.project_name,
            version=dist.version,
            description=dist.get_metadata("DESCRIPTION") or "",
            author=dist.get_metadata("AUTHOR") or "",
            entry_points={},  # Will be filled by entry point scanning
            dependencies=deps,
        )
