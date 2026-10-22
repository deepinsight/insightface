"""Optional model addons, independent of the selected base model package."""

from .catalog import ADDON_CATALOG, ensure_addon
from .liveness import Liveness

__all__ = ["ADDON_CATALOG", "Liveness", "ensure_addon"]
