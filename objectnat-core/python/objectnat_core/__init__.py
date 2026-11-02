"""Optional native (Rust) computational core for ObjectNat.

ObjectNat uses this package automatically when it is installed; every function has a pure-Python
fallback inside ObjectNat itself.
"""

from ._core import __version__, visibility_polygons

__all__ = ["__version__", "visibility_polygons"]
