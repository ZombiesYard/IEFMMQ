"""Compatibility shim for moved config module.

Use `simtutor.config` for imports in packaged/runtime code.
"""

from simtutor.config import *  # noqa: F401,F403

