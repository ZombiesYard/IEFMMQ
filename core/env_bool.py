from __future__ import annotations

import logging
import os

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})


def parse_env_bool(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return bool(default)

    normalized = raw_value.strip().lower()
    if normalized in _TRUTHY:
        return True
    if normalized in _FALSY:
        return False

    logging.warning(
        "Invalid boolean environment value for %s=%r; falling back to default=%s",
        name,
        raw_value,
        bool(default),
    )
    return bool(default)


__all__ = ["parse_env_bool"]
