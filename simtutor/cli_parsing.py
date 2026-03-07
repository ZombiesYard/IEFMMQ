from __future__ import annotations

import argparse
import logging
import os


def parse_env_int(name: str, default: int = 0, *, minimum: int | None = None) -> int:
    raw_value = os.getenv(name)
    fallback = int(default)
    if raw_value is None:
        return fallback

    normalized = raw_value.strip()
    try:
        value = int(normalized)
    except ValueError:
        logging.warning(
            "Invalid integer environment value for %s=%r; falling back to default=%s",
            name,
            raw_value,
            fallback,
        )
        return fallback

    if minimum is not None and value < minimum:
        logging.warning(
            "Out-of-range integer environment value for %s=%r; expected >= %s, falling back to default=%s",
            name,
            raw_value,
            minimum,
            fallback,
        )
        return fallback
    return value


def parse_non_negative_int_arg(raw_value: str) -> int:
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return value


__all__ = ["parse_env_int", "parse_non_negative_int_arg"]
