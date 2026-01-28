"""
SimTutor CLI and utilities.

Only lightweight helpers live here to avoid pulling domain dependencies into
the command-line tooling. Validation uses the JSON Schemas stored in
`schemas/v1`.
"""

__all__ = ["cli"]

