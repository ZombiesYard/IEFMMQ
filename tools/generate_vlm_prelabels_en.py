"""
English-default wrapper for the composite-panel VLM prelabel tool.
"""

from __future__ import annotations

from typing import Sequence

from tools.generate_vlm_prelabels import run_cli


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(default_lang="en", argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
