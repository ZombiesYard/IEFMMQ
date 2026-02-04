from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Iterable, Iterator, Union

from core.types_v2 import TelemetryFrame


class TelemetryWriter:
    def __init__(self, path: Union[str, Path], compression: str = "none"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self._fh = self._open_write(self.path, compression)

    def _open_write(self, path: Path, compression: str):
        if compression == "none":
            return path.open("w", encoding="utf-8")
        if compression == "gzip":
            import gzip

            return gzip.open(path, "wt", encoding="utf-8")
        if compression == "zstd":
            try:
                import zstandard as zstd  # type: ignore
            except Exception as exc:
                raise RuntimeError("zstd compression requested but zstandard is not installed") from exc
            raw = path.open("wb")
            stream = zstd.ZstdCompressor().stream_writer(raw)
            return io.TextIOWrapper(stream, encoding="utf-8")
        raise ValueError(f"Unknown compression: {compression}")

    def append(self, frame: TelemetryFrame | dict) -> None:
        payload = frame if isinstance(frame, dict) else frame.to_dict()
        self._fh.write(json.dumps(payload))
        self._fh.write("\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh and not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "TelemetryWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def iter_frames(path: Union[str, Path]) -> Iterator[dict]:
        path = Path(path)
        opener = TelemetryWriter._open_read(path)
        with opener as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    @staticmethod
    def load(path: Union[str, Path]) -> list[dict]:
        return list(TelemetryWriter.iter_frames(path))

    @staticmethod
    def _open_read(path: Path):
        if path.suffix == ".gz":
            import gzip

            return gzip.open(path, "rt", encoding="utf-8")
        if path.suffix in {".zst", ".zstd"}:
            try:
                import zstandard as zstd  # type: ignore
            except Exception as exc:
                raise RuntimeError("zstd file provided but zstandard is not installed") from exc
            raw = path.open("rb")
            stream = zstd.ZstdDecompressor().stream_reader(raw)
            return io.TextIOWrapper(stream, encoding="utf-8")
        return path.open("r", encoding="utf-8")

