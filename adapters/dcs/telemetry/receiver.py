from __future__ import annotations

import socket
from typing import Optional, Tuple

from core.types import Observation
from core.types_v2 import DcsObservation

from adapters.dcs.telemetry.codec import decode_dcs_observation


class DcsTelemetryReceiver:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7780,
        timeout: float = 0.5,
        source: str = "dcs",
    ) -> None:
        self.server = (host, port)
        self.source = source
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.server)
        self.sock.settimeout(timeout)
        self._last_seq: Optional[int] = None

    def close(self) -> None:
        self.sock.close()

    def _process_frame(self, frame: DcsObservation, addr: Tuple[str, int]) -> Optional[Observation]:
        seq = frame.seq
        gap = 0
        if self._last_seq is not None:
            if seq <= self._last_seq:
                return None
            gap = max(0, seq - self._last_seq - 1)
        self._last_seq = seq
        metadata = {"seq": seq, "gap": gap, "from_addr": f"{addr[0]}:{addr[1]}"}
        return Observation(source=self.source, payload=frame.to_dict(), metadata=metadata)

    def get_observation(self) -> Optional[Observation]:
        try:
            data, addr = self.sock.recvfrom(65535)
        except socket.timeout:
            return None
        except OSError:
            return None
        try:
            frame = decode_dcs_observation(data)
        except ValueError:
            return None
        return self._process_frame(frame, addr)

    def __enter__(self) -> "DcsTelemetryReceiver":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

