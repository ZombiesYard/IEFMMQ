from __future__ import annotations

import ipaddress
import json
import socket
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from core.types import Observation


class DcsBiosReceiver:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 7790,
        timeout: float = 0.5,
        source: str = "dcs_bios",
        merge_full_state: bool = True,
    ) -> None:
        self.server = (host, port)
        self.source = source
        self.merge_full_state = merge_full_state
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.server)
        self.sock.settimeout(timeout)
        self._last_seq: Optional[int] = None
        self._state: Dict[str, Any] = {}

    def close(self) -> None:
        self.sock.close()

    def _decode_payload(self, data: bytes) -> Optional[dict]:
        try:
            payload = json.loads(data.decode("utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        if "bios" not in payload or not isinstance(payload.get("bios"), dict):
            return None
        return payload

    def _process_payload(self, payload: dict, addr: Tuple[str, int]) -> Optional[Observation]:
        seq = payload.get("seq")
        if not isinstance(seq, int):
            return None
        if self._last_seq is not None and seq <= self._last_seq:
            return None
        gap = 0
        if self._last_seq is not None:
            gap = max(0, seq - self._last_seq - 1)
        self._last_seq = seq

        delta = payload.get("bios", {})
        if self.merge_full_state:
            self._state.update(delta)
            bios_state = dict(self._state)
        else:
            bios_state = dict(delta)

        frame = {
            "schema_version": "v2",
            "seq": seq,
            "t_wall": float(payload.get("t_wall", time.time())),
            "aircraft": str(payload.get("aircraft", "")),
            "bios": bios_state,
            "delta": delta,
        }
        metadata = {
            "seq": seq,
            "gap": gap,
            "from_addr": f"{addr[0]}:{addr[1]}",
            "delta_count": len(delta),
        }
        return Observation(source=self.source, payload=frame, metadata=metadata)

    def get_observation(self) -> Optional[Observation]:
        try:
            data, addr = self.sock.recvfrom(65535)
        except socket.timeout:
            return None
        except OSError:
            return None
        payload = self._decode_payload(data)
        if not payload:
            return None
        return self._process_payload(payload, addr)

    def __enter__(self) -> "DcsBiosReceiver":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


SYNC = b"\x55\x55\x55\x55"
STATE_SIZE = 65536


@dataclass(frozen=True)
class OutputDef:
    identifier: str
    address: int
    kind: str
    mask: int = 0xFFFF
    shift_by: int = 0
    max_length: int = 0


def _flatten_controls(data: dict) -> Iterable[tuple[str, dict]]:
    for category, controls in data.items():
        if not isinstance(controls, dict):
            continue
        for _, control in controls.items():
            if not isinstance(control, dict):
                continue
            yield category, control


def _load_control_reference(paths: Iterable[Path]) -> list[OutputDef]:
    outputs: list[OutputDef] = []
    for path in paths:
        if not path or not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for category, control in _flatten_controls(data):
            identifier = str(control.get("identifier") or "")
            if not identifier:
                continue
            for out in control.get("outputs", []) or []:
                if not isinstance(out, dict):
                    continue
                address = out.get("address")
                if address is None:
                    continue
                kind = str(out.get("type") or "integer")
                outputs.append(
                    OutputDef(
                        identifier=identifier,
                        address=int(address),
                        kind=kind,
                        mask=int(out.get("mask") or 0xFFFF),
                        shift_by=int(out.get("shift_by") or 0),
                        max_length=int(out.get("max_length") or 0),
                    )
                )
    return outputs


def _build_address_map(outputs: Iterable[OutputDef]) -> dict[int, list[OutputDef]]:
    mapping: dict[int, list[OutputDef]] = {}
    for out in outputs:
        if out.kind == "string" and out.max_length > 0:
            for addr in range(out.address, out.address + out.max_length):
                mapping.setdefault(addr, []).append(out)
        else:
            mapping.setdefault(out.address, []).append(out)
    return mapping


class DcsBiosRawReceiver:
    def __init__(
        self,
        host: str = "239.255.50.10",
        port: int = 5010,
        timeout: float = 0.5,
        source: str = "dcs_bios_raw",
        merge_full_state: bool = True,
        control_reference_dir: str | None = None,
        aircraft: str | None = None,
        control_reference_paths: Iterable[str] | None = None,
        include_metadata: bool = True,
    ) -> None:
        self.source = source
        self.merge_full_state = merge_full_state
        self._buffer = bytearray()
        self._state = bytearray(STATE_SIZE)
        self._values: Dict[str, Any] = {}
        self._queue: list[Observation] = []
        self._frame_seq = 0
        self._aircraft = aircraft or ""

        ref_paths = []
        if control_reference_paths:
            ref_paths = [Path(p) for p in control_reference_paths]
        else:
            base_dir = (
                Path(control_reference_dir)
                if control_reference_dir
                else Path("DCS/Scripts/DCS-BIOS/doc/json")
            )
            if aircraft:
                ref_paths.append(base_dir / f"{aircraft}.json")
            if include_metadata:
                ref_paths.append(base_dir / "MetadataStart.json")
        outputs = _load_control_reference(ref_paths)
        self._addr_map = _build_address_map(outputs)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        is_multicast = ipaddress.ip_address(host).is_multicast
        if is_multicast:
            self.sock.bind(("", port))
            mreq = struct.pack("4s4s", socket.inet_aton(host), socket.inet_aton("0.0.0.0"))
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        else:
            self.sock.bind((host, port))
        self.sock.settimeout(timeout)

    def close(self) -> None:
        self.sock.close()

    def _parse_frames(self, data: bytes) -> list[bytes]:
        self._buffer.extend(data)
        frames: list[bytes] = []
        while True:
            start = self._buffer.find(SYNC)
            if start == -1:
                if len(self._buffer) > 3:
                    self._buffer = self._buffer[-3:]
                break
            if start > 0:
                del self._buffer[:start]
            next_sync = self._buffer.find(SYNC, 4)
            if next_sync == -1:
                break
            frame = bytes(self._buffer[4:next_sync])
            frames.append(frame)
            del self._buffer[:next_sync]
        return frames

    def _read_uint16(self, address: int) -> int:
        if address < 0 or address >= len(self._state) - 1:
            return 0
        return self._state[address] | (self._state[address + 1] << 8)

    def _decode_output(self, out: OutputDef) -> Any:
        if out.kind == "string" and out.max_length > 0:
            raw = bytes(self._state[out.address : out.address + out.max_length])
            return raw.split(b"\x00", 1)[0].decode("utf-8", errors="ignore")
        word = self._read_uint16(out.address)
        return (word & out.mask) >> out.shift_by

    def _apply_frame(self, frame: bytes) -> Dict[str, Any]:
        updated_addresses: set[int] = set()
        idx = 0
        while idx + 4 <= len(frame):
            addr = int.from_bytes(frame[idx : idx + 2], "little")
            length = int.from_bytes(frame[idx + 2 : idx + 4], "little")
            idx += 4
            if length <= 0:
                continue
            if idx + length > len(frame):
                break
            end = addr + length
            if end <= len(self._state):
                self._state[addr:end] = frame[idx : idx + length]
                for a in range(addr, end):
                    if a in self._addr_map:
                        updated_addresses.add(a)
            idx += length

        delta: Dict[str, Any] = {}
        seen: set[str] = set()
        for address in updated_addresses:
            for out in self._addr_map.get(address, []):
                if out.identifier in seen:
                    continue
                seen.add(out.identifier)
                value = self._decode_output(out)
                if self._values.get(out.identifier) != value:
                    self._values[out.identifier] = value
                    delta[out.identifier] = value
        if "_ACFT_NAME" in delta and isinstance(delta["_ACFT_NAME"], str):
            self._aircraft = delta["_ACFT_NAME"]
        return delta

    def _enqueue_observation(self, delta: Dict[str, Any], addr: Tuple[str, int]) -> None:
        self._frame_seq += 1
        bios_state = dict(self._values) if self.merge_full_state else dict(delta)
        frame = {
            "schema_version": "v2",
            "seq": self._frame_seq,
            "t_wall": time.time(),
            "aircraft": self._aircraft,
            "bios": bios_state,
            "delta": delta,
        }
        metadata = {
            "seq": self._frame_seq,
            "from_addr": f"{addr[0]}:{addr[1]}",
            "delta_count": len(delta),
        }
        self._queue.append(Observation(source=self.source, payload=frame, metadata=metadata))

    def get_observation(self) -> Optional[Observation]:
        if self._queue:
            return self._queue.pop(0)
        try:
            data, addr = self.sock.recvfrom(65535)
        except socket.timeout:
            return None
        except OSError:
            return None
        frames = self._parse_frames(data)
        for frame in frames:
            delta = self._apply_frame(frame)
            if delta:
                self._enqueue_observation(delta, addr)
        if self._queue:
            return self._queue.pop(0)
        return None

    def __enter__(self) -> "DcsBiosRawReceiver":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
