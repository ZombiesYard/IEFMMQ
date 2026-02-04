from __future__ import annotations

import json
from pathlib import Path

from adapters.dcs_bios.receiver import DcsBiosRawReceiver, SYNC


def _build_sample_reference(path: Path) -> None:
    data = {
        "Meta": {
            "_ACFT_NAME": {
                "identifier": "_ACFT_NAME",
                "category": "Meta",
                "description": "Aircraft Name",
                "control_type": "display",
                "outputs": [
                    {
                        "address": 0,
                        "max_length": 4,
                        "suffix": "",
                        "type": "string",
                    }
                ],
            }
        },
        "TEST": {
            "SWITCH_1": {
                "identifier": "SWITCH_1",
                "category": "TEST",
                "description": "A switch",
                "control_type": "selector",
                "outputs": [
                    {
                        "address": 4,
                        "mask": 65535,
                        "shift_by": 0,
                        "max_value": 65535,
                        "suffix": "",
                        "type": "integer",
                    }
                ],
            }
        },
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_raw_receiver_decodes_frame(tmp_path: Path) -> None:
    ref_path = tmp_path / "controls.json"
    _build_sample_reference(ref_path)

    rx = DcsBiosRawReceiver(
        host="127.0.0.1",
        port=0,
        control_reference_paths=[str(ref_path)],
        include_metadata=False,
    )
    try:
        # Frame: write 4 bytes at addr 0, then 2 bytes at addr 4.
        frame = (
            b"\x00\x00\x04\x00" + b"A\x00\x00\x00" + b"\x04\x00\x02\x00" + b"\x34\x12"
        )
        frames = rx._parse_frames(SYNC + frame + SYNC)
        assert len(frames) == 1
        delta = rx._apply_frame(frames[0])
        assert delta["_ACFT_NAME"] == "A"
        assert delta["SWITCH_1"] == 0x1234
    finally:
        rx.close()

