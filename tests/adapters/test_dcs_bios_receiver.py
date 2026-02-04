from __future__ import annotations

from adapters.dcs_bios.receiver import DcsBiosReceiver


def test_receiver_merges_state() -> None:
    with DcsBiosReceiver(host="127.0.0.1", port=0, merge_full_state=True) as rx:
        obs1 = rx._process_payload(
            {"seq": 1, "t_wall": 1.0, "aircraft": "FA-18C", "bios": {"A": 1}},
            ("127.0.0.1", 1000),
        )
        obs2 = rx._process_payload(
            {"seq": 2, "t_wall": 2.0, "aircraft": "FA-18C", "bios": {"B": 2}},
            ("127.0.0.1", 1000),
        )
        assert obs1 is not None
        assert obs2 is not None
        assert obs2.payload["bios"]["A"] == 1
        assert obs2.payload["bios"]["B"] == 2
        assert obs2.payload["delta"] == {"B": 2}


def test_receiver_drops_out_of_order() -> None:
    with DcsBiosReceiver(host="127.0.0.1", port=0, merge_full_state=True) as rx:
        obs1 = rx._process_payload(
            {"seq": 2, "t_wall": 1.0, "aircraft": "FA-18C", "bios": {"A": 1}},
            ("127.0.0.1", 1000),
        )
        obs2 = rx._process_payload(
            {"seq": 1, "t_wall": 2.0, "aircraft": "FA-18C", "bios": {"A": 2}},
            ("127.0.0.1", 1000),
        )
        assert obs1 is not None
        assert obs2 is None
