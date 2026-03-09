from __future__ import annotations

from core.types_v2 import VisionObservation
from ports.vision_port import VisionPort
from simtutor.schemas import SCHEMA_INDEX, load_schema, validate_instance


def test_schema_registry_exposes_vision_observation() -> None:
    assert "vision_observation" in SCHEMA_INDEX
    schema = load_schema("vision_observation")
    assert schema["title"] == "Vision Observation v2"


def test_schema_registry_exposes_vision_frame_manifest_entry() -> None:
    assert "vision_frame_manifest_entry" in SCHEMA_INDEX
    schema = load_schema("vision_frame_manifest_entry")
    assert schema["title"] == "Vision Frame Manifest Entry v1"


def test_vision_observation_schema_accepts_defaults() -> None:
    obs = VisionObservation(
        source="vision_stub",
        channel="center_mfd",
        observation_ref="4c0a8ee7-5043-46ee-8e70-d7c689c7f958",
    ).to_dict()
    validate_instance(obs, "vision_observation")


def test_vision_observation_schema_accepts_layout_id() -> None:
    obs = VisionObservation(
        frame_id="1772872444902_000123",
        source="vision_stub",
        channel="native_viewports_strip",
        layout_id="fa18c_composite_panel_v2",
        capture_wall_ms=1772872444902,
        frame_seq=123,
        image_uri="/tmp/frames/sess-live/composite_panel/artifacts/1772872444902_000123_vlm.png",
        source_image_path="/tmp/frames/sess-live/composite_panel/1772872444902_000123.png",
        width=880,
        height=1440,
        source_session_id="sess-live",
    ).to_dict()
    validate_instance(obs, "vision_observation")


def test_vision_port_protocol_shape() -> None:
    class DummyVisionAdapter:
        def start(self, session_id: str) -> None:
            self.session_id = session_id

        def poll(self) -> list[VisionObservation]:
            return []

        def stop(self) -> None:
            self.session_id = None

    adapter = DummyVisionAdapter()
    assert isinstance(adapter, VisionPort)
