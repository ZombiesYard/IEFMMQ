from __future__ import annotations

from pathlib import Path

from adapters.openai_compat_model import OpenAICompatModel
from tests._fakes import FakeClient


def test_resolve_inference_vars_preserves_original_vars_source_missing() -> None:
    model = OpenAICompatModel(client=FakeClient(responses=[]), lang="en")

    resolved = model._resolve_inference_vars(
        {
            "battery_on": None,
            "apu_on": True,
            "vars_source_missing": ["battery_on"],
        }
    )

    assert "battery_on" in resolved["vars_source_missing"]


def test_resolve_inference_vars_prefers_context_telemetry_map_path(tmp_path: Path) -> None:
    telemetry_map_path = tmp_path / "telemetry_map.yaml"
    telemetry_map_path.write_text(
        "\n".join(
            [
                "vars:",
                '  custom_ready: "derived(vars.switch_on and vars.power_on)"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    model = OpenAICompatModel(client=FakeClient(responses=[]), lang="en")

    resolved = model._resolve_inference_vars(
        {
            "switch_on": True,
            "power_on": True,
        },
        context={"telemetry_map_path": str(telemetry_map_path)},
    )

    assert resolved["custom_ready"] is True


def test_resolve_active_telemetry_map_path_falls_back_to_pack_sibling(tmp_path: Path) -> None:
    pack_dir = tmp_path / "custom_pack"
    pack_dir.mkdir()
    pack_path = pack_dir / "pack.yaml"
    pack_path.write_text("steps: []\n", encoding="utf-8")
    telemetry_map_path = pack_dir / "telemetry_map.yaml"
    telemetry_map_path.write_text("vars: {}\n", encoding="utf-8")
    model = OpenAICompatModel(client=FakeClient(responses=[]), lang="en")

    resolved_path = model._resolve_active_telemetry_map_path({"pack_path": str(pack_path)})

    assert resolved_path == telemetry_map_path.resolve()
