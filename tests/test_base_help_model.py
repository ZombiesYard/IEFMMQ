from __future__ import annotations

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
