from __future__ import annotations

import json

from tools.benchmark_qwen35_vlm_facts import (
    CORE_FACT_IDS,
    PredictionRecord,
    ReviewedBenchmarkSample,
    _normalize_prediction,
    compute_metrics,
)


def _facts(state: str = "not_seen") -> dict[str, str]:
    return {fact_id: state for fact_id in CORE_FACT_IDS}


def test_normalize_prediction_accepts_json_and_fills_missing_facts() -> None:
    raw = json.dumps(
        {
            "summary": "ok",
            "facts": [
                {
                    "fact_id": "fcs_page_visible",
                    "state": "seen",
                    "evidence_note": "FCS page is visible.",
                }
            ],
        }
    )

    record = _normalize_prediction(raw, sample_id="sample-1")

    assert record.json_valid is True
    assert record.schema_valid is False
    assert record.facts["fcs_page_visible"] == "seen"
    assert record.facts["ins_go"] == "uncertain"
    assert any(item.startswith("missing_facts:") for item in record.warnings)


def test_normalize_prediction_flags_forbidden_fields() -> None:
    raw = json.dumps(
        {
            "facts": [
                {
                    "fact_id": "ins_go",
                    "state": "seen",
                    "confidence": 0.99,
                    "evidence_note": "GO is visible.",
                }
            ]
        }
    )

    record = _normalize_prediction(raw, sample_id="sample-1")

    assert record.json_valid is True
    assert record.schema_valid is False
    assert record.facts["ins_go"] == "seen"
    assert any("unknown_keys:confidence" in item for item in record.warnings)


def test_compute_metrics_counts_critical_false_positives() -> None:
    gold = _facts("not_seen")
    sample = ReviewedBenchmarkSample(
        sample_id="sample-1",
        image_data_url="data:image/png;base64,AA==",
        facts=gold,
    )
    predicted = _facts("not_seen")
    predicted["ins_go"] = "seen"
    prediction = PredictionRecord(
        sample_id="sample-1",
        raw_text="{}",
        facts=predicted,
        json_valid=True,
        schema_valid=True,
    )

    metrics, errors = compute_metrics(
        samples=[sample],
        predictions={"sample-1": prediction},
        model_label="lora",
    )

    assert metrics["fact_accuracy"] == 0.875
    assert metrics["sample_exact_match"] == 0.0
    assert metrics["critical_false_positive_count"] == 1
    assert metrics["critical_false_positives_by_fact"]["ins_go"] == 1
    assert len(errors) == 1
    assert errors[0]["fact_id"] == "ins_go"


def test_compute_metrics_exact_match() -> None:
    gold = _facts("not_seen")
    sample = ReviewedBenchmarkSample(
        sample_id="sample-1",
        image_data_url="data:image/png;base64,AA==",
        facts=gold,
    )
    prediction = PredictionRecord(
        sample_id="sample-1",
        raw_text="{}",
        facts=dict(gold),
        json_valid=True,
        schema_valid=True,
    )

    metrics, errors = compute_metrics(
        samples=[sample],
        predictions={"sample-1": prediction},
        model_label="base",
    )

    assert metrics["fact_accuracy"] == 1.0
    assert metrics["sample_exact_match"] == 1.0
    assert metrics["critical_false_positive_count"] == 0
    assert errors == []
