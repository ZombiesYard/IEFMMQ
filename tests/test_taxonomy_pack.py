from pathlib import Path

import yaml


def test_taxonomy_loads_and_has_categories():
    path = Path(__file__).resolve().parent.parent / "packs" / "fa18c_startup" / "taxonomy.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["version"] == "v1"
    taxonomy = data["taxonomy"]
    cats = {c["code"]: c for c in taxonomy["categories"]}
    for code in ["OM", "CO", "OR", "PA", "SV"]:
        assert code in cats
        assert cats[code]["name"]
        assert cats[code]["definition"]
        assert "weight" in cats[code]
        assert isinstance(cats[code]["weight"], int)
    scoring = data["scoring"]
    assert "base_weights" in scoring
    assert scoring["critical_step_multiplier"] == 1.5
    assert scoring["rounding"] == "ceil"
    assert scoring["per_step_formula"]
    assert isinstance(scoring["notes"], list)

    base_weights = scoring["base_weights"]
    assert set(base_weights.keys()) == set(cats.keys())
    for code, cat in cats.items():
        assert cat["weight"] == base_weights[code]

    trial_flags = taxonomy["trial_flags"]
    assert isinstance(trial_flags, list)
    assert trial_flags
    for flag in trial_flags:
        assert flag["code"]
        assert flag["name"]
        assert flag["scope"]
        assert flag["definition"]

    decision_rules = data["decision_rules"]
    assert decision_rules["description"]
    rules = decision_rules["rules"]
    assert isinstance(rules, list)
    assert rules
    for rule in rules:
        assert rule["id"]
        assert rule["category"]
        assert rule["target_step"]
        assert "placeholder" in rule
        assert rule["logic"]

    metadata = data["metadata"]
    assert metadata["source_document"]
    assert metadata["maintained_by"]
    assert metadata["notes"]

