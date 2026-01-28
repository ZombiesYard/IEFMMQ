from pathlib import Path

import yaml


def test_taxonomy_loads_and_has_categories():
    path = Path("packs/fa18c_startup/taxonomy.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["version"] == "v1"
    cats = {c["code"]: c for c in data["taxonomy"]["categories"]}
    for code in ["OM", "CO", "OR", "PA", "SV"]:
        assert code in cats
        assert "weight" in cats[code]
    scoring = data["scoring"]
    assert scoring["critical_step_multiplier"] == 1.5
    assert scoring["rounding"] == "ceil"

