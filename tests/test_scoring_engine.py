from simtutor.runner import run_simulation, score_run


def test_scoring_distinguishes_scenarios():
    log_good = run_simulation(
        pack_path="packs/fa18c_startup/pack.yaml",
        scenario_path="mock_scenarios/correct_process.json",
        log_path="logs/test_good.jsonl",
    )
    log_missing = run_simulation(
        pack_path="packs/fa18c_startup/pack.yaml",
        scenario_path="mock_scenarios/missing_steps.json",
        log_path="logs/test_missing.jsonl",
    )
    log_prem = run_simulation(
        pack_path="packs/fa18c_startup/pack.yaml",
        scenario_path="mock_scenarios/premature_acceleration.json",
        log_path="logs/test_prem.jsonl",
    )

    score_good = score_run(str(log_good), "packs/fa18c_startup/pack.yaml", "packs/fa18c_startup/taxonomy.yaml")
    score_missing = score_run(str(log_missing), "packs/fa18c_startup/pack.yaml", "packs/fa18c_startup/taxonomy.yaml")
    score_prem = score_run(str(log_prem), "packs/fa18c_startup/pack.yaml", "packs/fa18c_startup/taxonomy.yaml")

    assert score_good["TotalErrorScore"] < score_missing["TotalErrorScore"]
    assert score_good["TotalErrorScore"] < score_prem["TotalErrorScore"] or score_prem["TotalErrorScore"] > 0
    assert score_missing["Count_OM"] > score_good["Count_OM"]
