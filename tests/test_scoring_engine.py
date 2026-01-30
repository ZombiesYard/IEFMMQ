from pathlib import Path
from time import sleep

from simtutor.runner import run_simulation, score_run


BASE_DIR = Path(__file__).resolve().parent.parent
SCENARIOS_DIR = BASE_DIR / "mock_scenarios"
PACK_PATH = BASE_DIR / "packs" / "fa18c_startup" / "pack.yaml"
TAXONOMY_PATH = BASE_DIR / "packs" / "fa18c_startup" / "taxonomy.yaml"
TEMP_DIR = Path("tests/.tmp_logs")


def cleanup_path(path: Path) -> None:
    for _ in range(3):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            sleep(0.01)


def test_scoring_distinguishes_scenarios():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    log_good = run_simulation(
        pack_path=str(PACK_PATH),
        scenario_path=str(SCENARIOS_DIR / "correct_process.json"),
        log_path=str(TEMP_DIR / "test_good.jsonl"),
    )
    log_missing = run_simulation(
        pack_path=str(PACK_PATH),
        scenario_path=str(SCENARIOS_DIR / "missing_steps.json"),
        log_path=str(TEMP_DIR / "test_missing.jsonl"),
    )
    log_prem = run_simulation(
        pack_path=str(PACK_PATH),
        scenario_path=str(SCENARIOS_DIR / "premature_acceleration.json"),
        log_path=str(TEMP_DIR / "test_prem.jsonl"),
    )

    try:
        score_good = score_run(str(log_good), str(PACK_PATH), str(TAXONOMY_PATH))
        score_missing = score_run(str(log_missing), str(PACK_PATH), str(TAXONOMY_PATH))
        score_prem = score_run(str(log_prem), str(PACK_PATH), str(TAXONOMY_PATH))

        assert score_good["TotalErrorScore"] < score_missing["TotalErrorScore"]
        assert score_prem["TotalErrorScore"] > score_good["TotalErrorScore"]
        assert score_prem["Count_SV"] > 0
        assert score_missing["Count_OM"] > score_good["Count_OM"]
    finally:
        cleanup_path(Path(log_good))
        cleanup_path(Path(log_missing))
        cleanup_path(Path(log_prem))
