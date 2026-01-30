from pathlib import Path
from time import sleep

from simtutor.runner import batch_run


BASE_DIR = Path(__file__).resolve().parent.parent
SCENARIO = BASE_DIR / "mock_scenarios" / "correct_process.json"
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


def test_batch_run_creates_unique_logs_and_fields():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    results = batch_run(
        pack_path=str(PACK_PATH),
        scenarios=[str(SCENARIO), str(SCENARIO)],
        output_dir=str(TEMP_DIR),
        taxonomy_path=str(TAXONOMY_PATH),
    )
    try:
        assert len(results) == 2
        log_paths = [Path(r["log_path"]) for r in results]
        assert log_paths[0] != log_paths[1]
        assert all(p.exists() for p in log_paths)
        assert all(r["scenario"] == SCENARIO.stem for r in results)
        assert all("TotalErrorScore" in r for r in results)
    finally:
        for r in results:
            cleanup_path(Path(r["log_path"]))
