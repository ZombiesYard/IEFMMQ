from __future__ import annotations

from tools.train_qwen35_vlm_unsloth import _select_train_rows


def test_select_train_rows_shuffles_before_truncating_prefix_ordered_data() -> None:
    rows = [{"sample_id": f"row-{index}"} for index in range(10)]

    selected = _select_train_rows(rows, max_train_samples=4, seed=3407)

    assert len(selected) == 4
    assert selected != rows[:4]
    assert selected == _select_train_rows(rows, max_train_samples=4, seed=3407)


def test_select_train_rows_keeps_full_dataset_order_when_limit_is_disabled_or_non_truncating() -> None:
    rows = [{"sample_id": f"row-{index}"} for index in range(4)]

    assert _select_train_rows(rows, max_train_samples=0, seed=3407) == rows
    assert _select_train_rows(rows, max_train_samples=4, seed=3407) == rows
    assert _select_train_rows(rows, max_train_samples=99, seed=3407) == rows
