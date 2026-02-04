from datetime import datetime, timedelta, timezone

from core.interaction_metrics import compute_interaction_metrics


def _iso(dt):
    return dt.isoformat()


def test_interaction_metrics_counts_and_stall_time():
    t0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(seconds=5)
    events = [
        {"kind": "step_activated", "payload": {"step_id": "S01", "timestamp": _iso(t0)}},
        {"kind": "tutor_request", "payload": {"intent": "ask_hint"}},
        {"kind": "tutor_response", "payload": {"metadata": {"model": "stub"}}},
        {"kind": "ui.highlight_requested", "payload": {"element_id": "pnt_1"}},
        {"kind": "ui.highlight_ack", "payload": {"element_id": "pnt_1"}},
        {"kind": "ui.user_clicked", "payload": {"element_id": "pnt_1", "success": True}},
        {"kind": "step_completed", "payload": {"step_id": "S01", "timestamp": _iso(t1)}},
    ]
    metrics = compute_interaction_metrics(events)
    assert metrics.help_requests == 1
    assert metrics.llm_triggers == 1
    assert metrics.ui_clicks == 1
    assert metrics.ui_click_failures == 0
    assert metrics.highlight_requests == 1
    assert metrics.highlight_acks == 1
    assert metrics.stall_time_total_s == 5.0
