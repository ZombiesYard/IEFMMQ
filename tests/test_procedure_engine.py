from datetime import datetime, timedelta, timezone

import pytest

from core.procedure import ProcedureEngine, StepStatus


STEPS = [{"id": f"S{str(i).zfill(2)}"} for i in range(1, 4)]


def test_activate_and_complete_emits_events():
    engine = ProcedureEngine(STEPS)
    st1 = engine.activate_next()
    assert st1.step_id == "S01"
    assert engine.events[-1]["type"] == "step_activated"

    st1_done = engine.complete_active()
    assert st1_done.status == StepStatus.DONE
    assert engine.events[-1]["type"] == "step_completed"

    st2 = engine.activate_next()
    assert st2.step_id == "S02"


def test_rewind_moves_last_done_to_active():
    engine = ProcedureEngine(STEPS)
    engine.activate_next()
    engine.complete_active()  # S01 done
    engine.activate_next()  # S02 active
    engine.complete_active()  # S02 done

    st = engine.rewind()
    assert st.step_id == "S02"
    assert engine.active_step() == "S02"
    assert engine.status("S02") == StepStatus.ACTIVE


def test_repeated_prompts_count():
    engine = ProcedureEngine(STEPS)
    engine.activate_next()
    assert engine.request_prompt() == 1
    assert engine.request_prompt() == 2


def test_timeout_blocks_active_step():
    engine = ProcedureEngine(STEPS)
    engine.activate_next()
    # set activation time in past
    engine.set_activation_time("S01", datetime.now(timezone.utc) - timedelta(seconds=11))

    blocked = engine.check_timeout(datetime.now(timezone.utc), timeout_seconds=10)
    assert blocked is not None
    assert engine.status("S01") == StepStatus.BLOCKED
    assert engine.events[-1]["type"] == "step_blocked"


def test_block_active_with_reason():
    engine = ProcedureEngine(STEPS)
    engine.activate_next()
    engine.block_active(reason="instructor stop")
    assert engine.status("S01") == StepStatus.BLOCKED
    assert engine.events[-1]["reason"] == "instructor stop"


def test_activate_next_when_active_returns_current():
    engine = ProcedureEngine(STEPS)
    st1 = engine.activate_next()
    assert engine.events[-1]["type"] == "step_activated"
    event_count = len(engine.events)

    st1_again = engine.activate_next()
    assert st1_again.step_id == st1.step_id
    assert engine.active_step() == "S01"
    assert len(engine.events) == event_count


def test_complete_active_when_none_raises():
    engine = ProcedureEngine(STEPS)
    with pytest.raises(RuntimeError, match="No active step to complete"):
        engine.complete_active()


def test_block_active_when_none_raises():
    engine = ProcedureEngine(STEPS)
    with pytest.raises(RuntimeError, match="No active step to block"):
        engine.block_active(reason="timeout")


def test_rewind_with_no_done_steps_raises():
    engine = ProcedureEngine(STEPS)
    with pytest.raises(RuntimeError, match="No completed step to rewind to"):
        engine.rewind()


def test_activate_next_after_all_steps_done_raises():
    engine = ProcedureEngine(STEPS)
    for _ in STEPS:
        engine.activate_next()
        engine.complete_active()
    with pytest.raises(RuntimeError, match="No pending steps to activate"):
        engine.activate_next()


def test_request_prompt_invalid_step_id_raises_runtime_error():
    engine = ProcedureEngine(STEPS)
    with pytest.raises(RuntimeError, match="Unknown step_id 'S99' for prompt request"):
        engine.request_prompt(step_id="S99")


def test_status_invalid_step_id_raises_key_error():
    engine = ProcedureEngine(STEPS)
    with pytest.raises(KeyError):
        engine.status("S99")


def test_check_timeout_with_no_active_step_returns_none():
    engine = ProcedureEngine(STEPS)
    result = engine.check_timeout(datetime.now(timezone.utc), timeout_seconds=10)
    assert result is None

