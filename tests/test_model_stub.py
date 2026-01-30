from adapters.model_stub import ModelStub
from core.types import Observation


def test_stub_mode_a_sets_rule_card():
    obs = Observation(procedure_hint="S05")
    stub = ModelStub(mode="A")
    res = stub.plan_next_step(obs)
    assert "[RULE]" in res.metadata["card"]
    assert "S05" in res.message


def test_stub_mode_c_adds_rag_note():
    obs = Observation(procedure_hint="S06")
    stub = ModelStub(mode="C")
    res = stub.explain_error(obs)
    assert "RAG" in res.metadata["card"]
    assert "S06" in res.message


def test_stub_mode_b_sets_llm_card():
    obs = Observation(procedure_hint="S07")
    stub = ModelStub(mode="B")
    res = stub.plan_next_step(obs)
    assert "[LLM-STUB]" in res.metadata["card"]
    assert "S07" in res.message

