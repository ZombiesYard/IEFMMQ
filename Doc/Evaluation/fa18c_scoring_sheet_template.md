# F/A-18C Scoring Sheet Template

This template defines two scoring tables:

- Trial-level summary (one row per participant x condition x trial)
- Step-level coding (one row per S01-S25 step per trial)

Use this template together with `fa18c_error_coding_guide.md`.

---

## 1. Trial-Level Summary

| ParticipantID | Condition | TrialID | Completed | DeadEnd_DE | HelpRequest_HR | TaskTime_sec | TimeToFirstHelp_sec | StuckEpisodeCount | TotalStuckDuration_sec | MaxSingleStuckDuration_sec | RecoveredFromStuck | TotalErrorScore | CriticalErrorScore | NonCriticalErrorScore | Count_OM | Count_CO | Count_OR | Count_PA | Count_SV | NASA_RawTLX | Notes |
|---------------|-----------|---------|-----------|------------|----------------|--------------|---------------------|-------------------|------------------------|----------------------------|--------------------|-----------------|--------------------|-----------------------|----------|----------|----------|----------|----------|-------------|-------|
| P01 | grounded_rag_llm | T01 | yes | no | no | 980 | 320 | 1 | 72 | 72 | yes | 12 | 9 | 3 | 1 | 0 | 2 | 1 | 1 | 41 | example |
| P02 | video_notes | T01 | no | yes | yes | 1200 | 610 | 3 | 310 | 150 | no | 27 | 21 | 6 | 3 | 1 | 4 | 1 | 3 | 63 | timeout |

### Field Rules

- `TaskTime_sec`: from first cockpit interaction to completion or timeout.
- `TimeToFirstHelp_sec`: blank if no help event; otherwise seconds from task start.
- `StuckEpisodeCount`: count episodes by stuck criterion (>=90 s no progress, or repeated ineffective loop).
- `TotalStuckDuration_sec`: sum of all stuck episode durations.
- `MaxSingleStuckDuration_sec`: max duration among stuck episodes.
- `RecoveredFromStuck`: `yes` if at least one stuck episode occurred and later recovered.
- `NASA_RawTLX`: average of six NASA Raw-TLX dimensions (0-20 scale each).

---

## 2. Step-Level Coding

| ParticipantID | Condition | TrialID | StepID | Phase | Critical | StepStart_sec | StepEnd_sec | Performed | Error_OM | Error_CO | Error_OR | Error_PA | Error_SV | StepErrorScore | PreconditionsSatisfied | Notes |
|---------------|-----------|---------|--------|-------|----------|---------------|-------------|-----------|----------|----------|----------|----------|----------|----------------|------------------------|-------|
| P01 | grounded_rag_llm | T01 | S01 | P1 | yes | 3 | 12 | yes | 0 | 0 | 0 | 0 | 0 | 0 | yes | |
| P01 | grounded_rag_llm | T01 | S05 | P2 | yes | 140 | 172 | yes | 0 | 0 | 1 | 0 | 1 | 6 | no | throttle advanced too early |
| P01 | grounded_rag_llm | T01 | S12 | P4 | yes | 420 | 447 | yes | 0 | 0 | 1 | 0 | 0 | 3 | yes | INS set before canonical order |

### Field Rules

- `StepStart_sec`, `StepEnd_sec`: seconds from trial start.
- `Performed`: `yes/no`.
- `Error_*`: binary (`0/1`) per category, max one per category per step.
- `PreconditionsSatisfied`: `yes/no/na` for state-gated steps.
- `StepErrorScore` formula:

```text
StepErrorScore = ceil((2*OM + 1*CO + 2*OR + 1*PA + 2*SV) * (1.5 if Critical else 1.0))
```

---

## 3. Derived Metrics (Recommended)

Compute from trial table:

1. Completion rate by condition.
2. Mean and median `TaskTime_sec` by condition.
3. Mean `TotalErrorScore` and `CriticalErrorScore` by condition.
4. Mean `StuckEpisodeCount` and `TotalStuckDuration_sec` by condition.
5. Help dependence: proportion with `HelpRequest_HR=yes` and distribution of `TimeToFirstHelp_sec`.
6. Correlation between `NASA_RawTLX` and error/stuck metrics.

These metrics map directly to thesis research questions on efficiency, correctness,
workload, and flow interruption.
