# F/A-18C Scoring Sheet Template

This template defines two levels of scoring data:

- **Trial-level summary**: one row per participant × condition × trial.
- **Step-level coding**: one row per canonical step (S01–S25) per trial.

---

## 1. Trial-Level Summary

| ParticipantID | Condition            | TrialID | Completed | DeadEnd_DE | HelpRequest_HR | TaskTime_sec | TotalErrorScore | CriticalErrorScore | NonCriticalErrorScore | Count_OM | Count_CO | Count_OR | Count_PA | Count_SV | Comments |
|---------------|----------------------|---------|-----------|------------|----------------|--------------|-----------------|--------------------|-----------------------|----------|----------|----------|----------|----------|----------|
| P01           | grounded_rag_llm     | T01     | yes       | no         | no             | 1234         | 18              | 12                 | 6                     | 2        | 1        | 3        | 2        | 2        | example only |
| P02           | video_notes          | T01     | no        | yes        | yes            | 1500         | 26              | 20                 | 6                     | 3        | 2        | 4        | 1        | 3        | example only |
| …             | …                    | …       | …         | …          | …              | …            | …               | …                  | …                     | …        | …        | …        | …        | …        | …        |

**Field hints**

- `ParticipantID`: e.g., `P01`, `P02`, …
- `Condition`: e.g., `video_notes`, `llm_ungrounded`, `grounded_rag_llm`.
- `TrialID`: e.g., `T01` (if each participant runs multiple trials).
- `Completed`: `yes` / `no` (taxi-ready criteria met?).
- `DeadEnd_DE`: `yes` / `no` (coded from step-level + observation).
- `HelpRequest_HR`: `yes` / `no`.
- `TaskTime_sec`: time from first cockpit interaction to end condition.
- `TotalErrorScore`: sum of all `StepErrorScore` for the trial.
- `CriticalErrorScore`: sum of `StepErrorScore` on steps with `Critical = yes`.
- `NonCriticalErrorScore`: sum of `StepErrorScore` on steps with `Critical = no`.
- `Count_OM`, `Count_CO`, `Count_OR`, `Count_PA`, `Count_SV`: total counts per error category.
- `Comments`: free-text notes for anomalies, interruptions, etc.

---

## 2. Step-Level Coding

| ParticipantID | Condition        | TrialID | StepID | Phase | Critical | Performed | Error_OM | Error_CO | Error_OR | Error_PA | Error_SV | StepErrorScore | Notes                    |
|---------------|------------------|---------|--------|-------|----------|-----------|----------|----------|----------|----------|----------|----------------|--------------------------|
| P01           | grounded_rag_llm | T01     | S01    | P1    | yes      | yes       | 0        | 0        | 0        | 0        | 0        | 0              |                          |
| P01           | grounded_rag_llm | T01     | S02    | P1    | yes      | no        | 1        | 0        | 0        | 0        | 0        | 3              | omission of fire test    |
| P01           | grounded_rag_llm | T01     | S05    | P2    | yes      | yes       | 0        | 0        | 1        | 0        | 1        | 6              | throttle too early       |
| P01           | grounded_rag_llm | T01     | S08    | P3    | no       | yes       | 0        | 0        | 0        | 0        | 0        | 0              |                          |
| P01           | grounded_rag_llm | T01     | S12    | P4    | yes      | yes       | 0        | 0        | 1        | 0        | 0        | 3              | INS set too early        |
| …             | …                | …       | …      | …     | …        | …         | …        | …        | …        | …        | …        | …              | …                        |

**Field hints**

- `StepID`: `S01`–`S25` as defined in `fa18c_startup_master.md`.
- `Phase`: `P1`–`P6` (Power-up, Right Engine, Displays, Left Engine/Core, FCS, Pre-taxi).
- `Critical`: `yes` / `no` (from step table).
- `Performed`: `yes` / `no` (did the participant attempt this step at all?).
- `Error_OM`, `Error_CO`, `Error_OR`, `Error_PA`, `Error_SV`: `0` or `1` for each category.
- `StepErrorScore`: numeric value computed from error codes and criticality, e.g.:  
  `StepErrorScore = (2*OM + 1*CO + 2*OR + 1*PA + 2*SV) * (Critical ? 1.5 : 1)`
- `Notes`: short comments, e.g. “INS mode set before engine stable”.

---
