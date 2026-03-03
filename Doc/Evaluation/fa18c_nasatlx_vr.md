# NASA Raw-TLX - Subjective Workload Questionnaire
## Task: F/A-18C Cold Start in DCS World (VR)

Please answer this questionnaire immediately after each trial.
All six items use the same 0-20 scale.

- 0 = very low / very good
- 10 = medium
- 20 = very high / very poor

You may use any integer in between.

---

### 1. Mental Demand

How mentally demanding was the task?

**Rating (0-20):** ______

### 2. Physical Demand

How physically demanding was the task?

**Rating (0-20):** ______

### 3. Temporal Demand

How hurried or rushed was the pace of the task?

**Rating (0-20):** ______

### 4. Performance

How successful do you think you were in accomplishing the task?

**Rating (0-20):** ______

### 5. Effort

How hard did you have to work (mentally and physically) to accomplish your performance?

**Rating (0-20):** ______

### 6. Frustration Level

How insecure, discouraged, irritated, stressed, and annoyed versus secure, content,
relaxed, and satisfied did you feel during the task?

**Rating (0-20):** ______

---

## Optional Comments

What made the trial easy or difficult (controls, sequence memory, display reading,
help card clarity, etc.)?

____________________________________________________________________

____________________________________________________________________

____________________________________________________________________

---

## Scoring Rule (for raters)

Use unweighted Raw-TLX:

```text
NASA_RawTLX = (Mental + Physical + Temporal + Performance + Effort + Frustration) / 6
```

Store both item-level scores and `NASA_RawTLX` in the trial summary sheet.
