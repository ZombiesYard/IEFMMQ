# FA-18C Stable Vars (S01-S25)

This document records the startup vars resolved from `telemetry_map.yaml`,
their DCS-BIOS source keys, and operating-range assumptions used by SimTutor.

## Numeric Vars

| Var | Source BIOS key | Transform | Assumed range / meaning |
| --- | --- | --- | --- |
| `rpm_r` | `IFEI_RPM_R` | `num(...)` | Right engine RPM percent (0-110) |
| `rpm_l` | `IFEI_RPM_L` | `num(...)` | Left engine RPM percent (0-110) |
| `temp_r` | `IFEI_TEMP_R` | `num(...)` | Right engine temp degC (0-1200) |
| `temp_l` | `IFEI_TEMP_L` | `num(...)` | Left engine temp degC (0-1200) |
| `ff_r` | `IFEI_FF_R` | `num(...) * 100` | Right fuel flow PPH (IFEI shows x100, so display `7` => `700` PPH) |
| `ff_l` | `IFEI_FF_L` | `num(...) * 100` | Left fuel flow PPH (IFEI shows x100, so display `7` => `700` PPH) |
| `oil_r` | `IFEI_OIL_PRESS_R` | `num(...)` | Right oil pressure psi (0-200) |
| `oil_l` | `IFEI_OIL_PRESS_L` | `num(...)` | Left oil pressure psi (0-200) |
| `noz_r` | `EXT_NOZZLE_POS_R` | `(num(...) * 100) / 65535` | Right nozzle percent (0-100) |
| `noz_l` | `EXT_NOZZLE_POS_L` | `(num(...) * 100) / 65535` | Left nozzle percent (0-100) |

## Derived Bool Vars

| Var | Logic | Checklist intent |
| --- | --- | --- |
| `rpm_r_gte_25` | `rpm_r >= 25` | S05 throttle-right OFF->IDLE gate |
| `rpm_r_gte_60` | `rpm_r >= 60` | S06 BLEED AIR cycle precondition |
| `rpm_l_gte_60` | `rpm_l >= 60` | Future left-engine stabilization check |
| `rpm_r_in_range` | `63 <= rpm_r <= 70` | S10 nominal right-engine window |
| `temp_r_in_range` | `190 <= temp_r <= 590` | S10 nominal right-engine window |
| `ff_r_in_range` | `420 <= ff_r <= 900` | S10 nominal right-engine window |
| `oil_r_in_range` | `45 <= oil_r <= 110` | S10 nominal right-engine window |
| `noz_r_in_range` | `73 <= noz_r <= 84` | S10 nominal right-engine window |
| `right_engine_nominal_start_params` | all `_in_range` above are true | S10 precondition summary flag |

## Missing-source helper

- `vars_source_missing`: a list of var names whose source data is missing/blank or
  whose resolved value is `None` in current frame.
- Intended use: prompt `missing_conditions` and deterministic fallback when data is
  unavailable (avoid LLM guessing from raw/incomplete telemetry).

## Tri-state unknown convention (S14-S25)

- For checklist items that are not yet reliably observable from trusted telemetry,
  resolver emits string `"unknown"` instead of coercing to `false`.
- Consumers must treat these vars as tri-state (`true` / `false` / `unknown`) and
  should not rely on generic truthiness.
- Gate/inference paths interpret `"unknown"` as unmet/unknown evidence, enabling
  explicit degrade behavior instead of false-positive pass/fail.
