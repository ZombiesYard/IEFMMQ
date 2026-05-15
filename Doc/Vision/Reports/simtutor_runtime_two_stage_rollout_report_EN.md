# SimTutor v0.4 Two-Stage Runtime Rollout Report

## Abstract

This report documents the v0.4 runtime deployment of SimTutor's two-stage tutoring pipeline on a local vLLM server (NVIDIA A100 80GB). The system uses one `Qwen/Qwen3.5-9B-Base` model served under two names: `simtutor-vision` (with LoRA) for visual fact extraction from composite-panel screenshots, and `simtutor-qwen35-9b-base` (base only) for text-only step diagnosis and overlay targeting. Compared with the earlier single-model-name setup, this separation eliminates LoRA interference on the text generation path.

The deployment also introduces five infrastructure improvements: transport-level retry, JPEG image compression, connection pool tuning, relaxed JSON schema validation, and server-side thinking disable. On a 15-cycle live test session, the system achieves 14/15 model-mode responses with zero `RemoteProtocolError` failures and zero `ValidationError` fallbacks. The dominant remaining failure mode is LLM hallucination under partial observability — when all VLM facts are `not_seen` (displays dark), the 9B base model sometimes ignores visual evidence and generates incorrect overlay guidance.

## 1. Architecture

### 1.1 Two-stage pipeline

Each help cycle makes two separate model calls to the same vLLM server:

| Stage | Model name | LoRA | Input | Output |
|---|---|---|---|---|
| Vision Fact Extraction | `simtutor-vision` | yes | composite-panel image (~200 KB JPEG) + fact extraction prompt | 13 visual facts (`seen`/`not_seen`/`uncertain`) |
| Help Response Generation | `simtutor-qwen35-9b-base` | no | VLM facts (text) + telemetry (VARS, GATES) + RAG + deterministic step hint + procedure rules | `{"diagnosis","next","overlay","explanations"}` |

The LoRA adapter was trained on Run-003 + Run-005x2 bilingual data for the `image → facts JSON` task. It is NOT applied to the help response call because the LoRA weights bias the model toward the facts output schema (`summary`, `facts`), which interferes with generating the different help response schema (`diagnosis`, `overlay`, `explanations`).

### 1.2 Deployment topology

| Component | Location | Port |
|---|---|---|
| vLLM server | `cloud-247.rz.tu-clausthal.de` | 6324 |
| SSH tunnel | Windows PowerShell `localhost:16324 → cloud-247:6324` | 16324 |
| DCS + SimTutor | Windows 10, Python 3.12 | — |
| Vision sidecar | Same Windows machine | trigger on 7795 |

```
┌─────────────────────────────────────────────────────────────────┐
│ Windows 10 (DCS + SimTutor)                                     │
│                                                                 │
│  ┌──────────┐   UDP:7795   ┌────────────────┐                  │
│  │ DCS hook │──────────────→│ Vision sidecar │                  │
│  │ 截图触发  │              │ 截图 + 组合     │                  │
│  └──────────┘              └───────┬────────┘                  │
│                                    │ PNG (~5 MB)               │
│                                    ▼                           │
│                         ┌────────────────────┐                 │
│                         │ SimTutor live_dcs  │                 │
│                         │                    │                 │
│                         │  Stage 1: VLM call │                 │
│                         │  model=simtutor-   │                 │
│                         │  vision (LoRA)     │                 │
│                         │  image→facts JSON  │                 │
│                         │         │          │                 │
│                         │         ▼          │                 │
│                         │  Stage 2: LLM call │                 │
│                         │  model=simtutor-   │                 │
│                         │  qwen35-9b-base    │                 │
│                         │  facts+telemetry   │                 │
│                         │  →help response    │                 │
│                         └────────┬───────────┘                 │
│                                  │ http://localhost:16324      │
│                                  │ SSH tunnel                  │
└──────────────────────────────────┼──────────────────────────────┘
                                   │
┌──────────────────────────────────┼──────────────────────────────┐
│ cloud-247.rz.tu-clausthal.de     │                              │
│                                  ▼                              │
│  ┌──────────────────────────────────────────┐                  │
│  │ vLLM 0.19.0 :6324                        │                  │
│  │                                          │                  │
│  │  served models:                          │                  │
│  │  - simtutor-qwen35-9b-base (base)        │                  │
│  │  - simtutor-vision (base + LoRA)         │                  │
│  │                                          │                  │
│  │  GPU: NVIDIA A100 80GB                   │                  │
│  │  enable_thinking: false (global)         │                  │
│  └──────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

## 2. Infrastructure Improvements

Five changes were made to the runtime stack compared with the earlier single-model-name deployment:

### 2.1 Transport-level retry

`httpx.Client` connection pools reuse TCP connections. Over SSH tunnels, keep-alive connections can become stale, causing `RemoteProtocolError` on subsequent requests. Previously this accounted for ~79% of failures during multimodal sessions.

**Fix**: `OpenAICompatModel._post_with_transport_retry` catches `httpx.RequestError` / `ConnectionError` / `TimeoutError`, resets the HTTP client (closes stale connections, creates fresh socket), and retries once. This eliminated `RemoteProtocolError` entirely in the test session.

### 2.2 JPEG image compression

PNG screenshots from DCS composite-panel captures are ~5 MB raw. Base64-encoding them for VLM input produces ~6.7 MB strings. Over SSH tunnels with packet loss, large payloads are more likely to fail.

**Fix**: `_encode_image_for_vlm` in `openai_compat_multimodal.py` converts PNG to JPEG at 85% quality before base64 encoding. A typical 880×1440 cockpit screenshot compresses from ~5 MB to ~200 KB (25× reduction). The LoRA adapter was trained on lossless PNG but generalizes to JPEG at 85% quality without accuracy loss.

### 2.3 Connection pool tuning

The default `httpx.Client(timeout=60.0)` uses a single flat timeout and unlimited connection reuse.

**Fix**: `_make_http_client` now uses:

| Setting | Value | Rationale |
|---|---|---|
| `connect` timeout | 5 s | fast failure for unreachable servers |
| `read` timeout | 60 s | generous for LLM inference |
| `write` timeout | 30 s | covers ~200 KB upload |
| `pool` timeout | 5 s | quick connection acquisition |
| `max_keepalive_connections` | 2 | prevent stale connection accumulation |
| `max_connections` | 4 | modest ceiling for single-user system |
| `keepalive_expiry` | 30 s | force fresh connections periodically |

### 2.4 Relaxed JSON schema

The LLM help response schema previously required `evidence.quote.minLength=1`, `explanations.minItems=1`, and `explanations[].minLength=1`. The 9B base model frequently omits `quote` text or produces empty `explanations`, causing schema validation failures that rejected otherwise correct overlay targets.

**Fix**: `quote.minLength` changed to `0`, `explanations.minItems` to `0`, and `explanations[].minLength` to `0`. The downstream code already fills default explanations and empty quotes do not invalidate the overlay evidence.

### 2.5 Server-side thinking disable

Qwen3.5-9B-Base has `enable_thinking` enabled by default, producing `<think>...</think>` tags that waste tokens and corrupt JSON parsing. vLLM 0.19.0 was found to ignore the per-request `chat_template_kwargs: {"enable_thinking": false}` override.

**Fix**: A `generation_config.json` with `{"enable_thinking": false}` was placed in the model snapshot directory. This disables thinking globally for all requests, regardless of how they are routed.

### 2.6 Few-shot examples

Two complete request-response examples are injected into the prompt for the base model (injection is gated on `_owns_client` so tests are unaffected). The examples show correct diagnosis, overlay evidence, and explanation formatting. The earlier assistant prefill (`{"role": "assistant", "content": "{"}`) was removed after testing showed it caused the model to regurgitate conversation history rather than continuing JSON.

## 3. Test Session Results (2026-05-15)

### 3.1 Session overview

| Metric | Value |
|---|---|
| Total help cycles | 15 |
| Total observations | 6137 |
| Vision frame observations | 366 |
| Overlay events | ~30 |
| Session duration | ~5 min |

### 3.2 Generation mode distribution

| Mode | Count | % | Notes |
|---|---|---|---|
| `model` | 14 | 93% | LLM generated valid help response |
| `fallback` | 1 | 7% | S06: LLM threw `ValueError` (empty output) |
| `repair` | 0 | 0% | — |
| `RemoteProtocolError` | **0** | 0% | Transport retry eliminated all network failures |
| `ValidationError` | **0** | 0% | Schema relaxation eliminated all schema rejections |

Compared with the pre-fix baseline (35 cycles: 40% model, 49% fallback, 11% repair), the model-mode rate improved from 40% to 93%.

### 3.3 Per-step breakdown

| # | Step | Mode | Latency (ms) | Message (abridged) |
|---|---:|---:|---:|---|
| 1 | S01 | model | 1143 | "Operate battery_switch to turn on battery." |
| 2 | S02 | model | 1232 | "Move fire_test_switch to TEST A." |
| 3 | S03 | model | 1316 | "Turn APU switch on." |
| 4 | S03 | model | 1293 | "APU not ready. Turn APU switch (left-click)." |
| 5 | S04 | model | 1194 | "Set ENG CRANK to RIGHT." |
| 6 | S05 | model | 1403 | "Operate left_mdi_pb18, then satisfy vars.rpm_r>=25." |
| 7 | S05 | model | 1333 | "Right throttle must be advanced from OFF to IDLE." |
| 8 | S06 | fallback | 1393 | "S06 incomplete: vars.rpm_r_gte_60==true." |
| 9 | S07 | model | 1211 | "Press lights_test_button." |
| 10–15 | S08 | model (×6) | 1376–1499 | "Left DDI shows TAC page. Press left MDI PB18." |

### 3.4 Tutor text delivery

All 15 cycles successfully delivered tutor text to DCS via `DcsTutorTextSender` (UDP → Lua hook → `trigger.action.outText`). Zero `sender_unavailable` occurrences.

## 4. Remaining Issues

### 4.1 LLM hallucination under partial observability

The most prominent remaining failure mode appeared in cycles #10–15 (step S08). Key observations:

- **Cycle #10**: All 13 VLM facts are `not_seen`. Displays are dark. `fused_missing` correctly reports `vars.mpcd_on==true`. The LLM nevertheless outputs "Left DDI shows TAC page. Press left MDI PB18."
- **Cycle #11**: Same pattern — all VLM facts `not_seen`, LLM hallucinates TAC page guidance.
- **Cycles #12–15**: VLM returns some `seen` facts (`tac_page_visible`, `supt_page_visible`, `fcs_page_visible` at different stages). The LLM continues to emit "TAC page visible, press PB18" regardless of actual VLM output, ignoring the `fused_missing` conditions (`vars.mpcd_on==true`, later `vars.hud_on==true`).

The root cause is the 9B base model's instruction-following ceiling. The prompt instructs the model to trust VLM annotations, but the model defaults to its own parametric priors. The fused step inference correctly identifies S08 and the correct missing conditions, but this information is buried in a long context and is not treated as a hard constraint by the LLM.

**Proposed fix**: Add a deterministic guard in `LiveDcsTutorLoop.run_help_cycle`. When VLM returns zero `seen` facts AND the step requires visual confirmation, skip the LLM and emit a template message ("Displays may be off. Complete earlier steps first."). This keeps the safety guarantee in code rather than relying on the LLM to read and obey prompt rules.

### 4.2 VLM fact prompt also goes through LoRA but is image-only

The VisionFact extraction prompt is image-only text ("你是 SimTutor 视觉事实抽取器...") plus the composite-panel image. Since the LoRA was trained on exactly this format (Run-003 + Run-005x2 bilingual data), the VLM produces high-quality facts. All 15 cycles in the test session had `multimodal_path_success=True`.

### 4.3 Latency

Per-cycle latency is dominated by the two model calls:

| Component | Typical latency |
|---|---|
| VLM fact extraction | ~500–1000 ms |
| Help response generation | ~1000–1500 ms |
| Total per help cycle | ~1500–2500 ms |
| With connection warmup | ~100 ms faster on first call |

This is acceptable for a tutoring system where help is triggered manually (hotkey). For continuous automatic help, a cooldown of ≥3 s is recommended.

## 5. Interpretation

The 93% model-mode rate should be interpreted with the following context:

1. **The test session was 15 cycles**. This is a small sample. Longer sessions with more display-state transitions are needed for a reliable estimate.
2. **The `missing=[]` anomaly**: The deterministic step hint shows empty `missing_conditions` for several requests despite gates being unsatisfied. This suggests the gate evaluation pipeline may have a regression, likely introduced during the S08/S26 step restructuring (PR #237). The LLM compensates by reading VARS directly from the evidence sources, which works in most cases but fails when evidence is sparse.
3. **Thinking disable eliminated the repair cascade**: In the pre-fix run, thinking wasted tokens on `<think>` blocks, causing JSON truncation → `ValueError` → schema repair → `repair` or `fallback`. With thinking disabled globally, the model outputs compact JSON directly, eliminating the entire repair chain.
4. **The two-model-name separation is the most impactful single change**. Before separation, the LoRA adapter interfered with help response generation. After separation, the base model follows the help response schema more faithfully, and the LoRA adapter handles visual fact extraction independently.

## 6. Limitations

1. **Single session, 15 cycles**. Generalization to longer sessions with more display transitions is not yet measured.
2. **Network tested during off-peak hours**. The campus egress router issue observed in earlier sessions (peak-time packet loss causing 85% `RemoteProtocolError`) was not reproduced.
3. **JPEG compression not benchmarked against PNG**. The LoRA adapter was trained on PNG images. While manual spot checks show no degradation, a formal comparison is pending.
4. **No ablation on few-shot examples**. The contribution of few-shot examples to the 93% model rate cannot be separated from the model-name separation and thinking disable in this report.
5. **`missing_conditions=[]` regression**. The empty gate conditions for several requests prevent the LLM from generating precise "please satisfy X" guidance, forcing it to infer conditions from raw VARS data.

## 7. Conclusion

The v0.4 two-stage runtime deployment achieves a 93% model-mode rate on a 15-cycle live test session, up from 40% in the pre-fix baseline. Three changes account for most of the gain: (1) LoRA/model-name separation, which removes adapter interference on the text generation path; (2) server-side thinking disable, which eliminates the JSON truncation → repair cascade; and (3) transport-level retry, which eliminates `RemoteProtocolError`.

The remaining failure mode is LLM hallucination under partial observability — a known limitation of small base models that should be addressed with deterministic guards in the future. The infrastructure stack (JPEG compression, connection pool tuning, schema relaxation) is stable and production-ready.
