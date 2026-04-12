# LoRA Fine-Tuning Qwen3.5-9B VLM for SimTutor Cockpit Visual Fact Extraction

## Abstract

This report studies a task-oriented multimodal learning problem: whether a small set of human-reviewed cockpit screenshots is sufficient to adapt `Qwen/Qwen3.5-9B-Base` into a reliable structured visual fact extractor for the F/A-18C cold-start procedure. The experiment uses a single composite cockpit-panel image as input and trains the model to produce JSON labels for eight core visual facts. The fine-tuning method is Unsloth 4-bit LoRA VLM SFT. The training set is derived from 180 human-reviewed Run-001 images and exported into English and Chinese SFT variants, yielding 360 training examples. Evaluation is performed in two stages: Run-001 as a contaminated development set and Run-002 as a heldout new session.

The results show that LoRA-v1 substantially improves structured output and most cockpit visual facts. On Run-002, fact accuracy improves from `0.7600` to `0.9150`, and seen F1 improves from `0.4714` to `0.8380`. However, `ins_go` produces 13 false positives on Run-002, revealing that the current target ontology compresses a higher-level procedural state into a single-frame visual fact. The experiment therefore supports the hypothesis that small human-reviewed domain data can significantly improve structured VLM extraction, while also showing that some targets must be redesigned into lower-level, directly observable visual evidence.

## 1. Domain Primer: What Are Cockpit Visual Facts?


The F/A-18C is the simulated aircraft cockpit used in this project. A cold-start procedure is a sequence of actions and checks that bring the aircraft from an unprepared state into an operational state. The cockpit contains several multifunction displays. This experiment focuses on three fixed visual regions:

| Region | Meaning | Role in the image |
|---|---|---|
| `left_ddi` | Left Digital Display Indicator | Shows system pages, BIT pages, FCS pages, etc. |
| `ampcd` | Advanced Multipurpose Color Display | Often shows maps, INS alignment pages, and navigation data |
| `right_ddi` | Right Digital Display Indicator | Shows test state, FCS-MC pages, and BIT status |

In this report, a `fact` means a structured visual proposition extracted from a single cockpit panel image. It is not a natural-language answer and not a final task decision. Instead, it is an intermediate representation between visual perception and downstream task/state reasoning. Each fact has one of three labels:

| State | Meaning |
|---|---|
| `seen` | The visual fact is clearly visible in the current image |
| `not_seen` | The visual fact is not visible in the current image |
| `uncertain` | The image is blurry, occluded, incomplete, or insufficient for a decision |

The first experiment uses eight core facts:

| fact_id | Plain-language meaning | Why it matters |
|---|---|---|
| `fcs_page_visible` | Whether the flight-control-system FCS page is visible | Indicates that the flight-control page is available for inspection |
| `bit_root_page_visible` | Whether the built-in-test BIT root page is visible | Indicates proximity to a BIT menu or test entry state |
| `bit_page_failure_visible` | Whether a BIT failure list page is visible | Indicates visible system failure/status entries |
| `right_ddi_fcsmc_page_visible` | Whether the right DDI shows an FCS-MC related page | Indicates that the flight-control computer test page is visible |
| `right_ddi_in_test_visible` | Whether the right DDI clearly shows an in-test state | Indicates that a test is still running |
| `fcs_bit_result_visible` | Whether the final FCS BIT result is visible | Indicates that the FCS BIT result can be read |
| `ins_alignment_page_visible` | Whether the AMPCD shows the INS alignment page | Indicates that the inertial navigation alignment process is visible |
| `ins_go` | Whether the INS has reached a visible GO state | Indicates a visible completion signal for INS alignment |

These facts correspond to visual checkpoints in the cold-start procedure. A downstream reasoning module can use them to decide whether a procedure state has been observed, whether the system should wait, or whether additional evidence is needed. Importantly, these eight facts are a first-pass engineering abstraction rather than a final visual ontology.

## 2. System Architecture

The SimTutor visual pipeline is layered:

```text
DCS viewport screenshot
  -> VLM-ready composite panel image
  -> VLM visual fact extraction
  -> downstream task/state reasoning
```

The first stage captures and renders the DCS viewport into a VLM-ready artifact. The second stage uses a VLM to extract structured visual facts. The third stage combines these facts with procedure context and telemetry for task/state reasoning.

The experiment does not train the VLM to directly produce a tutor answer or a procedural recommendation. Direct natural-language answers are harder to evaluate, harder to debug, and less suitable for replay analysis. Structured facts are easier to review, measure, and compare. They also preserve a clean separation: the VLM provides visual evidence, while downstream reasoning combines that evidence with time, procedure state, and telemetry.

The input is a single composite-panel image rather than three separate cropped images. This keeps training, pre-labeling, evaluation, and later system usage on the same input distribution. It avoids a train/evaluation mismatch where the model is trained on three region crops but evaluated on a single composite image. The prompt explicitly states that the fixed top-to-bottom regions are `left_ddi`, `ampcd`, and `right_ddi`, but the model still receives one image.

## 3. Dataset Construction

The dataset pipeline is:

```text
DCS screenshot capture
  -> VLM-ready artifact rendering
  -> initial Qwen 397B VLM pre-labeling
  -> Label Studio human review
  -> reviewed JSONL
  -> OpenAI-compatible multimodal chat SFT JSONL
```

Run-001 is derived from `fa18c-coldstart-run-001` and contains 180 human-reviewed images. It is exported into English and Chinese SFT files, 180 examples per language and 360 examples total. LoRA-v1 is trained on these 360 examples, with 10% held out as an internal eval split. Therefore, Run-001 benchmark results are labeled as a contaminated development set and are used for regression analysis rather than independent generalization claims.

Run-002 is derived from `fa18c-coldstart-run-002` and contains 50 human-reviewed images. It is not used in LoRA-v1 training and is used as a heldout new-session benchmark. Run-002 includes cockpit screens that differ from Run-001 and therefore better exposes generalization and ontology issues.

### 3.1 Run-001 Label Distribution

| fact_id | seen | not_seen | uncertain |
|---|---:|---:|---:|
| `fcs_page_visible` | 146 | 34 | 0 |
| `bit_root_page_visible` | 110 | 70 | 0 |
| `bit_page_failure_visible` | 110 | 69 | 1 |
| `right_ddi_fcsmc_page_visible` | 44 | 136 | 0 |
| `right_ddi_in_test_visible` | 27 | 153 | 0 |
| `fcs_bit_result_visible` | 20 | 159 | 1 |
| `ins_alignment_page_visible` | 113 | 67 | 0 |
| `ins_go` | 18 | 160 | 2 |

### 3.2 Run-002 Label Distribution

| fact_id | seen | not_seen | uncertain |
|---|---:|---:|---:|
| `fcs_page_visible` | 13 | 37 | 0 |
| `bit_root_page_visible` | 9 | 41 | 0 |
| `bit_page_failure_visible` | 9 | 41 | 0 |
| `right_ddi_fcsmc_page_visible` | 9 | 41 | 0 |
| `right_ddi_in_test_visible` | 6 | 44 | 0 |
| `fcs_bit_result_visible` | 3 | 47 | 0 |
| `ins_alignment_page_visible` | 43 | 7 | 0 |
| `ins_go` | 4 | 46 | 0 |

Both runs are clearly imbalanced. `fcs_bit_result_visible` and `ins_go` have very few positive examples, while also being semantically important. This makes it difficult for the model to achieve both high recall and low false positives.

## 4. Bilingual SFT Dataset Format

The SFT data uses OpenAI-compatible multimodal chat JSONL. Each line is a complete sample with a system message, a user message, and an assistant message. The user message contains both the image and the textual instruction. The assistant message is the supervised JSON target.

### 4.1 English Sample Shape

The example below preserves the real structure while truncating the base64 image:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are SimTutor visual fact extractor. Reply with JSON only."
    },
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABEQAAAXo..."
          }
        },
        {
          "type": "text",
          "text": "You are the SimTutor visual fact extractor for the F/A-18C cold-start dataset. The input is exactly one composite-panel image. Its fixed top-to-bottom regions are: left_ddi, ampcd, right_ddi. Inspect only this image and output visual fact labels for the 8 core facts..."
        }
      ]
    },
    {
      "role": "assistant",
      "content": "{\"facts\":[{\"fact_id\":\"fcs_page_visible\",\"state\":\"not_seen\",\"evidence_note\":\"Left DDI screen is blank; no FCS page content is visible.\"},{\"fact_id\":\"bit_root_page_visible\",\"state\":\"not_seen\",\"evidence_note\":\"No BIT root menu is visible on any display.\"}],\"summary\":\"All three displays are powered on but show blank black screens with no visible data or pages.\"}"
    }
  ]
}
```

### 4.2 Chinese Sample Shape

The Chinese version uses the same image and the same fact labels, but the system and user instructions are in Chinese:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "你是 SimTutor 的视觉事实抽取器。只能输出 JSON。"
    },
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABEQAAAXo..."
          }
        },
        {
          "type": "text",
          "text": "你是 SimTutor 的视觉事实抽取器，负责给 F/A-18C 冷启动数据集做视觉事实标注。输入只有一张组合面板图。这张图内部的固定区域从上到下依次是：left_ddi、ampcd、right_ddi..."
        }
      ]
    },
    {
      "role": "assistant",
      "content": "{\"facts\":[{\"fact_id\":\"fcs_page_visible\",\"state\":\"not_seen\",\"evidence_note\":\"Left DDI screen is blank; no FCS page content is visible.\"}],\"summary\":\"All three displays are powered on but show blank black screens with no visible data or pages.\"}"
    }
  ]
}
```

This format is chosen for several reasons. It aligns with Qwen/OpenAI-compatible multimodal chat APIs and with the message format expected by Unsloth VLM SFT. It also keeps the learning target clear: image plus instruction should produce a structured JSON object. The assistant target contains only `summary` and `facts`. Each fact contains only `fact_id`, `state`, and `evidence_note`.

The training target intentionally excludes `frame_id`, `session_id`, `artifact_image_path`, `raw_image_path`, `source_frame_id`, and `confidence`. These fields are either dataset management metadata or external identifiers produced by the capture/runtime framework; they should not be generated from image content. `confidence` is also excluded because self-reported model confidence is not calibrated and can create a misleading sense of reliability. The VLM is trained to output auditable states and evidence notes, not pseudo-probabilities.

Bilingual SFT is used to increase instruction diversity rather than visual diversity. English and Chinese examples share the same image and fact labels. This can improve JSON-format robustness under both languages, but it cannot replace additional cockpit-state screenshots or hard-negative visual data.

## 5. Fine-Tuning Details

The base model is `Qwen/Qwen3.5-9B-Base`. Fine-tuning uses Unsloth 4-bit LoRA VLM SFT. The LoRA adapter is stored at:

```text
/scratch/yz50/iefmmq_vlm_ft_unsloth/runs/full_qwen35_9b_base_bilingual_v1/adapter
```

The training set consists of the Run-001 bilingual SFT data: 360 total examples, split into 324 train examples and 36 eval examples. Key training parameters are:

| Parameter | Value | Rationale |
|---|---:|---|
| `max_seq_length` | 4096 | Covers the single-image instruction, 8 facts, and JSON output while controlling memory |
| `num_train_epochs` | 4 | Small data requires repeated exposure to the JSON schema and cockpit fact boundaries; final eval loss remains low |
| `learning_rate` | 2e-4 | Common LoRA SFT learning rate for fast domain adaptation |
| `per_device_train_batch_size` | 1 | VLM image inputs are memory intensive |
| `gradient_accumulation_steps` | 4 | Provides effective batch size 4 under memory constraints |
| `lora_r` | 16 | Medium-capacity setting for small-data domain adaptation |
| `lora_alpha` | 16 | Matches rank and keeps LoRA update scale stable |
| `lora_dropout` | 0.0 | First experiment prioritizes learnability and avoids extra regularization instability |
| `seed` | 3407 | Makes the split and training setup reproducible |

4-bit LoRA is chosen for resource efficiency. Full fine-tuning a 9B VLM would require significantly more memory and training time, while this task has a small dataset and a constrained output space. LoRA is therefore a suitable first adaptation method. The `r=16` and `alpha=16` setting provides moderate adaptation capacity: enough to learn cockpit layout and JSON formatting, but not as large as a high-capacity adapter that might overfit the small dataset more aggressively.

The number of epochs is set to 4 as a first-pass compromise. With only 360 examples, too few epochs may not reliably teach the model the fixed schema and fact boundaries. Too many epochs may overfit Run-001. The final training record is:

| Metric | Value |
|---|---:|
| train rows | 324 |
| eval rows | 36 |
| train loss | 0.57036 |
| final eval loss | 0.03269 |
| train runtime | 2201s |

The eval split comes from the same Run-001 data pool. It is useful for training sanity checks, but it is not an independent measure of generalization. Generalization is primarily evaluated through Run-002.

## 6. Benchmark Method

The benchmark compares the base model against the same model with the LoRA adapter loaded. Both models receive the same reviewed JSONL examples, the same images, and the same English prompt. The generated JSON is parsed, normalized, and compared with the human-reviewed labels.

The metrics are:

| Metric | Meaning |
|---|---|
| JSON valid rate | Whether the output is parseable as a JSON object |
| schema valid rate | Whether the output fields match the expected schema |
| fact accuracy | Three-class accuracy over all facts |
| macro F1 | Macro F1 across `seen`, `not_seen`, and `uncertain` |
| seen F1 | F1 for the positive `seen` class |
| sample exact match | A sample is correct only if all 8 facts match |
| critical false positives | Errors where critical facts change from `not_seen/uncertain` to `seen` |

Critical false positives are tracked separately because, in task-state interpretation, incorrectly observing an unfinished state as complete is often more concerning than conservative `not_seen` predictions. The critical facts in this experiment are `fcs_bit_result_visible`, `ins_go`, `right_ddi_in_test_visible`, and `right_ddi_fcsmc_page_visible`.

## 7. Results

### 7.1 Run-001: Contaminated Development Set

Run-001 is evaluated on the same data pool from which training examples were derived. It is therefore a contaminated development set and is used to test whether the model learned the schema and in-distribution visual boundaries.

![Run-001 overall accuracy](assets/qwen35_vlm_finetune/run001_overall_accuracy.png)

![Run-001 fact F1 by model](assets/qwen35_vlm_finetune/run001_fact_f1_by_model.png)

![Run-001 seen F1 by fact](assets/qwen35_vlm_finetune/run001_seen_f1_by_fact.png)

![Run-001 critical false positives](assets/qwen35_vlm_finetune/run001_critical_false_positives.png)

| Metric | Base | LoRA-v1 |
|---|---:|---:|
| JSON valid rate | 1.0000 | 1.0000 |
| schema valid rate | 1.0000 | 1.0000 |
| fact accuracy | 0.7951 | 0.9694 |
| macro F1 | 0.4705 | 0.6375 |
| seen F1 | 0.6080 | 0.9517 |
| sample exact match | 0.1833 | 0.7778 |
| critical false positives | 19 | 2 |

Run-001 shows that LoRA-v1 learned the JSON output format and the cockpit visual facts within the training distribution. The improvements are large, especially for `fcs_page_visible`, `right_ddi_in_test_visible`, and `fcs_bit_result_visible`. However, because the benchmark is not independent of the training source, it should not be interpreted as the primary generalization result.

### 7.2 Run-002: Heldout New Session

Run-002 is a newly captured session and is not used in LoRA-v1 training. It is the more informative test for cross-session generalization.

![Run-002 overall accuracy](assets/qwen35_vlm_finetune/run002_overall_accuracy.png)

![Run-002 fact F1 by model](assets/qwen35_vlm_finetune/run002_fact_f1_by_model.png)

![Run-002 seen F1 by fact](assets/qwen35_vlm_finetune/run002_seen_f1_by_fact.png)

![Run-002 critical false positives](assets/qwen35_vlm_finetune/run002_critical_false_positives.png)

| Metric | Base | LoRA-v1 |
|---|---:|---:|
| JSON valid rate | 1.0000 | 1.0000 |
| schema valid rate | 1.0000 | 1.0000 |
| fact accuracy | 0.7600 | 0.9150 |
| macro F1 | 0.4241 | 0.5847 |
| seen F1 | 0.4714 | 0.8380 |
| sample exact match | 0.0600 | 0.3600 |
| critical false positives | 13 | 15 |

Run-002 confirms that LoRA-v1 improves overall fact extraction beyond the original data pool. `fcs_bit_result_visible`, `right_ddi_fcsmc_page_visible`, and `bit_page_failure_visible` are particularly stable. `ins_alignment_page_visible` is also substantially improved compared with the base model.

At the same time, Run-002 exposes the dominant failure mode: `ins_go` false positives. The base model is extremely conservative on `ins_go`, yielding zero seen F1 but also zero false positives. LoRA-v1 learns to detect some positive GO cases, but it also predicts `seen` for 13 samples that are human-labeled `not_seen`.

![Run-002 ins_go base confusion matrix](assets/qwen35_vlm_finetune/run002_confusion_ins_go_base.png)

![Run-002 ins_go LoRA confusion matrix](assets/qwen35_vlm_finetune/run002_confusion_ins_go_lora.png)

### 7.3 Per-Fact Observation

LoRA-v1 per-fact results on Run-002 are:

| fact_id | accuracy | seen precision | seen recall | seen F1 | seen false positives |
|---|---:|---:|---:|---:|---:|
| `fcs_page_visible` | 0.8400 | 0.6190 | 1.0000 | 0.7647 | 8 |
| `bit_root_page_visible` | 0.9400 | 0.7500 | 1.0000 | 0.8571 | 3 |
| `bit_page_failure_visible` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |
| `right_ddi_fcsmc_page_visible` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |
| `right_ddi_in_test_visible` | 0.9600 | 0.7500 | 1.0000 | 0.8571 | 2 |
| `fcs_bit_result_visible` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |
| `ins_alignment_page_visible` | 0.8800 | 1.0000 | 0.8605 | 0.9250 | 0 |
| `ins_go` | 0.7000 | 0.1875 | 0.7500 | 0.3000 | 13 |

The model does not fail globally. It improves most facts substantially. The main issue is concentrated in the semantic boundary of `ins_go`.

## 8. Discussion

### 8.1 Why `ins_go` Fails

`ins_go` is not a simple page-visibility fact. It is closer to a completion signal for the INS alignment process. Unlike `fcs_page_visible` or `bit_page_failure_visible`, it depends on specific text, page context, and procedural phase. In a single frame, AMPCD map layers, INS alignment pages, countdown states, and GO states can share similar locations, colors, and local visual structures. If hard negatives are underrepresented, the model may overgeneralize from “alignment-like AMPCD screen” to `ins_go=seen`.

This suggests that the current ontology mixes abstraction levels. Some facts are low-level visual observations, such as a visible failure list. Others are higher-level state judgments, such as INS being ready. The latter may not be appropriate as direct single-frame VLM targets. They should be decomposed into lower-level evidence and combined by downstream state reasoning.

### 8.2 Limitations of the Current Eight Facts

The eight facts are useful because they convert an open-ended visual QA task into a measurable structured extraction problem. However, they are not a final ontology.

The main limitations are:

- Mixed abstraction levels: page visibility, in-progress status, final result, and procedural completion are placed in the same layer.
- Overlapping semantics: `bit_root_page_visible` and `bit_page_failure_visible` can overlap in some cockpit UI states.
- Insufficient single-frame evidence: `ins_go` needs stricter evidence definition.
- Coarse labels: `seen/not_seen/uncertain` cannot express “partially visible but insufficient for procedural interpretation.”

Architecturally, the VLM should output low-level, locally verifiable evidence, such as “AMPCD shows an INS page,” “GO text is visible,” “MAP layer is visible,” or “alignment countdown is visible.” A downstream reasoning layer can then combine those facts with time and telemetry.

### 8.3 What Small-Data Fine-Tuning Demonstrates

The experiment shows that a small amount of human-reviewed data can significantly improve domain-specific structured VLM behavior. The Run-002 results are especially useful because they show that the model gains some cross-session generalization rather than merely reproducing Run-001.

However, small-data fine-tuning also amplifies ontology and data coverage issues. For rare, critical, and semantically complex states such as `ins_go`, bilingual augmentation does not replace visual hard negatives. It improves instruction robustness, not cockpit-state coverage. The next step should not simply maximize overall accuracy; it should redesign the targets and reduce critical false positives.

## 9. Future Work

The next phase should shift from maximizing overall accuracy to redesigning the visual ontology and reducing critical false positives.

First, decompose `ins_go` into lower-level observable facts:

| New fact | Meaning |
|---|---|
| `ampcd_ins_page_visible` | Whether the AMPCD shows an INS alignment-related page |
| `ampcd_ins_status_go_text_visible` | Whether explicit GO text is visible |
| `ampcd_map_layer_visible` | Whether the AMPCD shows the MAP layer |
| `ins_countdown_visible` | Whether an INS countdown or alignment value is visible |

Second, redefine BIT-related facts. `bit_root_page_visible` and `bit_page_failure_visible` should be separated into lower-level visual evidence such as menu entry, failure header, and failure list items.

Third, collect hard negatives. Run-003 should focus on AMPCD MAP screens, non-GO alignment states, countdown intermediate states, blank displays, and page transitions. If Run-002 failures are used for LoRA-v2 training, a new Run-003 must be used as an independent holdout.

Fourth, refine evaluation. Fact accuracy and seen F1 should still be reported, but critical false positives should become a primary criterion, especially for `ins_go`-related states.

## 10. Conclusion

This experiment completes a full loop from cockpit image capture, AI pre-labeling, human review, bilingual SFT export, Unsloth LoRA fine-tuning, and base-vs-LoRA benchmarking. The results show that `Qwen/Qwen3.5-9B-Base` can be substantially adapted to SimTutor cockpit visual fact extraction with a small amount of reviewed domain data. On the independent Run-002 benchmark, LoRA-v1 improves fact accuracy from `0.7600` to `0.9150` and seen F1 from `0.4714` to `0.8380`.

At the same time, the `ins_go` false positives reveal a significant weakness in the current target design. The main contribution of this iteration is therefore not that the first ontology is complete, but that the workflow makes such weaknesses measurable. A more robust next iteration should decompose higher-level procedural states into lower-level visual evidence and evaluate them on a new independent session.
