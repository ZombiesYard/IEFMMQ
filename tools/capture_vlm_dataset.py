"""
Help-triggered dataset capture tool for VLM fine-tuning.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping

from PIL import Image

from adapters.vision_capture_trigger import (
    DEFAULT_VISION_CAPTURE_TRIGGER_HOST,
    DEFAULT_VISION_CAPTURE_TRIGGER_PORT,
)
from adapters.windows_global_help_trigger import (
    DEFAULT_GLOBAL_HELP_COOLDOWN_MS,
    WindowsGlobalHelpTrigger,
)
from adapters.vision_frames import (
    DEFAULT_ARTIFACT_SUFFIX,
    build_frame_filename,
    build_frame_id,
    render_vlm_ready_frame,
)
from simtutor.schemas import validate_instance
from tools.capture_vision_sidecar import (
    UdpCaptureRequestListener,
    capture_screen_region,
    load_sidecar_config,
)

DEFAULT_CAPTURE_FPS = 2.0
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / ".captures"


@dataclass(frozen=True)
class DatasetCaptureConfig:
    session_id: str
    output_root: Path
    channel: str
    layout_id: str
    screen_width: int
    screen_height: int
    render_vlm_artifacts: bool


@dataclass(frozen=True)
class DatasetCaptureStats:
    frames_written: int
    help_start_captures: int
    interval_captures: int
    started: bool
    last_frame_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames_written": self.frames_written,
            "help_start_captures": self.help_start_captures,
            "interval_captures": self.interval_captures,
            "started": self.started,
            "last_frame_id": self.last_frame_id,
        }


@dataclass(frozen=True)
class CapturePlanItem:
    seq: int
    total: int
    category_id: str
    category_name: str
    target_display: str
    left_ddi_content: str
    ampcd_content: str
    right_ddi_content: str
    expected_primary_content: str
    expected_other_displays: str
    expected_key_facts: str
    capture_instruction: str

    def to_row(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "total": self.total,
            "category_id": self.category_id,
            "category_name": self.category_name,
            "target_display": self.target_display,
            "left_ddi_content": self.left_ddi_content,
            "ampcd_content": self.ampcd_content,
            "right_ddi_content": self.right_ddi_content,
            "expected_primary_content": self.expected_primary_content,
            "expected_other_displays": self.expected_other_displays,
            "expected_key_facts": self.expected_key_facts,
            "capture_instruction": self.capture_instruction,
        }


def _default_config_path(saved_games_dir: Path) -> Path:
    return saved_games_dir / "Scripts" / "SimTutor" / "SimTutorConfig.lua"


_CAPTURE_PLAN_FIELDNAMES = [
    "seq",
    "total",
    "category_id",
    "category_name",
    "target_display",
    "left_ddi_content",
    "ampcd_content",
    "right_ddi_content",
    "expected_primary_content",
    "expected_other_displays",
    "expected_key_facts",
    "capture_instruction",
]

_CAPTURE_PLAN_PROGRESS_FIELDNAMES = [
    *_CAPTURE_PLAN_FIELDNAMES,
    "captured_frame_id",
    "capture_reason",
    "artifact_image_path",
    "raw_image_path",
    "captured_at_wall_ms",
    "status",
    "operator_note",
]


def _build_fa18c_run003_v3_220_plan() -> list[CapturePlanItem]:
    total = 220
    non_target_pool = "EW/RADAR/STORES/MIDS/SA/DATA/FPAS/CHKLIST or another non-target page"
    specs = [
        {
            "count": 8,
            "category_id": "tac_page",
            "category_name": "TAC page on left DDI",
            "target_display": "left_ddi",
            "left_ddi_content": "TAC / TAC MENU",
            "ampcd_content": "Natural cold-start context; HSI/MAP is acceptable.",
            "right_ddi_content": "Natural context; avoid BIT root or FCS-MC if practical.",
            "expected_primary_content": "TAC / TAC MENU",
            "expected_other_displays": "AMPCD and right DDI stay realistic; avoid intentionally showing another target page.",
            "expected_key_facts": "tac_page_visible=seen; supt_page_visible=not_seen; fcs_page_visible=not_seen",
            "capture_instruction": "Set left DDI to TAC / TAC MENU. Other displays may remain natural.",
        },
        {
            "count": 2,
            "category_id": "tac_page",
            "category_name": "TAC page off-canonical display",
            "target_display": "right_ddi",
            "left_ddi_content": "Natural non-target or blank page.",
            "ampcd_content": "Natural cold-start context; HSI/MAP is acceptable.",
            "right_ddi_content": "TAC / TAC MENU",
            "expected_primary_content": "TAC / TAC MENU",
            "expected_other_displays": "Small display-position perturbation for TAC recognition.",
            "expected_key_facts": "tac_page_visible=seen; supt_page_visible=not_seen; fcs_page_visible=not_seen",
            "capture_instruction": "Set right DDI to TAC / TAC MENU if convenient.",
        },
        {
            "count": 8,
            "category_id": "supt_page",
            "category_name": "SUPT page on left DDI",
            "target_display": "left_ddi",
            "left_ddi_content": "SUPT page",
            "ampcd_content": "Natural cold-start context; HSI/MAP is acceptable.",
            "right_ddi_content": "Natural context; avoid BIT root or FCS-MC if practical.",
            "expected_primary_content": "SUPT page",
            "expected_other_displays": "AMPCD and right DDI stay realistic; avoid intentionally showing another target page.",
            "expected_key_facts": "supt_page_visible=seen; tac_page_visible=not_seen; fcs_page_visible=not_seen",
            "capture_instruction": "Set left DDI to SUPT.",
        },
        {
            "count": 2,
            "category_id": "supt_page",
            "category_name": "SUPT page off-canonical display",
            "target_display": "right_ddi",
            "left_ddi_content": "Natural non-target or blank page.",
            "ampcd_content": "Natural cold-start context; HSI/MAP is acceptable.",
            "right_ddi_content": "SUPT page",
            "expected_primary_content": "SUPT page",
            "expected_other_displays": "Small display-position perturbation for SUPT recognition.",
            "expected_key_facts": "supt_page_visible=seen; tac_page_visible=not_seen; fcs_page_visible=not_seen",
            "capture_instruction": "Set right DDI to SUPT if convenient.",
        },
        {
            "count": 6,
            "category_id": "other_non_target_page",
            "category_name": "Left DDI non-target page",
            "target_display": "left_ddi",
            "left_ddi_content": non_target_pool,
            "ampcd_content": "Non-HSI page, blank, or map-free non-target if practical.",
            "right_ddi_content": "Non-target or blank page.",
            "expected_primary_content": "No TAC/SUPT/FCS/BIT root/FCS-MC/HSI target page visible",
            "expected_other_displays": "Use this block for none-of-target negatives, not for page classification.",
            "expected_key_facts": "all target page facts should be not_seen unless the image is genuinely unreadable",
            "capture_instruction": "Put a non-target page on left DDI; avoid all target pages on all displays.",
        },
        {
            "count": 6,
            "category_id": "other_non_target_page",
            "category_name": "Right DDI non-target page",
            "target_display": "right_ddi",
            "left_ddi_content": "Non-target or blank page.",
            "ampcd_content": "Non-HSI page, blank, or map-free non-target if practical.",
            "right_ddi_content": non_target_pool,
            "expected_primary_content": "No TAC/SUPT/FCS/BIT root/FCS-MC/HSI target page visible",
            "expected_other_displays": "Use this block for none-of-target negatives, not for page classification.",
            "expected_key_facts": "all target page facts should be not_seen unless the image is genuinely unreadable",
            "capture_instruction": "Put a non-target page on right DDI; avoid all target pages on all displays.",
        },
        {
            "count": 4,
            "category_id": "other_non_target_page",
            "category_name": "AMPCD non-HSI page",
            "target_display": "ampcd",
            "left_ddi_content": "Non-target or blank page.",
            "ampcd_content": "Non-HSI/non-INS page if practical; otherwise blank/dark stable display.",
            "right_ddi_content": "Non-target or blank page.",
            "expected_primary_content": "No HSI/INS page on AMPCD and no target page elsewhere",
            "expected_other_displays": "This block tests that AMPCD content is not automatically HSI/INS.",
            "expected_key_facts": "hsi_page_visible=not_seen; other target page facts=not_seen",
            "capture_instruction": "Put AMPCD on a non-HSI page or stable blank/dark state.",
        },
        {
            "count": 8,
            "category_id": "other_non_target_page",
            "category_name": "All displays non-target",
            "target_display": "all",
            "left_ddi_content": non_target_pool,
            "ampcd_content": "Non-HSI page, blank, or stable non-target display.",
            "right_ddi_content": non_target_pool,
            "expected_primary_content": "All three displays are non-target pages",
            "expected_other_displays": "This block represents a user who has fully navigated away from the procedure pages.",
            "expected_key_facts": "all target page facts should be not_seen",
            "capture_instruction": "Set all three displays to non-target pages; rotate EW/RADAR/STORES/MIDS/SA/DATA when convenient.",
        },
        {
            "count": 4,
            "category_id": "other_non_target_page",
            "category_name": "Mixed non-target plus stable blank",
            "target_display": "all",
            "left_ddi_content": "Non-target or stable blank.",
            "ampcd_content": "Non-HSI non-target or stable blank.",
            "right_ddi_content": "Non-target or stable blank.",
            "expected_primary_content": "No target page visible; at least one display may be stable blank/dark",
            "expected_other_displays": "Stable blank/dark is allowed here, but not a transition blur.",
            "expected_key_facts": "all target page facts should be not_seen",
            "capture_instruction": "Capture mixed non-target and stable blank/dark displays.",
        },
        {
            "count": 12,
            "category_id": "fcs_page_with_x",
            "category_name": "FCS page with X marks on left DDI",
            "target_display": "left_ddi",
            "left_ddi_content": "FCS page with obvious X/fault fills",
            "ampcd_content": "Natural cold-start context; HSI/MAP is acceptable.",
            "right_ddi_content": "Natural context; avoid BIT root or FCS-MC if practical.",
            "expected_primary_content": "FCS page with obvious X/fault fills",
            "expected_other_displays": "Do not intentionally show FCS-MC on another display.",
            "expected_key_facts": "fcs_page_visible=seen; fcs_page_x_marks_visible=seen; fcsmc_page_visible=not_seen",
            "capture_instruction": "Set left DDI to the true FCS page before FCS reset, with X/fault fills visible.",
        },
        {
            "count": 4,
            "category_id": "fcs_page_with_x",
            "category_name": "FCS page with X marks off-canonical display",
            "target_display": "right_ddi",
            "left_ddi_content": "Natural non-target or blank page.",
            "ampcd_content": "Natural cold-start context; HSI/MAP is acceptable.",
            "right_ddi_content": "FCS page with obvious X/fault fills",
            "expected_primary_content": "FCS page with obvious X/fault fills",
            "expected_other_displays": "Small display-position perturbation for FCS recognition.",
            "expected_key_facts": "fcs_page_visible=seen; fcs_page_x_marks_visible=seen; fcsmc_page_visible=not_seen",
            "capture_instruction": "Set right DDI to the true FCS page before FCS reset if convenient.",
        },
        {
            "count": 10,
            "category_id": "fcs_page_without_obvious_x",
            "category_name": "FCS page reset-cleared on left DDI",
            "target_display": "left_ddi",
            "left_ddi_content": "FCS page without obvious X/fault fills",
            "ampcd_content": "Natural cold-start context; HSI/MAP is acceptable.",
            "right_ddi_content": "Natural context; avoid BIT root or FCS-MC if practical.",
            "expected_primary_content": "FCS page without obvious X/fault fills",
            "expected_other_displays": "Do not intentionally show FCS-MC on another display.",
            "expected_key_facts": "fcs_page_visible=seen; fcs_page_x_marks_visible=not_seen; fcsmc_page_visible=not_seen",
            "capture_instruction": "Set left DDI to the true FCS page after FCS reset or when X/fault fills are no longer obvious.",
        },
        {
            "count": 4,
            "category_id": "fcs_page_without_obvious_x",
            "category_name": "FCS page reset-cleared off-canonical display",
            "target_display": "right_ddi",
            "left_ddi_content": "Natural non-target or blank page.",
            "ampcd_content": "Natural cold-start context; HSI/MAP is acceptable.",
            "right_ddi_content": "FCS page without obvious X/fault fills",
            "expected_primary_content": "FCS page without obvious X/fault fills",
            "expected_other_displays": "Small display-position perturbation for reset-cleared FCS recognition.",
            "expected_key_facts": "fcs_page_visible=seen; fcs_page_x_marks_visible=not_seen; fcsmc_page_visible=not_seen",
            "capture_instruction": "Set right DDI to the true FCS page after reset if convenient.",
        },
        {
            "count": 14,
            "category_id": "bit_root_failures",
            "category_name": "BIT root on right DDI",
            "target_display": "right_ddi",
            "left_ddi_content": "Natural non-target or SUPT only if needed for navigation.",
            "ampcd_content": "Natural cold-start context; HSI/MAP is acceptable.",
            "right_ddi_content": "BIT FAILURES / BIT root page",
            "expected_primary_content": "BIT FAILURES / BIT root",
            "expected_other_displays": "Avoid showing FCS-MC on any other display.",
            "expected_key_facts": "bit_root_page_visible=seen; fcsmc_page_visible=not_seen",
            "capture_instruction": "Set right DDI to BIT FAILURES / BIT root; do not enter the FCS-MC subpage yet.",
        },
        {
            "count": 4,
            "category_id": "bit_root_failures",
            "category_name": "BIT root off-canonical display",
            "target_display": "left_ddi",
            "left_ddi_content": "BIT FAILURES / BIT root page",
            "ampcd_content": "Natural cold-start context; HSI/MAP is acceptable.",
            "right_ddi_content": "Natural non-target or blank page.",
            "expected_primary_content": "BIT FAILURES / BIT root",
            "expected_other_displays": "Small display-position perturbation for BIT root recognition.",
            "expected_key_facts": "bit_root_page_visible=seen; fcsmc_page_visible=not_seen",
            "capture_instruction": "Set left DDI to BIT FAILURES / BIT root if convenient; do not enter FCS-MC.",
        },
        {
            "count": 16,
            "category_id": "fcsmc_pbit_go",
            "category_name": "FCS-MC PBIT GO on right DDI",
            "target_display": "right_ddi",
            "left_ddi_content": "Natural non-target or blank page.",
            "ampcd_content": "Natural cold-start context; HSI/MAP is acceptable.",
            "right_ddi_content": "FCS-MC page with FCSA/FCSB PBIT GO",
            "expected_primary_content": "FCS-MC with PBIT GO",
            "expected_other_displays": "PBIT GO is an intermediate status, not final GO.",
            "expected_key_facts": "fcsmc_page_visible=seen; fcsmc_intermediate_result_visible=seen; fcsmc_final_go_result_visible=not_seen",
            "capture_instruction": "Set right DDI to FCS-MC before running the test, with FCSA/FCSB PBIT GO visible.",
        },
        {
            "count": 2,
            "category_id": "fcsmc_pbit_go",
            "category_name": "FCS-MC PBIT GO off-canonical display",
            "target_display": "left_ddi",
            "left_ddi_content": "FCS-MC page with FCSA/FCSB PBIT GO",
            "ampcd_content": "Natural cold-start context; HSI/MAP is acceptable.",
            "right_ddi_content": "Natural non-target or blank page.",
            "expected_primary_content": "FCS-MC with PBIT GO",
            "expected_other_displays": "Small display-position perturbation for FCS-MC recognition.",
            "expected_key_facts": "fcsmc_page_visible=seen; fcsmc_intermediate_result_visible=seen; fcsmc_final_go_result_visible=not_seen",
            "capture_instruction": "Set left DDI to FCS-MC PBIT GO if convenient.",
        },
        {
            "count": 16,
            "category_id": "fcsmc_in_test",
            "category_name": "FCS-MC IN TEST on right DDI",
            "target_display": "right_ddi",
            "left_ddi_content": "Natural non-target or blank page.",
            "ampcd_content": "Natural cold-start context; HSI/MAP is acceptable.",
            "right_ddi_content": "FCS-MC page showing IN TEST",
            "expected_primary_content": "FCS-MC IN TEST",
            "expected_other_displays": "IN TEST is a running state, not final GO.",
            "expected_key_facts": "fcsmc_page_visible=seen; fcsmc_in_test_visible=seen; fcsmc_final_go_result_visible=not_seen",
            "capture_instruction": "Start FCS-MC BIT and capture while the page clearly shows IN TEST.",
        },
        {
            "count": 2,
            "category_id": "fcsmc_in_test",
            "category_name": "FCS-MC IN TEST off-canonical display",
            "target_display": "left_ddi",
            "left_ddi_content": "FCS-MC page showing IN TEST",
            "ampcd_content": "Natural cold-start context; HSI/MAP is acceptable.",
            "right_ddi_content": "Natural non-target or blank page.",
            "expected_primary_content": "FCS-MC IN TEST",
            "expected_other_displays": "Small display-position perturbation for FCS-MC IN TEST.",
            "expected_key_facts": "fcsmc_page_visible=seen; fcsmc_in_test_visible=seen; fcsmc_final_go_result_visible=not_seen",
            "capture_instruction": "Set left DDI to FCS-MC IN TEST if convenient.",
        },
        {
            "count": 16,
            "category_id": "fcsmc_final_go",
            "category_name": "FCS-MC final GO on right DDI",
            "target_display": "right_ddi",
            "left_ddi_content": "Natural non-target or blank page.",
            "ampcd_content": "Natural cold-start context; HSI/MAP is acceptable.",
            "right_ddi_content": "FCS-MC page with final GO results",
            "expected_primary_content": "FCS-MC final GO",
            "expected_other_displays": "Final GO should be clearly distinguishable from PBIT GO and IN TEST.",
            "expected_key_facts": "fcsmc_page_visible=seen; fcsmc_final_go_result_visible=seen; fcsmc_intermediate_result_visible=not_seen; fcsmc_in_test_visible=not_seen",
            "capture_instruction": "Capture after FCS-MC BIT completes and the final GO results are clearly visible.",
        },
        {
            "count": 2,
            "category_id": "fcsmc_final_go",
            "category_name": "FCS-MC final GO off-canonical display",
            "target_display": "left_ddi",
            "left_ddi_content": "FCS-MC page with final GO results",
            "ampcd_content": "Natural cold-start context; HSI/MAP is acceptable.",
            "right_ddi_content": "Natural non-target or blank page.",
            "expected_primary_content": "FCS-MC final GO",
            "expected_other_displays": "Small display-position perturbation for final GO.",
            "expected_key_facts": "fcsmc_page_visible=seen; fcsmc_final_go_result_visible=seen; fcsmc_intermediate_result_visible=not_seen; fcsmc_in_test_visible=not_seen",
            "capture_instruction": "Set left DDI to FCS-MC final GO if convenient.",
        },
        {
            "count": 20,
            "category_id": "hsi_map_overlay",
            "category_name": "HSI with MAP overlay on AMPCD",
            "target_display": "ampcd",
            "left_ddi_content": "Natural non-target or procedure context.",
            "ampcd_content": "HSI with MAP overlay; INS/QUAL text may be hard to read.",
            "right_ddi_content": "Natural non-target or procedure context.",
            "expected_primary_content": "HSI with MAP overlay",
            "expected_other_displays": "This is the main INS hard negative block.",
            "expected_key_facts": "hsi_page_visible=seen; hsi_map_layer_visible=seen; ins_ok_text_visible=not_seen/uncertain",
            "capture_instruction": "Set AMPCD to HSI with MAP overlay on; capture cases where INS text is partly obscured or hard to read.",
        },
        {
            "count": 2,
            "category_id": "hsi_map_overlay",
            "category_name": "HSI with MAP overlay on AMPCD, alternate DDI context",
            "target_display": "ampcd",
            "left_ddi_content": "Natural non-target, TAC/SUPT navigation context, or stable blank page.",
            "ampcd_content": "HSI with MAP overlay; use a different range/zoom/clutter state if practical.",
            "right_ddi_content": "Natural non-target, TAC/SUPT navigation context, or stable blank page.",
            "expected_primary_content": "HSI with MAP overlay",
            "expected_other_displays": "DDIs provide background variation; MAP overlay remains on AMPCD because DDI MAP overlay is not available.",
            "expected_key_facts": "hsi_page_visible=seen; hsi_map_layer_visible=seen; ins_ok_text_visible=not_seen/uncertain",
            "capture_instruction": "Keep HSI with MAP overlay on AMPCD; vary left/right DDI background pages instead of moving MAP overlay to a DDI.",
        },
        {
            "count": 2,
            "category_id": "hsi_map_overlay",
            "category_name": "HSI with MAP overlay on AMPCD, hard-to-read variant",
            "target_display": "ampcd",
            "left_ddi_content": "Natural non-target, TAC/SUPT navigation context, or stable blank page.",
            "ampcd_content": "HSI with MAP overlay; prefer hard-to-read INS/QUAL text, clutter, or similar-looking map symbols.",
            "right_ddi_content": "Natural non-target, TAC/SUPT navigation context, or stable blank page.",
            "expected_primary_content": "HSI with MAP overlay",
            "expected_other_displays": "This replaces the impossible DDI MAP-overlay perturbation with AMPCD visual variation.",
            "expected_key_facts": "hsi_page_visible=seen; hsi_map_layer_visible=seen; ins_ok_text_visible=not_seen/uncertain",
            "capture_instruction": "Keep HSI with MAP overlay on AMPCD; choose a visually difficult MAP overlay frame if possible.",
        },
        {
            "count": 18,
            "category_id": "hsi_alignment_running_map_off",
            "category_name": "HSI alignment running, MAP off on AMPCD",
            "target_display": "ampcd",
            "left_ddi_content": "Natural non-target or procedure context.",
            "ampcd_content": "HSI with MAP off; GRND/QUAL/countdown visible; no OK.",
            "right_ddi_content": "Natural non-target or procedure context.",
            "expected_primary_content": "GRND/QUAL/countdown, no OK",
            "expected_other_displays": "This separates alignment running from final OK.",
            "expected_key_facts": "hsi_page_visible=seen; hsi_map_layer_visible=not_seen; ins_grnd_alignment_text_visible=seen; ins_ok_text_visible=not_seen",
            "capture_instruction": "Turn MAP off on HSI and capture alignment running states with QUAL/countdown visible but no OK.",
        },
        {
            "count": 2,
            "category_id": "hsi_alignment_running_map_off",
            "category_name": "HSI alignment running off-canonical display",
            "target_display": "left_ddi",
            "left_ddi_content": "HSI with MAP off; GRND/QUAL/countdown visible; no OK.",
            "ampcd_content": "Natural non-HSI or blank if practical.",
            "right_ddi_content": "Natural non-target or blank page.",
            "expected_primary_content": "GRND/QUAL/countdown, no OK",
            "expected_other_displays": "Small display-position perturbation for HSI alignment running.",
            "expected_key_facts": "hsi_page_visible=seen; hsi_map_layer_visible=not_seen; ins_grnd_alignment_text_visible=seen; ins_ok_text_visible=not_seen",
            "capture_instruction": "Set left DDI to HSI MAP off with running alignment if convenient.",
        },
        {
            "count": 2,
            "category_id": "hsi_alignment_running_map_off",
            "category_name": "HSI alignment running off-canonical display",
            "target_display": "right_ddi",
            "left_ddi_content": "Natural non-target or blank page.",
            "ampcd_content": "Natural non-HSI or blank if practical.",
            "right_ddi_content": "HSI with MAP off; GRND/QUAL/countdown visible; no OK.",
            "expected_primary_content": "GRND/QUAL/countdown, no OK",
            "expected_other_displays": "Small display-position perturbation for HSI alignment running.",
            "expected_key_facts": "hsi_page_visible=seen; hsi_map_layer_visible=not_seen; ins_grnd_alignment_text_visible=seen; ins_ok_text_visible=not_seen",
            "capture_instruction": "Set right DDI to HSI MAP off with running alignment if convenient.",
        },
        {
            "count": 10,
            "category_id": "hsi_ok",
            "category_name": "HSI OK on AMPCD",
            "target_display": "ampcd",
            "left_ddi_content": "Natural non-target or procedure context.",
            "ampcd_content": "HSI with clear OK text.",
            "right_ddi_content": "Natural non-target or procedure context.",
            "expected_primary_content": "HSI OK",
            "expected_other_displays": "Final OK positive examples are intentionally fewer than running/MAP hard negatives.",
            "expected_key_facts": "hsi_page_visible=seen; ins_ok_text_visible=seen",
            "capture_instruction": "Capture HSI after alignment reaches a clearly visible OK state.",
        },
        {
            "count": 1,
            "category_id": "hsi_ok",
            "category_name": "HSI OK off-canonical display",
            "target_display": "left_ddi",
            "left_ddi_content": "HSI with clear OK text.",
            "ampcd_content": "Natural non-HSI or blank if practical.",
            "right_ddi_content": "Natural non-target or blank page.",
            "expected_primary_content": "HSI OK",
            "expected_other_displays": "Small display-position perturbation for HSI OK.",
            "expected_key_facts": "hsi_page_visible=seen; ins_ok_text_visible=seen",
            "capture_instruction": "Set left DDI to HSI OK if convenient.",
        },
        {
            "count": 1,
            "category_id": "hsi_ok",
            "category_name": "HSI OK off-canonical display",
            "target_display": "right_ddi",
            "left_ddi_content": "Natural non-target or blank page.",
            "ampcd_content": "Natural non-HSI or blank if practical.",
            "right_ddi_content": "HSI with clear OK text.",
            "expected_primary_content": "HSI OK",
            "expected_other_displays": "Small display-position perturbation for HSI OK.",
            "expected_key_facts": "hsi_page_visible=seen; ins_ok_text_visible=seen",
            "capture_instruction": "Set right DDI to HSI OK if convenient.",
        },
        {
            "count": 4,
            "category_id": "unreadable_transition",
            "category_name": "Page transition / blur",
            "target_display": "any",
            "left_ddi_content": "Any display may be changing pages, blurred, or partly unreadable.",
            "ampcd_content": "Any display may be changing pages, blurred, or partly unreadable.",
            "right_ddi_content": "Any display may be changing pages, blurred, or partly unreadable.",
            "expected_primary_content": "transition/blur/unreadable",
            "expected_other_displays": "Capture realistic page-change moments or motion blur.",
            "expected_key_facts": "ambiguous target facts should be uncertain rather than guessed seen",
            "capture_instruction": "Press capture during a page transition or while the text is visibly blurred/unreadable.",
        },
        {
            "count": 4,
            "category_id": "unreadable_transition",
            "category_name": "Stable dark or partly blank display",
            "target_display": "any",
            "left_ddi_content": "Stable blank/dark or partly unreadable display.",
            "ampcd_content": "Stable blank/dark or partly unreadable display.",
            "right_ddi_content": "Stable blank/dark or partly unreadable display.",
            "expected_primary_content": "stable blank/dark/unreadable state",
            "expected_other_displays": "This teaches conservative not_seen/uncertain behavior.",
            "expected_key_facts": "target facts should be not_seen when truly absent, uncertain when unreadable",
            "capture_instruction": "Capture stable blank/dark or partially unreadable displays.",
        },
        {
            "count": 4,
            "category_id": "unreadable_transition",
            "category_name": "Cropped or obscured text",
            "target_display": "any",
            "left_ddi_content": "Target or non-target page may be cropped/obscured.",
            "ampcd_content": "Target or non-target page may be cropped/obscured.",
            "right_ddi_content": "Target or non-target page may be cropped/obscured.",
            "expected_primary_content": "cropped/obscured/unreadable text",
            "expected_other_displays": "Useful for uncertain labels and failed OCR-like cases.",
            "expected_key_facts": "ambiguous target facts should be uncertain rather than guessed seen",
            "capture_instruction": "Capture a case where one or more relevant labels/text areas are not readable enough to trust.",
        },
    ]
    if sum(int(spec["count"]) for spec in specs) != total:
        raise AssertionError("manual capture plan must contain exactly 220 items")

    out: list[CapturePlanItem] = []
    seq = 1
    for spec in specs:
        for _ in range(int(spec["count"])):
            out.append(
                CapturePlanItem(
                    seq=seq,
                    total=total,
                    category_id=str(spec["category_id"]),
                    category_name=str(spec["category_name"]),
                    target_display=str(spec["target_display"]),
                    left_ddi_content=str(spec["left_ddi_content"]),
                    ampcd_content=str(spec["ampcd_content"]),
                    right_ddi_content=str(spec["right_ddi_content"]),
                    expected_primary_content=str(spec["expected_primary_content"]),
                    expected_other_displays=str(spec["expected_other_displays"]),
                    expected_key_facts=str(spec["expected_key_facts"]),
                    capture_instruction=str(spec["capture_instruction"]),
                )
            )
            seq += 1
    return out


def _resolve_manual_plan(name: str) -> list[CapturePlanItem]:
    normalized = str(name).strip()
    if normalized == "fa18c_run003_v3_220":
        return _build_fa18c_run003_v3_220_plan()
    if normalized == "fa18c_run003_v2_200":
        raise ValueError("fa18c_run003_v2_200 was superseded by fa18c_run003_v3_220")
    raise ValueError(f"unsupported manual plan: {name!r}")


def _write_capture_plan_csv(session_dir: Path, plan: list[CapturePlanItem]) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / "capture_plan.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CAPTURE_PLAN_FIELDNAMES)
        writer.writeheader()
        for item in plan:
            writer.writerow(item.to_row())


def _capture_plan_progress_row(
    *,
    item: CapturePlanItem,
    frame: Mapping[str, Any] | None,
    status: str,
    operator_note: str = "",
) -> dict[str, Any]:
    row = item.to_row()
    frame_payload = frame if isinstance(frame, Mapping) else {}
    row.update(
        {
            "captured_frame_id": frame_payload.get("frame_id") or "",
            "capture_reason": frame_payload.get("capture_reason") or f"manual_plan:{item.category_id}",
            "artifact_image_path": frame_payload.get("artifact_image_path") or "",
            "raw_image_path": frame_payload.get("image_path") or "",
            "captured_at_wall_ms": frame_payload.get("capture_wall_ms") or "",
            "status": status,
            "operator_note": operator_note,
        }
    )
    return row


def _append_capture_plan_progress_csv(
    session_dir: Path,
    *,
    item: CapturePlanItem,
    frame: Mapping[str, Any] | None,
    status: str,
    operator_note: str = "",
) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / "capture_plan_progress.csv"
    write_header = not path.exists() or path.stat().st_size == 0
    row = _capture_plan_progress_row(
        item=item,
        frame=frame,
        status=status,
        operator_note=operator_note,
    )
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CAPTURE_PLAN_PROGRESS_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _append_capture_plan_progress_jsonl(
    session_dir: Path,
    *,
    item: CapturePlanItem,
    frame: Mapping[str, Any] | None,
    status: str,
    operator_note: str = "",
) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / "capture_plan_progress.jsonl"
    row = _capture_plan_progress_row(
        item=item,
        frame=frame,
        status=status,
        operator_note=operator_note,
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def run_manual_plan_capture(
    *,
    writer: "DatasetFrameWriter",
    plan: list[CapturePlanItem],
    event_sources: list["CaptureEventSource"] | None = None,
    start_seq: int = 1,
    input_func: Callable[[str], str] = input,
    sleep: Callable[[float], None] = time.sleep,
    idle_sleep_s: float = 0.02,
) -> DatasetCaptureStats:
    if not plan:
        raise ValueError("manual capture plan must not be empty")
    start_seq = int(start_seq)
    if start_seq < 1:
        raise ValueError("manual plan start seq must be >= 1")
    if start_seq > plan[-1].total + 1:
        raise ValueError(f"manual plan start seq must be <= {plan[-1].total + 1}")
    session_dir = writer.session_dir
    plan_csv_path = session_dir / "capture_plan.csv"
    if start_seq <= 1 or not plan_csv_path.exists():
        _write_capture_plan_csv(session_dir, plan)
    print(
        "[CAPTURE_VLM_DATASET] manual_plan="
        + json.dumps(
            {
                "items": len(plan),
                "start_seq": start_seq,
                "capture_plan": str((session_dir / "capture_plan.csv").resolve()),
                "capture_plan_progress_csv": str((session_dir / "capture_plan_progress.csv").resolve()),
                "capture_plan_progress_jsonl": str((session_dir / "capture_plan_progress.jsonl").resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    frames_written = 0
    skipped = 0
    last_frame_id: str | None = None
    sources = list(event_sources or [])
    for item in [candidate for candidate in plan if candidate.seq >= start_seq]:
        if sources:
            print(
                f"[{item.seq:03d}/{item.total:03d}] {item.category_id} | "
                f"target_display={item.target_display}"
            )
            print(f"Goal: {item.expected_primary_content} on {item.target_display}.")
            print(f"Left DDI: {item.left_ddi_content}")
            print(f"AMPCD: {item.ampcd_content}")
            print(f"Right DDI: {item.right_ddi_content}")
            print(f"Instruction: {item.capture_instruction}")
            print("[CAPTURE_VLM_DATASET] press the configured trigger once to capture; Ctrl+C to stop")
            while True:
                request = None
                for event_source in sources:
                    request = event_source.poll()
                    if request is not None:
                        break
                if request is None:
                    sleep(max(0.0, idle_sleep_s))
                    continue
                reason = f"manual_plan:{item.category_id}"
                frame = writer.capture_frame(reason=reason)
                last_frame_id = str(frame["frame_id"])
                frames_written += 1
                _append_capture_plan_progress_csv(
                    session_dir,
                    item=item,
                    frame=frame,
                    status="captured",
                )
                _append_capture_plan_progress_jsonl(
                    session_dir,
                    item=item,
                    frame=frame,
                    status="captured",
                )
                _print_frame_event(frame)
                break
            continue

        while True:
            prompt = (
                f"[{item.seq:03d}/{item.total:03d}] {item.category_id} | "
                f"target_display={item.target_display}\n"
                f"Goal: {item.expected_primary_content} on {item.target_display}.\n"
                f"Left DDI: {item.left_ddi_content}\n"
                f"AMPCD: {item.ampcd_content}\n"
                f"Right DDI: {item.right_ddi_content}\n"
                f"Instruction: {item.capture_instruction}\n"
                "Press Enter to capture, s to skip, q to quit: "
            )
            try:
                command = input_func(prompt).strip().lower()
            except EOFError:
                command = "q"
            if command in {"q", "quit", "exit"}:
                print("[CAPTURE_VLM_DATASET] manual capture stopped by operator")
                return DatasetCaptureStats(
                    frames_written=frames_written,
                    help_start_captures=0,
                    interval_captures=frames_written,
                    started=frames_written > 0 or skipped > 0,
                    last_frame_id=last_frame_id,
                )
            if command in {"s", "skip"}:
                _append_capture_plan_progress_csv(
                    session_dir,
                    item=item,
                    frame=None,
                    status="skipped",
                )
                _append_capture_plan_progress_jsonl(
                    session_dir,
                    item=item,
                    frame=None,
                    status="skipped",
                )
                skipped += 1
                print(f"[CAPTURE_VLM_DATASET] skipped seq={item.seq:03d} category={item.category_id}")
                break
            if command == "":
                frame = writer.capture_frame(reason=f"manual_plan:{item.category_id}")
                last_frame_id = str(frame["frame_id"])
                frames_written += 1
                _append_capture_plan_progress_csv(
                    session_dir,
                    item=item,
                    frame=frame,
                    status="captured",
                )
                _append_capture_plan_progress_jsonl(
                    session_dir,
                    item=item,
                    frame=frame,
                    status="captured",
                )
                _print_frame_event(frame)
                break
            print("[CAPTURE_VLM_DATASET] unknown command; press Enter, s, or q")

    print("[CAPTURE_VLM_DATASET] manual capture plan completed")
    return DatasetCaptureStats(
        frames_written=frames_written,
        help_start_captures=0,
        interval_captures=frames_written,
        started=frames_written > 0 or skipped > 0,
        last_frame_id=last_frame_id,
    )


class DatasetFrameWriter:
    def __init__(
        self,
        *,
        output_root: Path,
        session_id: str,
        channel: str,
        layout_id: str,
        capture_callable: Callable[[], Image.Image],
        render_vlm_artifacts: bool = True,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.output_root = Path(output_root).expanduser().resolve()
        self.session_id = str(session_id).strip()
        self.channel = str(channel).strip()
        self.layout_id = str(layout_id).strip()
        if not self.session_id:
            raise ValueError("session_id must be non-empty")
        if not self.channel:
            raise ValueError("channel must be non-empty")
        if not self.layout_id:
            raise ValueError("layout_id must be non-empty")
        self.capture_callable = capture_callable
        self.render_vlm_artifacts = bool(render_vlm_artifacts)
        self.clock = clock
        self._frame_seq = self._resolve_next_frame_seq()

    @property
    def session_dir(self) -> Path:
        return self.output_root / self.session_id

    @property
    def raw_dir(self) -> Path:
        return self.session_dir / "raw"

    @property
    def artifact_dir(self) -> Path:
        return self.session_dir / "artifacts"

    @property
    def manifest_path(self) -> Path:
        return self.session_dir / "frames.jsonl"

    @property
    def capture_index_path(self) -> Path:
        return self.session_dir / "capture_index.jsonl"

    def _resolve_next_frame_seq(self) -> int:
        path = self.manifest_path
        if not path.exists():
            return 0
        max_frame_seq = -1
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                    frame_seq = int(payload.get("frame_seq", -1))
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
                max_frame_seq = max(max_frame_seq, frame_seq)
        return max_frame_seq + 1

    def capture_frame(self, *, reason: str) -> dict[str, Any]:
        image = self.capture_callable()
        if not isinstance(image, Image.Image):
            raise TypeError("capture_callable must return a PIL.Image.Image")

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        if self.render_vlm_artifacts:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)

        capture_wall_ms = int(round(float(self.clock()) * 1000.0))
        frame_seq = self._frame_seq
        self._frame_seq += 1
        frame_id = build_frame_id(capture_wall_ms=capture_wall_ms, frame_seq=frame_seq)
        filename = build_frame_filename(capture_wall_ms=capture_wall_ms, frame_seq=frame_seq)

        raw_path = self.raw_dir / filename
        temp_path = raw_path.with_suffix(raw_path.suffix + ".tmp")
        image.save(temp_path, format="PNG")
        temp_path.replace(raw_path)

        artifact_path: Path | None = None
        artifact_metadata: dict[str, Any] | None = None
        if self.render_vlm_artifacts:
            artifact_path = self.artifact_dir / f"{raw_path.stem}{DEFAULT_ARTIFACT_SUFFIX}"
            artifact_metadata = render_vlm_ready_frame(raw_path, artifact_path)

        manifest_entry = {
            "schema_version": "v2",
            "frame_id": frame_id,
            "capture_wall_ms": capture_wall_ms,
            "frame_seq": frame_seq,
            "channel": self.channel,
            "layout_id": self.layout_id,
            "image_path": str(raw_path.resolve()),
            "width": int(image.width),
            "height": int(image.height),
            "source_session_id": self.session_id,
        }
        validate_instance(manifest_entry, "vision_frame_manifest_entry")
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")
            handle.flush()

        index_entry = {
            "frame_id": frame_id,
            "capture_wall_ms": capture_wall_ms,
            "raw_image_path": str(raw_path.resolve()),
            "artifact_image_path": str(artifact_path.resolve()) if artifact_path is not None else None,
            "capture_reason": str(reason),
            "session_id": self.session_id,
            "channel": self.channel,
            "layout_id": self.layout_id,
        }
        with self.capture_index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(index_entry, ensure_ascii=False) + "\n")
            handle.flush()

        return {
            **manifest_entry,
            "capture_reason": str(reason),
            "artifact_image_path": index_entry["artifact_image_path"],
            "artifact_metadata": artifact_metadata,
            "timestamp": datetime.fromtimestamp(capture_wall_ms / 1000.0, tz=timezone.utc).isoformat(),
        }


class CaptureEventSource:
    def poll(self) -> dict[str, Any] | None:
        raise NotImplementedError

    def close(self) -> None:
        return


class GlobalHelpEventSource(CaptureEventSource):
    def __init__(
        self,
        *,
        hotkey: str,
        modifiers: str = "",
        cooldown_ms: int = DEFAULT_GLOBAL_HELP_COOLDOWN_MS,
        trigger: WindowsGlobalHelpTrigger | None = None,
    ) -> None:
        self.hotkey = str(hotkey).strip()
        self.modifiers = str(modifiers).strip()
        self.cooldown_ms = int(cooldown_ms)
        self._trigger = (
            trigger
            if trigger is not None
            else WindowsGlobalHelpTrigger(
                hotkey=self.hotkey,
                modifiers=self.modifiers,
                cooldown_ms=self.cooldown_ms,
            )
        )
        self.hotkey_label = getattr(self._trigger, "hotkey_label", self.hotkey)
        self._started = False

    def start(self) -> None:
        self._trigger.start()
        self._started = True

    def poll(self) -> dict[str, Any] | None:
        if hasattr(self._trigger, "poll") and self._trigger.poll():
            return {"reason": "help", "source": "global_hotkey", "hotkey_label": self.hotkey_label}
        return None

    def request_stop(self) -> None:
        if hasattr(self._trigger, "request_stop"):
            self._trigger.request_stop()

    def close(self) -> None:
        if not self._started:
            return
        self._trigger.close()


class UdpEventSource(CaptureEventSource):
    def __init__(self, listener: UdpCaptureRequestListener) -> None:
        self.listener = listener

    def poll(self) -> dict[str, Any] | None:
        payload = self.listener.poll()
        if payload is None:
            return None
        return {
            "reason": str(payload.get("reason") or "help"),
            "source": "udp",
        }

    def close(self) -> None:
        self.listener.close()


class HelpTriggeredDatasetCapture:
    def __init__(
        self,
        *,
        writer: DatasetFrameWriter,
        event_sources: list[CaptureEventSource] | None = None,
        capture_fps: float = DEFAULT_CAPTURE_FPS,
        start_on_launch: bool = False,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        print_frame: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        normalized_capture_fps = float(capture_fps)
        if normalized_capture_fps <= 0:
            raise ValueError("capture_fps must be > 0")
        self.writer = writer
        self.event_sources = list(event_sources or [])
        self.capture_interval_s = 1.0 / normalized_capture_fps
        self.start_on_launch = bool(start_on_launch)
        self.monotonic = monotonic
        self.sleep = sleep
        self.print_frame = print_frame

    def run(
        self,
        *,
        duration_s: float = 0.0,
        max_frames: int = 0,
        idle_sleep_s: float = 0.02,
    ) -> DatasetCaptureStats:
        if duration_s < 0:
            raise ValueError("duration_s must be >= 0")
        if max_frames < 0:
            raise ValueError("max_frames must be >= 0")

        frames_written = 0
        help_start_captures = 0
        interval_captures = 0
        last_frame_id: str | None = None
        start = self.monotonic()
        active = self.start_on_launch
        next_capture = start if active else None
        started = active

        while True:
            now = self.monotonic()
            if duration_s > 0 and (now - start) >= duration_s:
                break

            request = None
            for event_source in self.event_sources:
                request = event_source.poll()
                if request is not None:
                    break
            reason: str | None = None
            if request is not None and not active:
                active = True
                started = True
                reason = "help_start"
            elif active and next_capture is not None and now >= next_capture:
                reason = "interval"

            if reason is not None:
                frame = self.writer.capture_frame(reason=reason)
                last_frame_id = str(frame["frame_id"])
                if self.print_frame is not None:
                    self.print_frame(frame)
                frames_written += 1
                if reason == "help_start":
                    help_start_captures += 1
                else:
                    interval_captures += 1
                next_capture = self.monotonic() + self.capture_interval_s
                if max_frames > 0 and frames_written >= max_frames:
                    break
                continue

            if max_frames > 0 and frames_written >= max_frames:
                break
            self.sleep(max(0.0, idle_sleep_s))

        return DatasetCaptureStats(
            frames_written=frames_written,
            help_start_captures=help_start_captures,
            interval_captures=interval_captures,
            started=started,
            last_frame_id=last_frame_id,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture a VLM fine-tuning dataset after the first help trigger.")
    parser.add_argument("--session-id", required=True, help="Dataset capture session id.")
    parser.add_argument("--saved-games-dir", default=None, help="Saved Games/<variant> root containing SimTutorConfig.lua.")
    parser.add_argument("--config-path", default=None, help="Optional explicit SimTutorConfig.lua path.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Dataset capture output root.")
    parser.add_argument("--fps", type=float, default=DEFAULT_CAPTURE_FPS, help="Continuous capture FPS after help starts.")
    parser.add_argument(
        "--global-help-hotkey",
        default="X1",
        help="Optional Windows global help hotkey: ESC, F1-F24, X1/MOUSE4, X2/MOUSE5. Empty disables.",
    )
    parser.add_argument("--global-help-modifiers", default="", help="Optional modifiers joined by +, e.g. Ctrl+Shift.")
    parser.add_argument(
        "--global-help-cooldown-ms",
        type=int,
        default=DEFAULT_GLOBAL_HELP_COOLDOWN_MS,
        help="Debounce window for the Windows global help trigger in milliseconds.",
    )
    parser.add_argument("--help-trigger-host", default=DEFAULT_VISION_CAPTURE_TRIGGER_HOST, help="UDP host for help-trigger capture requests.")
    parser.add_argument("--help-trigger-port", type=int, default=DEFAULT_VISION_CAPTURE_TRIGGER_PORT, help="UDP port for help-trigger capture requests.")
    parser.add_argument("--trigger-timeout", type=float, default=0.1, help="UDP receive timeout in seconds.")
    parser.add_argument("--screen-width", type=int, default=None, help="Optional override for capture width.")
    parser.add_argument("--screen-height", type=int, default=None, help="Optional override for capture height.")
    parser.add_argument("--channel", default=None, help="Optional override for vision.channel.")
    parser.add_argument("--layout-id", default=None, help="Optional override for vision.layout_id.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional max frame count before exit.")
    parser.add_argument("--duration-s", type=float, default=0.0, help="Optional run duration in seconds.")
    parser.add_argument("--start-on-launch", action="store_true", help="Start capturing immediately without waiting for help.")
    parser.add_argument(
        "--manual-plan",
        default="",
        help="Optional manual capture plan name. Currently supports fa18c_run003_v3_220.",
    )
    parser.add_argument(
        "--manual-plan-start-seq",
        type=int,
        default=1,
        help="Start a manual plan from this 1-based seq number. Useful after repairing or resuming a session.",
    )
    parser.add_argument(
        "--no-render-vlm-artifacts",
        action="store_true",
        help="Disable VLM-ready artifact rendering and only keep raw frames.",
    )
    return parser


def _resolve_runtime_config(args: argparse.Namespace) -> tuple[DatasetCaptureConfig, Path]:
    if not args.config_path and not args.saved_games_dir:
        raise ValueError("either --saved-games-dir or --config-path must be provided")
    config_path = (
        Path(args.config_path).expanduser()
        if args.config_path
        else _default_config_path(Path(args.saved_games_dir).expanduser())
    )
    loaded = load_sidecar_config(config_path)
    screen_width = int(args.screen_width) if args.screen_width is not None else loaded.capture_width
    screen_height = int(args.screen_height) if args.screen_height is not None else loaded.capture_height
    if screen_width <= 0:
        raise ValueError("screen_width must be > 0")
    if screen_height <= 0:
        raise ValueError("screen_height must be > 0")
    return (
        DatasetCaptureConfig(
            session_id=str(args.session_id).strip(),
            output_root=Path(args.output_root).expanduser(),
            channel=str(args.channel).strip() if args.channel else loaded.channel,
            layout_id=str(args.layout_id).strip() if args.layout_id else loaded.layout_id,
            screen_width=screen_width,
            screen_height=screen_height,
            render_vlm_artifacts=not bool(args.no_render_vlm_artifacts),
        ),
        config_path,
    )


def _print_frame_event(frame: dict[str, Any]) -> None:
    print(
        "[CAPTURE_VLM_DATASET] frame="
        + json.dumps(
            {
                "frame_id": frame.get("frame_id"),
                "reason": frame.get("capture_reason"),
                "raw_image_path": frame.get("image_path"),
                "artifact_image_path": frame.get("artifact_image_path"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def main() -> int:
    args = build_arg_parser().parse_args()
    listener: UdpCaptureRequestListener | None = None
    global_hotkey_source: GlobalHelpEventSource | None = None
    try:
        config, config_path = _resolve_runtime_config(args)
        if str(args.manual_plan).strip() and not config.render_vlm_artifacts:
            raise ValueError("--manual-plan requires VLM artifact rendering; remove --no-render-vlm-artifacts")
        writer = DatasetFrameWriter(
            output_root=config.output_root,
            session_id=config.session_id,
            channel=config.channel,
            layout_id=config.layout_id,
            render_vlm_artifacts=config.render_vlm_artifacts,
            capture_callable=lambda: capture_screen_region(
                width=config.screen_width,
                height=config.screen_height,
            ),
        )
        event_sources: list[CaptureEventSource] = []
        if str(args.global_help_hotkey).strip():
            if sys.platform != "win32":
                raise RuntimeError("--global-help-hotkey is only supported on Windows")
            global_hotkey_source = GlobalHelpEventSource(
                hotkey=args.global_help_hotkey,
                modifiers=args.global_help_modifiers,
                cooldown_ms=args.global_help_cooldown_ms,
            )
            global_hotkey_source.start()
            event_sources.append(global_hotkey_source)
        if int(args.help_trigger_port) > 0:
            listener = UdpCaptureRequestListener(
                session_id=config.session_id,
                host=args.help_trigger_host,
                port=args.help_trigger_port,
                timeout=args.trigger_timeout,
            )
            event_sources.append(UdpEventSource(listener))
        try:
            if str(args.manual_plan).strip():
                stats = run_manual_plan_capture(
                    writer=writer,
                    plan=_resolve_manual_plan(args.manual_plan),
                    event_sources=event_sources,
                    start_seq=args.manual_plan_start_seq,
                )
                print(
                    f"[CAPTURE_VLM_DATASET] stats="
                    f"{json.dumps(stats.to_dict(), ensure_ascii=False, sort_keys=True)}"
                )
                return 0

            runner = HelpTriggeredDatasetCapture(
                writer=writer,
                event_sources=event_sources,
                capture_fps=args.fps,
                start_on_launch=args.start_on_launch,
                print_frame=_print_frame_event,
            )
            print(
                "[CAPTURE_VLM_DATASET] config="
                + json.dumps(
                    {
                        "config_path": str(config_path),
                        "output_root": str(config.output_root),
                        "session_id": config.session_id,
                        "channel": config.channel,
                        "layout_id": config.layout_id,
                        "screen_width": config.screen_width,
                        "screen_height": config.screen_height,
                        "fps": args.fps,
                        "global_help_hotkey": getattr(global_hotkey_source, "hotkey_label", None),
                        "trigger_host": args.help_trigger_host,
                        "trigger_port": listener.bound_port if listener is not None else 0,
                        "render_vlm_artifacts": config.render_vlm_artifacts,
                        "start_on_launch": bool(args.start_on_launch),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            if global_hotkey_source is not None:
                print(
                    f"[CAPTURE_VLM_DATASET] press {global_hotkey_source.hotkey_label} once to start capture, "
                    "then press Ctrl+C to stop"
                )
            stats = runner.run(duration_s=args.duration_s, max_frames=args.max_frames)
        finally:
            if listener is not None:
                listener.close()
            if global_hotkey_source is not None:
                global_hotkey_source.close()
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[CAPTURE_VLM_DATASET] failed: {type(exc).__name__}: {exc}")
        return 1

    print(f"[CAPTURE_VLM_DATASET] stats={json.dumps(stats.to_dict(), ensure_ascii=False, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
