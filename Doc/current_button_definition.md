# Current Button Definition (MVP)

## Purpose
In DCS-BIOS, we receive state deltas instead of direct button click events.  
For v0.3 MVP, "current button" is approximated by mapping recent DCS-BIOS delta keys to UI targets.

## Data Source
- Mapping file: `packs/fa18c_startup/bios_to_ui.yaml`
- UI allowlist source: `packs/fa18c_startup/ui_map.yaml` (`cockpit_elements` keys)

## MVP Rule
Given one delta frame:
- Input: `delta = {bios_key: value, ...}`
- For each `bios_key`, lookup mapped UI targets from `bios_to_ui.yaml`
- Output: ordered list of UI targets for this frame

If no key is mapped, return an empty list.

## Ordering and Stability
Output order is deterministic:
1. Preserve input `delta` key order
2. Preserve target order defined in YAML per key
3. Deduplicate while preserving first occurrence

This ensures stable "recent button" lists for prompt construction and replay.

## One-to-Many Mapping
One bios key can map to multiple UI targets:

```yaml
mappings:
  COMM1_CHAN:
    targets:
      - ufc_comm1_channel_selector_rotate
      - ufc_comm1_channel_selector_pull
```

## Validation and Safety
During load:
- Every mapped target must exist in `ui_map.yaml`
- Invalid mapping fails fast (`BiosUiMapError`)

This keeps overlay target resolution inside the allowlist boundary.

## ST-011 Implementation Notes

- Module: `adapters/recent_actions.py`
- Ring buffer: `RecentDeltaRingBuffer(window_s, max_items)`
  - `window_s`: keep only frames in recent N seconds
  - `max_items`: hard cap frame count to avoid prompt explosion
- Projection API:
  - `project_recent_ui_targets(recent_deltas, bios_to_ui)`
  - Output is recency-first and stable:
    1. Newer frame first
    2. Keep key order inside each frame
    3. Keep target order in mapping
    4. Deduplicate by first occurrence
- Prompt helper:
  - `build_recent_button_signal(...)` returns:
    - `current_button`
    - `recent_buttons`
