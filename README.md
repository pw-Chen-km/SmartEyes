## Visual Monitoring Streaming SDK (High‑Level Guide)

Your app sends frames (images) one-by-one; the SDK maintains internal state, detects interactions within an ROI, performs precheck on K1/K2 frames, and asynchronously requests a VLM decision. You can integrate without any HTTP server. Just call functions.

Key files:
- `core/sdk.py` (SDK entry points)
- `core/pipeline.py` (configuration type and underlying logic)

---

### How It Works (Architecture)
1) Session init
   - Loads models via `Orchestrator` (SAM, VLM, optional YOLOE tracker)
   - Builds a binary ROI mask from either:
     - Fixed polygon ROI (`roi_poly_norm`, recommended for fastest deployment), or
     - SAM prompts (`points`/`labels`) on the first frame
2) Per-frame processing
   - Optional person tracking (YOLOE)
   - Computes overlap between detections and the ROI mask
   - Drives a state machine: idle → contacting (capture K1) → post_leaving (sample K2) → dispatch
   - Runs K1/K2 precheck; if passed, dispatches to VLM asynchronously
3) Outputs
   - Returns live panel status and an annotated frame (overlay). If you only need YES/NO decisions, ignore the overlay
   - Emits events via callback: status heartbeat, precheck_passed, vlm_decision
   - Writes debugging artifacts under `outputs/<stream_id>/k_img/`; optionally writes video under `outputs/<stream_id>/video/`

---

### Quick Start (Minimal Code)

```python
from core.sdk import create_session, process_frame, register_event_callback, close_session
from core.pipeline import PipelineConfig
import cv2, time

cfg = PipelineConfig(
    input_path="",                      # ignored in streaming mode
    output_dir="/tmp/outputs",
    roi_poly_norm=[(0.2,0.4),(0.8,0.4),(0.8,0.95),(0.2,0.95)],  # QUICK ROI (normalized)
    points=[], labels=[],                # no SAM when roi_poly_norm is set
    trigger_frames=5,
    crop_k2=True, draw_sam=False,
    vlm_backend="qwen",
)

handle = create_session(cfg, enable_video_output=False)  # decisions only

def on_event(evt):
    # evt.type: "status" | "precheck_passed" | "vlm_decision"
    print("EVENT:", evt)

register_event_callback(handle, on_event)

cap = cv2.VideoCapture("/path/to/video.mp4")
while True:
    ok, frame = cap.read()
    if not ok: break
    _ = process_frame(handle, frame, ts_ms=time.time()*1000, fps_hint=30)

cap.release()
summary = close_session(handle)
print(summary)
```

---

### SDK Surface (Functions)

All functions live in `core/sdk.py`.

1) `create_session(cfg: PipelineConfig, enable_video_output: bool=False) -> SessionHandle`
   - Input: `cfg` (see configuration below); `enable_video_output` toggles annotated video writing
   - Output: `SessionHandle`
   - Effect: initializes models and ROI/SAM mask

2) `process_frame(handle: SessionHandle, frame_bgr, ts_ms: float|None=None, fps_hint: float|None=None) -> ProcessOutput`
   - Input: `frame_bgr` (BGR uint8 HxWx3), optional `ts_ms`, optional `fps_hint` (affects video writing only)
   - Output: `ProcessOutput { panel_mode, panel_message, vlm_text, events, overlay_bgr }`
   - Effect: advances state machine; may enqueue VLM jobs; returns any new events for this frame

3) `register_event_callback(handle, cb: Callable[[Event], None]|None)`
   - Subscribe/unsubscribe for events. Event types:
     - `status` (heartbeat every ~0.5s)
     - `precheck_passed`
     - `vlm_decision` (contains `decision` in {"yes","no","unsure"})

4) `set_should_stop(handle, fn: Callable[[], bool]|None)`
   - Optional stopper hook (reserved for cooperative stop)

5) `close_session(handle) -> dict`
   - Output: `{ "output_video_path": str|None, "event_count": int, "debug_dir": str|None }`

---

### The Fastest Way to Provide ROI (Recommended)

Use a fixed polygon with normalized coordinates (no SAM cost):

```python
cfg = PipelineConfig(
  output_dir="/tmp/outputs",
  roi_poly_norm=[(0.20,0.40),(0.80,0.40),(0.80,0.95),(0.20,0.95)],
  points=[], labels=[],
)
```

Alternatively, use SAM prompts (pixel points):

```python
cfg = PipelineConfig(
  output_dir="/tmp/outputs",
  roi_poly_norm=None,
  points=[(280, 259)], labels=[1],  # 1=positive, 0=negative
  draw_sam=True,
)
```

Tips:
- Prefer `roi_poly_norm` to get up and running quickly, then refine
- Turn on `draw_sam` to visualize only when using SAM

---

### Get YES/NO Decisions Only (No Video Output)

If your system only needs decisions and will raise alerts downstream:

1) Create session with `enable_video_output=False`
2) Subscribe to `vlm_decision` events
3) Use `evt.decision` ("yes" | "no" | "unsure") and ignore frames

```python
from core.sdk import create_session, process_frame, register_event_callback, close_session
from core.pipeline import PipelineConfig
import cv2, time

cfg = PipelineConfig(
    input_path="",
    output_dir="/tmp/outputs",
    roi_poly_norm=[(0.2,0.4),(0.8,0.4),(0.8,0.95),(0.2,0.95)],
    points=[], labels=[],
    trigger_frames=5, precheck_enabled=True, crop_k2=True,
    vlm_backend="qwen",
)

handle = create_session(cfg, enable_video_output=False)

seen = set()
def on_event(evt):
    if evt.type == "vlm_decision" and evt.event_index not in seen:
        seen.add(evt.event_index)
        if evt.decision == "yes":
            print("ALERT: YES")
        elif evt.decision == "no":
            print("SAFE: NO")
        else:
            print("UNSURE")

register_event_callback(handle, on_event)

cap = cv2.VideoCapture("/path/to/video.mp4")
while True:
    ok, frame = cap.read()
    if not ok: break
    _ = process_frame(handle, frame, ts_ms=time.time()*1000, fps_hint=30)

cap.release()
_ = close_session(handle)
```

---

### Key Configuration (What Matters Most)

ROI/Masking
- `roi_poly_norm`: list[(x_norm, y_norm)], normalized polygon; if set (>=3 points), SAM is skipped
- `points`, `labels`: SAM prompts (pixels); used only when `roi_poly_norm` is None; 1=positive, 0=negative
- `draw_sam`: overlay the initial SAM mask on outputs (for debugging)

Triggering & Precheck
- `iou_threshold`: overlap threshold between detection and ROI (start with 0.0)
- `trigger_frames`: consecutive frames to confirm contact (3–7 typical; 5 at 30FPS ≈ 0.17s)
- `leave_patience`: frames to wait after leaving (e.g., 15 at 30FPS ≈ 0.5s)
- `precheck_enabled`: gate before VLM; `precheck_scale`: crop scale for precheck (1.5–2.0 typical)

Cropping
- `crop_k2`: enable K1/K2 ROI-centered crops (recommended)
- `crop_margin_ratio`: 0.15–0.25 typical; `crop_min_size`/`crop_max_size`: adapt to resolution; `crop_square`: prefer square

Tracking (optional)
- `yolo_weights`: YOLOE weights (or env `YOLOE_WEIGHTS`); if missing, tracker is disabled
- `tracker_cfg`: e.g., `bytetrack.yaml`; if None, SDK omits the arg to avoid errors
- `track_conf`: 0.25–0.40 typical; `no_track`: force off; `track_interval`: run every N frames

Outputs & Misc
- `output_dir`: outputs root (`k_img/`, `video/`)
- `vlm_backend`: e.g., "qwen"

---

### Parameter Tuning Guide (With Examples)

Ready-to-use configurations:

Example A — Production starter (Fixed ROI):
```python
cfg = PipelineConfig(
    input_path="",
    output_dir="/tmp/outputs",
    roi_poly_norm=[(0.20,0.40),(0.80,0.40),(0.80,0.95),(0.20,0.95)],
    points=[], labels=[],
    iou_threshold=0.0,
    trigger_frames=5,
    leave_patience=15,
    precheck_enabled=True, precheck_scale=2.0,
    crop_k2=True, crop_margin_ratio=0.2, crop_min_size=160, crop_max_size=512,
    yolo_weights=None, tracker_cfg=None, track_conf=0.35, no_track=False,
    vlm_backend="qwen", draw_sam=False,
)
```

Example B — SAM prompts (pixel points):
```python
cfg = PipelineConfig(
    input_path="",
    output_dir="/tmp/outputs",
    roi_poly_norm=None,
    points=[(280, 259)], labels=[1],
    iou_threshold=0.0,
    trigger_frames=5,
    leave_patience=15,
    precheck_enabled=True, precheck_scale=2.0,
    crop_k2=True, crop_margin_ratio=0.2, crop_min_size=160, crop_max_size=512,
    yolo_weights=None, tracker_cfg=None, track_conf=0.35, no_track=False,
    vlm_backend="qwen", draw_sam=True,
)
```

Cheat sheet:
- Too insensitive? ↓`trigger_frames` (5→3), keep `iou_threshold=0.0` initially
- Too many false positives? ↑`trigger_frames` (5→7), ↑`iou_threshold` (0.0→0.05), keep `precheck_enabled=True`
- VLM unstable? ↑`crop_margin_ratio` (0.15→0.25), ↑`precheck_scale` (1.5→2.0), ensure K1/K2 clearly show hand vs ROI
- Limited compute? Prefer `roi_poly_norm`; ↑`track_interval` or set `no_track=True`; disable video output

---

### What Gets Written to Disk

- `outputs/<stream_id>/k_img/`: K1/K2 collages (`*_k1k2.jpg`), precheck logs (`precheck_log_*.json`), `events_summary.jsonl`
- `outputs/<stream_id>/video/`: annotated MP4 if `enable_video_output=True`

## Visual Monitoring Streaming SDK (High-Level Guide)

Your app sends frames (images) one-by-one; the SDK maintains internal state, detects interactions within an ROI, performs precheck on K1/K2 frames, and asynchronously requests a VLM decision.

You can integrate without any HTTP server. Just call functions.

Key files:
- `core/sdk.py` (SDK entry points)
- `core/pipeline.py` (original batch pipeline; reference implementation)

---

### What the SDK Does (Architecture Overview)
1) Session init
   - Loads models via `Orchestrator` (SAM, VLM, optional YOLOE tracker)
   - Prepares a binary ROI mask in either of two ways:
     - Fixed polygon ROI (recommended for quick deployment)
     - SAM from sparse point prompts (positive/negative points)
2) Streaming processing (per frame)
   - Optional person tracking (YOLOE)
   - Calculates overlap between detections and the ROI mask
   - Drives a state machine: idle → contacting (capture K1) → post_leaving (sample K2) → dispatch
   - Precheck on K1/K2 (both must say "yes" to continue)
   - Dispatches K1/K2 to VLM (asynchronous); when finished, a decision event is emitted
3) Outputs
   - On each call, returns the live panel status and an annotated image (overlay). If you only need YES/NO decisions, you can ignore the overlay entirely.
   - Writes debugging artifacts under `outputs/<stream_id>/k_img/` (K1/K2 collages, precheck logs)
   - Optionally writes a video with overlays under `outputs/<stream_id>/video/`

High-level flow: your app calls `process_frame(frame)` in a loop; the SDK returns current status and emits events when interactions are detected and when the VLM decision is ready.

---

### The Fastest Way to Deploy ROI (Strongly Recommended)

Use a fixed polygon ROI with normalized coordinates. This avoids SAM cost and is the easiest to set up.

- Provide `roi_poly_norm` in `PipelineConfig`: a list of `(x_norm, y_norm)` tuples where each value is in [0,1]. The SDK converts them to pixel coordinates on the first frame and builds a binary mask.
- Example: a full-frame ROI:
```python
roi_poly_norm = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
```
- Example: a cabinet region (roughly centered horizontally, bottom 60% of the frame):
```python
roi_poly_norm = [(0.2, 0.4), (0.8, 0.4), (0.8, 0.95), (0.2, 0.95)]
```

When `roi_poly_norm` is set (with 3+ points), the SDK skips SAM entirely and uses this mask for triggering and cropping. This is the quickest path to production.

If you prefer SAM instead, set points/labels (pixels) and leave `roi_poly_norm=None`.

---

### Quick Start (Minimal Code)

```python
from visual_monitoring_poc.core.sdk import create_session, process_frame, register_event_callback, close_session
from visual_monitoring_poc.core.pipeline import PipelineConfig
import cv2, time

cfg = PipelineConfig(
    input_path="",                      # ignored in streaming mode
    output_dir="/tmp/outputs",
    roi_poly_norm=[(0.2, 0.4), (0.8, 0.4), (0.8, 0.95), (0.2, 0.95)],  # QUICK ROI
    points=[], labels=[],                # no SAM when roi_poly_norm is set
    trigger_frames=5,
    crop_k2=True, draw_sam=False,
    vlm_backend="qwen",
)

handle = create_session(cfg, enable_video_output=True)

def on_event(evt):
    # evt.type: "status" | "precheck_passed" | "vlm_decision"
    print("EVENT:", evt)

register_event_callback(handle, on_event)

cap = cv2.VideoCapture("/path/to/video.mp4")
while True:
    ok, frame = cap.read()
    if not ok: break
    out = process_frame(handle, frame, ts_ms=time.time()*1000, fps_hint=30)
    # out.overlay_bgr: ready-to-display annotated frame

cap.release()
summary = close_session(handle)
print(summary)
```

---

### SDK Surface (Functions)

All functions are defined in `core/sdk.py`.

1) `create_session(cfg: PipelineConfig, enable_video_output: bool=False) -> SessionHandle`
   - Input:
     - `cfg`: pipeline configuration (see below)
     - `enable_video_output`: if True, writes annotated video under `outputs/<stream_id>/video/`
   - Output: `SessionHandle`
   - Effect: initializes `Orchestrator`, ROI/SAM mask, and optional tracker

2) `process_frame(handle: SessionHandle, frame_bgr: np.ndarray, ts_ms: float|None=None, fps_hint: float|None=None) -> ProcessOutput`
   - Input:
     - `frame_bgr`: BGR image, uint8, shape HxWx3
     - `ts_ms`: optional timestamp (ms)
     - `fps_hint`: affects only video writing FPS
   - Output: `ProcessOutput` with:
     - `panel_mode`: "none" | "green" | "orange" | "red"
     - `panel_message`: status message
     - `vlm_text`: latest VLM summary (string)
     - `events`: list of `Event` produced on this frame (often 0 or 1)
     - `overlay_bgr`: annotated frame (BGR)
   - Effect: advances the per-stream state machine, enqueues VLM tasks when ready

3) `register_event_callback(handle: SessionHandle, cb: Callable[[Event], None]|None) -> None`
   - Subscribe to events. Pass `None` to unsubscribe.
   - Event types:
     - `status`: periodic heartbeat (~0.5s) with panel color/message/VLM text
     - `precheck_passed`: emitted right after K1/K2 precheck passes
     - `vlm_decision`: emitted when VLM returns a decision; includes `decision` and `summary`

4) `set_should_stop(handle: SessionHandle, fn: Callable[[], bool]|None) -> None`
   - Registers a stopper function; currently a hook for future cooperative stop logic.

5) `close_session(handle: SessionHandle) -> dict`
   - Output: `{ "output_video_path": str|None, "event_count": int, "debug_dir": str|None }`
   - Effect: releases resources and finalizes outputs

---

### Get YES/NO Decisions Only (No Video Output)

If your system only needs decisions and will raise alerts downstream, do this:

1) Create the session with `enable_video_output=False`
2) Subscribe to `vlm_decision` via the event callback
3) In your callback, use `evt.decision` ("yes" | "no" | "unsure") and ignore frames entirely

Example:
```python
from visual_monitoring_poc.core.sdk import create_session, process_frame, register_event_callback, close_session
from visual_monitoring_poc.core.pipeline import PipelineConfig
import cv2, time

cfg = PipelineConfig(
    input_path="",
    output_dir="/tmp/outputs",
    roi_poly_norm=[(0.2,0.4),(0.8,0.4),(0.8,0.95),(0.2,0.95)],
    points=[], labels=[],
    trigger_frames=5, precheck_enabled=True, crop_k2=True,
    vlm_backend="qwen",
)

handle = create_session(cfg, enable_video_output=False)

seen = set()
def on_event(evt):
    if evt.type == "vlm_decision" and evt.event_index not in seen:
        seen.add(evt.event_index)
        if evt.decision == "yes":
            print("ALERT: YES")
        elif evt.decision == "no":
            print("SAFE: NO")
        else:
            print("UNSURE")

register_event_callback(handle, on_event)

cap = cv2.VideoCapture("/path/to/video.mp4")
while True:
    ok, frame = cap.read()
    if not ok: break
    _ = process_frame(handle, frame, ts_ms=time.time()*1000, fps_hint=30)  # ignore overlays

cap.release()
_ = close_session(handle)
```

Notes:
- Use a fixed ROI polygon (`roi_poly_norm`) for the fastest path to production
- You can still enable SAM with `points`/`labels` if needed; `draw_sam=True` helps visual debugging
- Decisions: "yes" → take action; "no" → safe; "unsure" → treat as safe or send to human review

---

### How to Provide ROI (Two Options)

Option A — Fixed ROI polygon (recommended):
```python
cfg = PipelineConfig(
  output_dir="/tmp/outputs",
  roi_poly_norm=[(0.22, 0.08), (0.22, 0.32), (0.42, 0.47), (0.73, 0.47), (0.94, 0.38), (0.95, 0.08)],
  points=[], labels=[],
)
```
- Use 3 or more normalized points in clockwise or counterclockwise order.
- Easiest to deploy, no SAM compute, stable across frames.

Option B — SAM point prompts:
```python
cfg = PipelineConfig(
  output_dir="/tmp/outputs",
  roi_poly_norm=None,
  points=[(280, 259)], labels=[1],  # pixel coordinates; 1=positive, 0=negative
  draw_sam=True,
)
```
- Provide pixel coordinates for a few positive/negative points describing the ROI object/area.
- The SDK runs SAM once on the first frame to initialize a binary mask.

Tips:
- Prefer Option A for speed and robustness.
- If you need to hand-tune, start with Option A to validate end-to-end, then experiment with SAM.

---

### Key Configuration (What Matters Most)

Within `PipelineConfig` (imported from `core/pipeline.py`):

- ROI/Masking
  - `roi_poly_norm`: list[(x_norm, y_norm)], normalized ROI polygon (if set, SAM is skipped)
  - `points`, `labels`: SAM prompts (pixels); used only when `roi_poly_norm` is None
  - `draw_sam`: overlay the initial mask on outputs (visual debugging)

- Triggering & Precheck
  - `iou_threshold`: how much overlap between detection and ROI triggers contact
  - `trigger_frames`: consecutive frames to confirm contact (smaller = more sensitive)
  - `precheck_enabled`: run K1/K2 precheck gate
  - `precheck_scale`: enlarge crops for precheck only (does not affect dispatch crops)

- Cropping
  - `crop_k2`, `crop_margin_ratio`, `crop_min_size`, `crop_max_size`, `crop_square`

- Tracking (optional)
  - `yolo_weights`: YOLOE checkpoint (or set env `YOLOE_WEIGHTS`); if missing, tracker is disabled
  - `tracker_cfg`: tracker config like `bytetrack.yaml`; if None, the SDK omits the parameter to avoid errors
  - `track_conf`: detection confidence threshold
  - `no_track`: disable tracking entirely

- Outputs & Misc
  - `output_dir`: outputs are saved to `outputs/<stream_id>/{k_img,video}`
  - `vlm_backend`: VLM backend (e.g., "qwen")

---

### What to Expect on Disk

- `outputs/<stream_id>/k_img/`
  - `*_k1k2.jpg`: K1/K2 collages for each event
  - `precheck_log_*.json`: precheck summaries (pass/fail)
  - `events_summary.jsonl`: compact per-event VLM decisions

- `outputs/<stream_id>/video/`
  - `*_out.mp4`: annotated video if `enable_video_output=True`

---



