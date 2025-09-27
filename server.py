from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
from fastapi import FastAPI, HTTPException, UploadFile, File, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse, JSONResponse

from core.pipeline import PipelineConfig
from core.sdk import create_session, process_frame, close_session, register_event_callback, set_should_stop


LOGGER = logging.getLogger("api.server")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s")


# ------------------------------
# Models
# ------------------------------

class CreateSessionRequest(BaseModel):
    mode: str = Field("rtsp", description="會話模式：'rtsp' 或 'frame'")
    rtsp_url: Optional[str] = Field(None, description="RTSP 串流來源 URL（mode=rtsp 時必填）")
    # 可選：覆蓋 PipelineConfig 的部分參數
    output_dir: str = Field("outputs", description="輸出目錄（k_img / video）")
    roi_poly_norm: Optional[List[Tuple[float, float]]] = None
    neg_roi_poly_norm: Optional[List[Tuple[float, float]]] = None
    iou_threshold: float = 0.0
    trigger_frames: int = 5
    precheck_enabled: bool = True
    precheck_scale: float = 2.0
    crop_k2: bool = True
    crop_margin_ratio: float = 0.2
    crop_min_size: int = 160
    crop_max_size: int = 512
    crop_square: bool = False
    vlm_backend: str = "qwen"
    yolo_weights: Optional[str] = None
    tracker_cfg: Optional[str] = None
    track_conf: float = 0.35
    no_track: bool = False
    draw_sam: bool = False
    # 若未提供 roi_poly_norm，且要用 SAM，可提供點與標籤
    points: List[Tuple[int, int]] = Field(default_factory=list)
    labels: List[int] = Field(default_factory=list)


class CreateSessionResponse(BaseModel):
    session_id: str


class SessionStatus(BaseModel):
    session_id: str
    rtsp_url: str
    running: bool
    last_error: Optional[str] = None
    last_event_index: int = 0
    created_ms: float


# ------------------------------
# Session Runtime Structures
# ------------------------------

@dataclass
class SessionState:
    session_id: str
    mode: str
    rtsp_url: Optional[str]
    cfg: PipelineConfig
    stop_event: threading.Event
    alert_queue: "queue.Queue[Dict[str, Any]]"
    thread: Optional[threading.Thread] = None
    handle: Any = None
    running: bool = False
    last_error: Optional[str] = None
    last_event_index: int = 0
    created_ms: float = 0.0


# 全域 session 管理
SESSIONS: Dict[str, SessionState] = {}


def _build_cfg(req: CreateSessionRequest) -> PipelineConfig:
    # input_path 在串流模式下不使用，保留空字串
    cfg = PipelineConfig(
        input_path="",
        output_dir=req.output_dir,
        points=list(req.points or []),
        labels=list(req.labels or []),
        iou_threshold=req.iou_threshold,
        trigger_frames=req.trigger_frames,
        leave_patience=30,
        yolo_weights=req.yolo_weights,
        tracker_cfg=req.tracker_cfg,
        track_conf=req.track_conf,
        no_track=req.no_track,
        crop_k2=req.crop_k2,
        crop_margin_ratio=req.crop_margin_ratio,
        crop_min_size=req.crop_min_size,
        crop_max_size=req.crop_max_size,
        crop_square=req.crop_square,
        vlm_backend=req.vlm_backend,
        draw_sam=req.draw_sam,
        roi_poly_norm=list(req.roi_poly_norm or []) if req.roi_poly_norm else None,
        neg_roi_poly_norm=list(req.neg_roi_poly_norm or []) if req.neg_roi_poly_norm else None,
        neg_iou_threshold=0.0,
        precheck_enabled=req.precheck_enabled,
        precheck_scale=req.precheck_scale,
        person_mask=False,
        mask_mode="keep",
    )
    return cfg


def _safe_put(queue_obj, item):
    import queue
    try:
        queue_obj.put_nowait(item)
    except queue.Full:
        try:
            _ = queue_obj.get_nowait()
        except Exception:
            pass
        try:
            queue_obj.put_nowait(item)
        except Exception:
            pass


def _rtsp_worker(state: SessionState) -> None:
    import queue
    session_id = state.session_id
    rtsp_url = state.rtsp_url
    cfg = state.cfg

    backoff_s = 1.0
    max_backoff_s = 10.0

    handle = None
    cap = None

    try:
        handle = create_session(cfg, enable_video_output=False)
        state.handle = handle

        def stopper() -> bool:
            return state.stop_event.is_set()

        set_should_stop(handle, stopper)

        def on_evt(evt):
            try:
                state.last_event_index = int(getattr(evt, "event_index", 0) or 0)
            except Exception:
                pass

        register_event_callback(handle, on_evt)

        def open_capture():
            c = cv2.VideoCapture(rtsp_url)
            if not c.isOpened():
                return None
            return c

        cap = open_capture()
        if cap is None:
            raise RuntimeError("Failed to open RTSP stream")

        state.running = True
        LOGGER.info("[%s] RTSP worker started", session_id)

        fps_hint = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        last_alive = time.time()

        while not state.stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                # 重連
                try:
                    cap.release()
                except Exception:
                    pass
                time.sleep(backoff_s)
                backoff_s = min(max_backoff_s, backoff_s * 1.5)
                cap = open_capture()
                if cap is None:
                    continue
                fps_hint = float(cap.get(cv2.CAP_PROP_FPS) or fps_hint or 25.0)
                continue

            backoff_s = 1.0

            out = process_frame(handle, frame, ts_ms=float(time.time() * 1000.0), fps_hint=fps_hint)
            try:
                for ev in (out.events or []):
                    et = str(getattr(ev, "type", ""))
                    if et == "vlm_decision" or et == "precheck_passed":
                        payload = {
                            "type": et,
                            "session_id": session_id,
                            "event_index": int(getattr(ev, "event_index", 0) or 0),
                            "decision": str(getattr(ev, "decision", "")),
                            "summary": str(getattr(ev, "summary", "")),
                            "k1k2_path": getattr(ev, "k1k2_path", None),
                            "debug_dir": getattr(ev, "debug_dir", None),
                            "timestamp_ms": float(getattr(ev, "timestamp_ms", time.time() * 1000.0)),
                        }
                        _safe_put(state.alert_queue, payload)
            except Exception:
                pass

            # 週期性小心跳，避免長時間無事件時隊列完全空
            if (time.time() - last_alive) > 10.0:
                last_alive = time.time()
                _safe_put(state.alert_queue, {"type": "_heartbeat", "ts": last_alive})

    except Exception as e:
        state.last_error = f"{type(e).__name__}: {e}"
        LOGGER.exception("[%s] worker crashed", session_id)
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            if handle is not None:
                close_session(handle)
        except Exception:
            pass
        state.running = False
        LOGGER.info("[%s] RTSP worker stopped", session_id)


# ------------------------------
# FastAPI App
# ------------------------------

app = FastAPI(title="SmartEyes Streaming API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/sessions", response_model=CreateSessionResponse)
def create_session_api(req: CreateSessionRequest):
    import queue
    session_id = uuid.uuid4().hex[:12]
    if session_id in SESSIONS:
        raise HTTPException(status_code=409, detail="session id conflict; retry")

    cfg = _build_cfg(req)
    os.makedirs(cfg.output_dir, exist_ok=True)

    st = SessionState(
        session_id=session_id,
        mode=str(req.mode or "rtsp"),
        rtsp_url=req.rtsp_url,
        cfg=cfg,
        stop_event=threading.Event(),
        alert_queue=queue.Queue(maxsize=100),
        created_ms=time.time() * 1000.0,
    )
    SESSIONS[session_id] = st

    # 僅在 RTSP 模式下啟動背景工作
    if st.mode == "rtsp":
        if not req.rtsp_url:
            raise HTTPException(status_code=400, detail="rtsp_url is required for rtsp mode")
        th = threading.Thread(target=_rtsp_worker, args=(st,), name=f"rtsp-{session_id}", daemon=True)
        st.thread = th
        th.start()
    else:
        # frame 模式：先初始化一次 pipeline，無背景執行緒
        try:
            handle = create_session(cfg, enable_video_output=False)
            st.handle = handle
            st.running = True
            # 註冊事件回呼：當 VLM=YES 時直接推入佇列（讓 SSE 立即收到）
            def on_evt(evt):
                try:
                    st.last_event_index = int(getattr(evt, "event_index", 0) or 0)
                    et = str(getattr(evt, "type", ""))
                    if et in ("vlm_decision", "precheck_passed"):
                        payload = {
                            "type": et,
                            "session_id": session_id,
                            "event_index": int(getattr(evt, "event_index", 0) or 0),
                            "decision": str(getattr(evt, "decision", "")),
                            "summary": str(getattr(evt, "summary", "")),
                            "k1k2_path": getattr(evt, "k1k2_path", None),
                            "debug_dir": getattr(evt, "debug_dir", None),
                            "timestamp_ms": float(getattr(evt, "timestamp_ms", time.time() * 1000.0)),
                        }
                        _safe_put(st.alert_queue, payload)
                except Exception:
                    pass
            register_event_callback(handle, on_evt)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"failed to init frame-mode pipeline: {e}")

    return CreateSessionResponse(session_id=session_id)


@app.delete("/sessions/{session_id}")
def delete_session_api(session_id: str):
    st = SESSIONS.get(session_id)
    if st is None:
        return JSONResponse({"ok": True, "message": "already deleted"})
    try:
        st.stop_event.set()
        if st.thread is not None and st.thread.is_alive():
            st.thread.join(timeout=5.0)
    except Exception:
        pass
    try:
        if st.handle is not None:
            close_session(st.handle)
    except Exception:
        pass
    SESSIONS.pop(session_id, None)
    return {"ok": True}


@app.get("/sessions/{session_id}/status", response_model=SessionStatus)
def get_status_api(session_id: str):
    st = SESSIONS.get(session_id)
    if st is None:
        raise HTTPException(status_code=404, detail="session not found")
    return SessionStatus(
        session_id=session_id,
        rtsp_url=st.rtsp_url or "",
        running=bool(st.running),
        last_error=st.last_error,
        last_event_index=int(st.last_event_index),
        created_ms=float(st.created_ms),
    )


@app.get("/sessions/{session_id}/alert/stream")
async def sse_alert_stream(session_id: str):
    st = SESSIONS.get(session_id)
    if st is None:
        raise HTTPException(status_code=404, detail="session not found")

    async def event_generator():
        # 將 thread-safe queue 與 async 世界銜接：使用 to_thread 阻塞等待
        last_keepalive = time.time()
        while True:
            if st.stop_event.is_set() and not st.running:
                break
            try:
                item = await asyncio.to_thread(st.alert_queue.get, True, 1.0)
            except Exception:
                item = None

            now = time.time()
            if item is None:
                # 週期性 keepalive，避免 Proxy 關閉連線
                if (now - last_keepalive) >= 10.0:
                    last_keepalive = now
                    yield ": keepalive\n\n"
                continue

            # 忽略內部心跳
            if isinstance(item, dict) and item.get("type") == "_heartbeat":
                if (now - last_keepalive) >= 10.0:
                    last_keepalive = now
                    yield ": keepalive\n\n"
                continue

            try:
                data = json.dumps(item, ensure_ascii=False)
            except Exception:
                continue
            yield f"event: alert\ndata: {data}\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)


@app.post("/sessions/{session_id}/frame")
async def upload_frame_api(session_id: str, file: UploadFile = File(...), verbose: int = Query(0, description="If 1, always return JSON diagnostics (no_trigger/precheck_passed/vlm_no/vlm_yes)")):
    st = SESSIONS.get(session_id)
    if st is None:
        raise HTTPException(status_code=404, detail="session not found")
    if st.mode != "frame":
        raise HTTPException(status_code=400, detail="session is not in frame mode")
    if st.handle is None:
        raise HTTPException(status_code=500, detail="pipeline not initialized")

    import numpy as np
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")

    # 解碼成 BGR 影像
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="failed to decode image")

    out = process_frame(st.handle, img, ts_ms=float(time.time() * 1000.0))

    # 篩選 YES 事件（立即回傳）
    try:
        diag_payload = None
        for ev in (out.events or []):
            et = str(getattr(ev, "type", ""))
            if et in ("vlm_decision", "precheck_passed"):
                payload = {
                    "type": et,
                    "session_id": session_id,
                    "event_index": int(getattr(ev, "event_index", 0) or 0),
                    "decision": str(getattr(ev, "decision", "")),
                    "summary": str(getattr(ev, "summary", "")),
                    "k1k2_path": getattr(ev, "k1k2_path", None),
                    "debug_dir": getattr(ev, "debug_dir", None),
                    "timestamp_ms": float(getattr(ev, "timestamp_ms", time.time() * 1000.0)),
                }
                _safe_put(st.alert_queue, payload)
                if et == "vlm_decision" and payload.get("decision") == "yes":
                    return JSONResponse(payload)
                # 優先保留決策訊息作為診斷，其次 precheck
                if diag_payload is None or (payload.get("type") == "vlm_decision"):
                    diag_payload = payload
    except Exception:
        pass

    # 若要求診斷：回傳詳細 JSON
    if verbose:
        if diag_payload is not None:
            # 若是 VLM 決策但非 YES，標示為 vlm_no/unsure
            if diag_payload.get("type") == "vlm_decision" and diag_payload.get("decision") != "yes":
                return JSONResponse(diag_payload)
            # 或只有 precheck_passed
            return JSONResponse(diag_payload)
        # 無任何事件
        return JSONResponse({"type": "none", "session_id": session_id, "timestamp_ms": float(time.time() * 1000.0)})

    # 沒有 YES 事件且未要求診斷 → 204 No Content
    return Response(status_code=204)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)


