from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Any
import os
import time
import logging

import cv2
import numpy as np

from .pipeline import PipelineConfig
from .orchestrator import Orchestrator
from .trigger import InteractionTrigger
from .buffer import ReasoningKeyframeBuffer
from .crop_utils import compute_overlap_bbox, crop_by_box
from .utils import yolo_polys_from_result, overlap_ratio_poly_with_mask
from .output import (
    ensure_dir,
    make_video_writer,
    put_text_bottom_right,
    draw_status_panel,
    overlay_sam_mask,
    save_kframes,
    save_precheck_log,
    save_precheck_images,
    append_event_summary,
)


LOGGER = logging.getLogger("core.sdk")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s")


@dataclass
class Event:
    type: str
    event_index: int
    decision: str
    summary: str
    k1k2_path: Optional[str]
    debug_dir: Optional[str]
    timestamp_ms: float


@dataclass
class ProcessOutput:
    panel_mode: str
    panel_message: str
    vlm_text: str
    events: List[Event]
    overlay_bgr: Optional[np.ndarray]


class StreamingPipeline:
    def __init__(self, cfg: PipelineConfig, enable_video_output: bool = False) -> None:
        self.cfg = cfg
        self.enable_video_output = enable_video_output

        self.orch: Optional[Orchestrator] = None

        # 面板狀態
        self.panel_mode: str = "none"
        self.panel_message: str = ""
        self.current_vlm_text: str = ""
        self.vlm_has_output: bool = False

        # 追蹤模型（可選）
        self.track_model = None

        # 初始 SAM 遮罩
        self._initial_sam_mask_bin: Optional[np.ndarray] = None
        self.use_fixed_roi_mask: bool = bool(self.cfg.roi_poly_norm and len(self.cfg.roi_poly_norm) >= 3)
        # 負向 ROI 遮罩
        self._neg_roi_mask_bin: Optional[np.ndarray] = None
        self.use_fixed_neg_roi_mask: bool = bool(getattr(self.cfg, "neg_roi_poly_norm", None) and len(self.cfg.neg_roi_poly_norm or []) >= 3)

        # 事件與 UI 回呼
        self._ui_event_cb: Optional[Callable[[Dict[str, Any]], None]] = None
        self._last_event_index: int = 0
        self._last_ui_emit_ts: float = 0.0
        self._emitted_events: List[Event] = []  # 累積由 callback 產生的事件

        # 影片輸出
        self._writer = None
        self._writer_size: Optional[Tuple[int, int]] = None
        self._fps_hint: float = 30.0
        self._out_path: Optional[str] = None
        self._output_dir_ready: bool = False

        # k1/k2 裁切資訊與對應幀
        self._k1_crop_box: Optional[Tuple[int, int, int, int]] = None
        self._k2_crop_box: Optional[Tuple[int, int, int, int]] = None
        self._last_frame_bgr: Optional[np.ndarray] = None
        self._last_frame_bgr_k1: Optional[np.ndarray] = None
        self._last_frame_bgr_k2: Optional[np.ndarray] = None
        self._first_frame_bgr_k2: Optional[np.ndarray] = None

        # 事件彙總
        self._debug_dir: Optional[str] = None
        self._current_event_index: int = 1
        self._stem: str = "stream"

        # 單幀狀態機
        self._trigger = InteractionTrigger(
            iou_threshold=self.cfg.iou_threshold,
            consecutive_frames=self.cfg.trigger_frames,
            neg_iou_threshold=float(getattr(self.cfg, "neg_iou_threshold", 0.0)),
        )
        self._state: str = "idle"
        self._contact_active: bool = False
        self._kbuf = ReasoningKeyframeBuffer()
        self._k1_collected: bool = False
        self._k2_sample_frames: List[int] = [1, 10,25]
        self._k2_sample_count: int = 0
        self._post_leaving_frame_counter: int = 0
        self._k1_crop_center: Optional[Tuple[int, int]] = None
        self._person_missing_frames: int = 0
        self._person_missing_threshold: int = 10
        self._frame_index: int = 0
        # red 狀態維持幀數倒數（60 幀後自動恢復 monitoring）
        self._red_cooldown_frames: int = 0

        # precheck 暫存
        self._last_precheck: Dict[str, Any] = {}

        self._initialize_models()

    # --- 公開 API ---
    def on_event(self, cb: Optional[Callable[[Event], None]]) -> None:
        self._ui_event_cb = (lambda e: None) if cb is None else (lambda d: cb(self._event_from_dict(d)))

    def set_should_stop(self, fn: Optional[Callable[[], bool]]) -> None:
        self._should_stop = (lambda: False) if fn is None else fn

    def process(self, frame_bgr: np.ndarray, ts_ms: Optional[float] = None, fps_hint: Optional[float] = None) -> ProcessOutput:
        if ts_ms is None:
            ts_ms = float(time.time() * 1000.0)
        if isinstance(fps_hint, (int, float)) and fps_hint > 0:
            self._fps_hint = float(fps_hint)

        h, w = frame_bgr.shape[:2]
        width, height = int(w), int(h)
        self._frame_index += 1
        self._last_event_index = int(self._current_event_index)

        # 準備輸出目錄與 writer（首次幀）
        if not self._output_dir_ready:
            ensure_dir(self.cfg.output_dir)
            self._stem = self._stem or "stream"
            video_dir = os.path.join(self.cfg.output_dir, self._stem, "video")
            kimg_dir = os.path.join(self.cfg.output_dir, self._stem, "k_img")
            ensure_dir(video_dir)
            ensure_dir(kimg_dir)
            self._debug_dir = kimg_dir
            self._output_dir_ready = True
        if self.enable_video_output and self._writer is None:
            size = (width, height)
            out_path = os.path.join(self.cfg.output_dir, self._stem, "video", f"{self._stem}_stream_out.mp4")
            self._writer = make_video_writer(out_path, self._fps_hint, size)
            self._writer_size = size
            self._out_path = out_path

        # 第一次幀：準備 SAM 或固定 ROI mask
        if self._frame_index == 1:
            self._prepare_roi_or_sam(width, height, frame_bgr)

        # YOLOE 追蹤（可選）
        annotated_bgr = None
        r = None
        if self.track_model is not None:
            try:
                if self.cfg.tracker_cfg:
                    results = self.track_model.track(frame_bgr, persist=True, tracker=self.cfg.tracker_cfg, conf=self.cfg.track_conf)
                else:
                    results = self.track_model.track(frame_bgr, persist=True, conf=self.cfg.track_conf)
                r = results[0] if isinstance(results, (list, tuple)) and len(results) > 0 else results
                annotated_bgr = r.plot()
            except Exception:
                LOGGER.exception("YOLOE.track failed; skip annotate")

        # 人是否存在，用於 panel 紅色切換
        person_present = self._is_person_present_from_result(r)
        if not person_present:
            self._person_missing_frames += 1
        else:
            self._person_missing_frames = 0
        if self.panel_mode == "orange" and self._person_missing_frames >= self._person_missing_threshold:
            self._set_panel("red", "person leave with item")

        # SAM mask 當前影格
        sam_bin = self._current_sam_bin(width, height)

        max_overlap = 0.0
        best_poly = None
        polys = None
        if r is not None and sam_bin is not None:
            polys = yolo_polys_from_result(r)
            for poly in polys:
                ov = overlap_ratio_poly_with_mask(poly, sam_bin, width, height)
                if ov > max_overlap:
                    max_overlap = ov
                    best_poly = poly

        # 計算負向 ROI overlap（若有）
        neg_overlap = 0.0
        if self.use_fixed_neg_roi_mask and isinstance(self._neg_roi_mask_bin, np.ndarray) and r is not None:
            try:
                poly_list = polys if polys is not None else yolo_polys_from_result(r)
                for poly in poly_list:
                    ov2 = overlap_ratio_poly_with_mask(poly, self._neg_roi_mask_bin, width, height)
                    if ov2 > neg_overlap:
                        neg_overlap = ov2
            except Exception:
                neg_overlap = 0.0

        # 觸發器
        event_flag = self._trigger.update(max_overlap)
        self._step_state_machine(event_flag, frame_bgr, annotated_bgr, width, height, sam_bin, best_poly, polys, ts_ms, neg_overlap)

        # 覆繪與寫檔
        out_text = self.current_vlm_text or ""
        base_bgr = annotated_bgr if annotated_bgr is not None else frame_bgr
        if self.cfg.draw_sam and self._initial_sam_mask_bin is not None:
            try:
                base_bgr = overlay_sam_mask(base_bgr, self._initial_sam_mask_bin, color=(0, 255, 0), alpha=0.35)
            except Exception:
                pass
        base_bgr = draw_status_panel(base_bgr, self.panel_mode, self.panel_message)
        out_frame = put_text_bottom_right(cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB), out_text)
        out_frame_bgr = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)

        if self._writer is not None and self._writer_size is not None:
            try:
                self._writer.write(out_frame_bgr)
            except Exception:
                LOGGER.exception("writer.write failed")

        # 心跳事件節流（~0.5s）
        try:
            now_ts = time.time()
            if (now_ts - self._last_ui_emit_ts) >= 0.5:
                self._emit_ui_event(self.panel_mode, self.panel_message, self.current_vlm_text)
                self._last_ui_emit_ts = now_ts
        except Exception:
            pass

        # red 狀態維持 60 幀後自動回到 monitoring（none）
        try:
            if self.panel_mode == "red":
                if self._red_cooldown_frames > 0:
                    self._red_cooldown_frames -= 1
                else:
                    self._set_panel("none", "")
        except Exception:
            pass

        # 將累積 callback 事件打包返回，並清空暫存
        events = self._emitted_events
        self._emitted_events = []
        return ProcessOutput(
            panel_mode=self.panel_mode,
            panel_message=self.panel_message,
            vlm_text=self.current_vlm_text,
            events=events,
            overlay_bgr=out_frame_bgr,
        )

    def close(self) -> Dict[str, Any]:
        try:
            if self._writer is not None:
                try:
                    self._writer.release()
                except Exception:
                    pass
            if self.orch is not None:
                try:
                    if hasattr(self.orch, 'reasoning') and hasattr(self.orch.reasoning, '_model'):
                        del self.orch.reasoning._model
                    if hasattr(self.orch, 'perception') and hasattr(self.orch.perception, '_sam_model'):
                        del self.orch.perception._sam_model
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
        finally:
            self.orch = None
        return {
            "output_video_path": self._out_path,
            "event_count": int(max(0, self._current_event_index - 1)),
            "debug_dir": self._debug_dir,
        }

    # --- 內部：初始化與基礎工具 ---
    def _initialize_models(self) -> None:
        self.orch = Orchestrator(iou_threshold=self.cfg.iou_threshold, vlm_backend=self.cfg.vlm_backend)
        self.orch.initialize_models()
        self.orch.set_result_callback(self._result_callback)
        self.orch.set_prompts(visual_template_image=None, text_prompt="person")
        self._prepare_tracker()

    def _prepare_tracker(self) -> None:
        if self.cfg.no_track:
            LOGGER.info("tracking disabled by config.no_track")
            self.track_model = None
            return
        try:
            from ultralytics import YOLOE as UltralyticsYOLOE  # type: ignore
        except Exception:
            LOGGER.warning("Ultralytics YOLOE not available; fallback to simple tracking")
            self.track_model = None
            return
        try:
            m = UltralyticsYOLOE(self.cfg.yolo_weights or os.getenv("YOLOE_WEIGHTS", "yoloe-11l-seg.pt"))
            names = ["person", "bottle"]
            try:
                txt_pe = m.get_text_pe(names)
                m.set_classes(names, txt_pe)
            except Exception:
                LOGGER.info("YOLOE set_classes by text prompt skipped")
            self.track_model = m
            LOGGER.info("YOLOE tracker ready: weights=%s tracker=%s conf=%.2f", self.cfg.yolo_weights, self.cfg.tracker_cfg, self.cfg.track_conf)
        except Exception:
            LOGGER.exception("failed to initialize YOLOE tracker; fallback to simple tracking")
            self.track_model = None

    def _prepare_roi_or_sam(self, width: int, height: int, first_bgr: np.ndarray) -> None:
        try:
            # 若 roi_poly_norm 看起來是像素座標（任何值 > 1），在首次幀自動轉為 normalize
            try:
                if isinstance(self.cfg.roi_poly_norm, list) and len(self.cfg.roi_poly_norm) > 0:
                    has_pixel_like = False
                    for xn, yn in (self.cfg.roi_poly_norm or []):
                        if float(xn) > 1.0 or float(yn) > 1.0:
                            has_pixel_like = True
                            break
                    if has_pixel_like and width > 0 and height > 0:
                        norm_list = []
                        for xn, yn in (self.cfg.roi_poly_norm or []):
                            x_norm = max(0.0, min(1.0, float(xn) / float(width)))
                            y_norm = max(0.0, min(1.0, float(yn) / float(height)))
                            norm_list.append((x_norm, y_norm))
                        self.cfg.roi_poly_norm = norm_list
                        self.use_fixed_roi_mask = bool(len(self.cfg.roi_poly_norm) >= 3)
                        LOGGER.info("auto-normalized roi_poly_norm from pixel coordinates (%dx%d)", width, height)
            except Exception:
                pass

            if self.use_fixed_roi_mask and self.cfg.roi_poly_norm is not None:
                poly_px: List[Tuple[int, int]] = []
                for xn, yn in (self.cfg.roi_poly_norm or []):
                    x = int(round(float(xn) * float(width)))
                    y = int(round(float(yn) * float(height)))
                    x = max(0, min(width - 1, x))
                    y = max(0, min(height - 1, y))
                    poly_px.append((x, y))
                mask0 = np.zeros((height, width), dtype=np.uint8)
                if len(poly_px) >= 3:
                    pts = np.array(poly_px, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(mask0, [pts], 255)
                    self._initial_sam_mask_bin = (mask0 > 0).astype(np.uint8)
                    LOGGER.info("using fixed ROI polygon mask with %d points", len(poly_px))
                else:
                    LOGGER.warning("roi_poly_norm has <3 points; fallback to SAM")
                    self.use_fixed_roi_mask = False
            # 構建負向 ROI 遮罩（若有）
            if self.use_fixed_neg_roi_mask and getattr(self.cfg, "neg_roi_poly_norm", None) is not None:
                try:
                    # 若 neg_roi_poly_norm 看起來是像素座標（任何值 > 1），在首次幀自動轉為 normalize
                    try:
                        if isinstance(self.cfg.neg_roi_poly_norm, list) and len(self.cfg.neg_roi_poly_norm) > 0:
                            has_pixel_like2 = False
                            for xn, yn in (self.cfg.neg_roi_poly_norm or []):
                                if float(xn) > 1.0 or float(yn) > 1.0:
                                    has_pixel_like2 = True
                                    break
                            if has_pixel_like2 and width > 0 and height > 0:
                                norm_list2 = []
                                for xn, yn in (self.cfg.neg_roi_poly_norm or []):
                                    x_norm = max(0.0, min(1.0, float(xn) / float(width)))
                                    y_norm = max(0.0, min(1.0, float(yn) / float(height)))
                                    norm_list2.append((x_norm, y_norm))
                                self.cfg.neg_roi_poly_norm = norm_list2
                                self.use_fixed_neg_roi_mask = bool(len(self.cfg.neg_roi_poly_norm) >= 3)
                                LOGGER.info("auto-normalized neg_roi_poly_norm from pixel coordinates (%dx%d)", width, height)
                    except Exception:
                        pass

                    poly_px2: List[Tuple[int, int]] = []
                    for xn, yn in (self.cfg.neg_roi_poly_norm or []):
                        x = int(round(float(xn) * float(width)))
                        y = int(round(float(yn) * float(height)))
                        x = max(0, min(width - 1, x))
                        y = max(0, min(height - 1, y))
                        poly_px2.append((x, y))
                    mask1 = np.zeros((height, width), dtype=np.uint8)
                    if len(poly_px2) >= 3:
                        pts2 = np.array(poly_px2, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.fillPoly(mask1, [pts2], 255)
                        self._neg_roi_mask_bin = (mask1 > 0).astype(np.uint8)
                        LOGGER.info("using fixed NEG-ROI polygon mask with %d points", len(poly_px2))
                    else:
                        self._neg_roi_mask_bin = None
                        self.use_fixed_neg_roi_mask = False
                except Exception:
                    self._neg_roi_mask_bin = None
                    self.use_fixed_neg_roi_mask = False
            if not self.use_fixed_roi_mask and self.orch is not None:
                if len(self.cfg.points) > 0:
                    self.orch.set_sam_prompts(points=self.cfg.points, labels=(self.cfg.labels or [1] * len(self.cfg.points)), roi_box=None)
                    first_rgb = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB)
                    _ = self.orch.preview_sam(first_rgb)
                    try:
                        m0 = self.orch.perception.get_last_sam_mask()
                        if isinstance(m0, np.ndarray) and m0.size > 0:
                            if m0.ndim == 3:
                                m0 = m0[0]
                            if m0.shape[0] != height or m0.shape[1] != width:
                                m0 = cv2.resize(m0, (width, height), interpolation=cv2.INTER_NEAREST)
                            self._initial_sam_mask_bin = (m0 > 0).astype(np.uint8)
                    except Exception:
                        self._initial_sam_mask_bin = None
                else:
                    LOGGER.warning("no SAM points provided; mask will be None")
        except Exception:
            LOGGER.exception("SAM/ROI mask preparation failed")

    def _current_sam_bin(self, width: int, height: int) -> Optional[np.ndarray]:
        sam = None
        sam_bin = None
        if self.use_fixed_roi_mask and self._initial_sam_mask_bin is not None:
            sam_bin = self._initial_sam_mask_bin
        elif self.orch is not None:
            sam = self.orch.perception.get_last_sam_mask()
            if isinstance(sam, np.ndarray) and sam.size > 0:
                if sam.ndim == 3:
                    sam = sam[0]
                if sam.shape[0] != height or sam.shape[1] != width:
                    sam = cv2.resize(sam, (width, height), interpolation=cv2.INTER_NEAREST)
                sam_bin = (sam > 0).astype(np.uint8)
        return sam_bin

    def _is_person_present_from_result(self, r: Any) -> bool:
        try:
            if r is None:
                return False
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                return False
            cls = getattr(boxes, "cls", None)
            conf = getattr(boxes, "conf", None)
            try:
                if cls is not None:
                    cls_arr = cls.cpu().numpy() if hasattr(cls, "cpu") else np.asarray(cls)
                    if conf is not None:
                        conf_arr = conf.cpu().numpy() if hasattr(conf, "cpu") else np.asarray(conf)
                        mask = (conf_arr >= max(0.05, float(self.cfg.track_conf) * 0.5))
                        if mask.shape == cls_arr.shape:
                            cls_arr = cls_arr[mask]
                    if cls_arr.size > 0 and int(cls_arr.min()) <= 0 <= int(cls_arr.max()):
                        return bool((np.sum(cls_arr == 0)) > 0)
            except Exception:
                pass
            n = len(boxes) if hasattr(boxes, "__len__") else 0
            return n > 0
        except Exception:
            return False

    # --- 內部：狀態與事件 ---
    def _set_panel(self, mode: str, message: str) -> None:
        try:
            if mode not in ("orange", "green", "red", "none"):
                return
            self.panel_mode = mode
            self.panel_message = str(message or "")
            # 進入 red 時啟動 60 幀倒數；其他狀態則清除倒數
            try:
                if mode == "red":
                    self._red_cooldown_frames = 60
                else:
                    self._red_cooldown_frames = 0
            except Exception:
                pass
            try:
                self._emit_ui_event(color=mode, message=self.panel_message, vlm_text=self.current_vlm_text)
            except Exception:
                pass
        except Exception:
            pass

    def _result_callback(self, res: Dict[str, Any]) -> None:
        try:
            summary = (res or {}).get("summary") or ""
            self.current_vlm_text = str(summary)
            self.vlm_has_output = True
            LOGGER.info("VLM: %s", self.current_vlm_text)

            decision = self._parse_decision(self.current_vlm_text)
            if decision == "yes":
                self._set_panel("orange", "Item taken (VLM)")
            elif decision == "no":
                self._set_panel("green", "SAFE")

            try:
                from .output import annotate_image_file
                k1k2_path = (res or {}).get("k1k2_path") or None
                if isinstance(k1k2_path, str) and os.path.isfile(k1k2_path):
                    color = (0, 165, 255) if decision == "yes" else ((0, 200, 0) if decision == "no" else (255, 0, 0))
                    label = (decision or "").upper() if decision else "UNSURE"
                    try:
                        annotate_image_file(k1k2_path, label, color=color)
                    except Exception:
                        pass
            except Exception:
                pass

            if self._debug_dir and self._current_event_index > 0:
                try:
                    append_event_summary(self._debug_dir, self._current_event_index, decision, summary)
                except Exception:
                    LOGGER.exception("Failed to write event summary")

            # 將 VLM 決策轉為事件同步回傳
            evt = Event(
                type="vlm_decision",
                event_index=int(self._current_event_index),
                decision=decision,
                summary=str(summary),
                k1k2_path=str((res or {}).get("k1k2_path") or ""),
                debug_dir=str(self._debug_dir or ""),
                timestamp_ms=float(time.time() * 1000.0),
            )
            self._emitted_events.append(evt)
            try:
                if self._ui_event_cb is not None:
                    self._ui_event_cb({
                        "color": self.panel_mode,
                        "message": self.panel_message,
                        "vlm_text": self.current_vlm_text,
                        "event_index": int(self._current_event_index),
                        "frame_index": int(self._frame_index),
                        "timestamp_ms": float(time.time() * 1000.0),
                    })
            except Exception:
                pass
        except Exception:
            LOGGER.exception("result callback error")

    @staticmethod
    def _parse_decision(text: str) -> str:
        try:
            s = (text or "").strip().lower()
            token = s.split()[0] if s else ""
            if token in ("yes", "y", "true"):
                return "yes"
            if token in ("no", "n", "false"):
                return "no"
            if token in ("unsure", "unknown", "maybe"):
                return "unsure"
        except Exception:
            pass
        return ""

    def _emit_ui_event(self, color: str, message: str, vlm_text: str) -> None:
        try:
            if self._ui_event_cb is None:
                return
            evt = {
                "color": str(color or "none"),
                "message": str(message or ""),
                "vlm_text": str(vlm_text or ""),
                "event_index": int(self._last_event_index),
                "frame_index": int(self._frame_index),
                "timestamp_ms": float(time.time() * 1000.0),
            }
            self._ui_event_cb(evt)
        except Exception:
            pass

    def _event_from_dict(self, d: Dict[str, Any]) -> Event:
        return Event(
            type="status",
            event_index=int(d.get("event_index", 0)),
            decision="",
            summary=str(d.get("vlm_text", "")),
            k1k2_path=None,
            debug_dir=None,
            timestamp_ms=float(d.get("timestamp_ms", time.time() * 1000.0)),
        )

    # --- 內部：狀態機步進 ---
    def _step_state_machine(self, event_flag: bool, frame_bgr: np.ndarray, annotated_bgr: Optional[np.ndarray], width: int, height: int, sam_bin: Optional[np.ndarray], best_poly: Optional[np.ndarray], polys: Optional[List[np.ndarray]], ts_ms: float, neg_overlap: float) -> None:
        # idle -> contacting
        if self._state == "idle":
            if event_flag and not self._contact_active:
                self._contact_active = True
                self._kbuf.reset()
                self._state = "contacting"
            # 繼續輸出面板即可
            return

        # contacting：收 k1，等待離開
        if self._state == "contacting":
            # 若負向 ROI 觸發，取消事件
            try:
                if self._trigger.is_negative_violated(neg_overlap):
                    self._kbuf.reset()
                    self._state = "idle"
                    self._contact_active = False
                    self._k1_collected = False
                    self._trigger.reset()
                    self._person_missing_frames = 0
                    self._k2_sample_count = 0
                    self._k2_crop_box = None
                    self._current_event_index += 1
                    return
            except Exception:
                pass
            if not self._k1_collected:
                frame_with_sam = frame_bgr
                if sam_bin is not None:
                    try:
                        frame_with_sam = overlay_sam_mask(frame_bgr, sam_bin, color=(0, 255, 0), alpha=0.35)
                    except Exception:
                        pass
                self._kbuf.add("k1", frame_with_sam)
                try:
                    if self.cfg.crop_k2 and sam_bin is not None:
                        yolo_poly_list = [best_poly] if (best_poly is not None) else (polys or [])
                        bbox = compute_overlap_bbox(sam_bin, yolo_poly_list, width, height)
                        if bbox is not None:
                            bx1, by1, bx2, by2 = bbox
                            cx = (bx1 + bx2) // 2
                            cy = (by1 + by2) // 2
                            base_size = max(bx2 - bx1, by2 - by1)
                            margin_px = int(min(width, height) * self.cfg.crop_margin_ratio)
                            cw = min(self.cfg.crop_max_size, max(self.cfg.crop_min_size, base_size + margin_px * 2))
                            ch = cw if self.cfg.crop_square else min(self.cfg.crop_max_size, max(self.cfg.crop_min_size, base_size + margin_px * 2))
                            x1 = max(0, cx - cw // 2)
                            y1 = max(0, cy - ch // 2)
                            x2 = min(width, x1 + cw)
                            y2 = min(height, y1 + ch)
                            if x2 > x1 and y2 > y1:
                                eb = (x1, y1, x2, y2)
                                cropped = crop_by_box(frame_with_sam, eb)
                                self._k1_crop_box = eb
                                self._last_frame_bgr = frame_bgr.copy()
                                self._last_frame_bgr_k1 = frame_bgr.copy()
                                self._kbuf.replace_last("k1", cropped)
                                self._k1_crop_center = (cx, cy)
                except Exception:
                    pass
                self._k1_collected = True

            if not event_flag and self._contact_active:
                self._contact_active = False
                self._k2_sample_count = 0
                try:
                    if self.cfg.crop_k2:
                        if self._k1_crop_box is not None:
                            x1, y1, x2, y2 = self._k1_crop_box
                            x1 = max(0, min(x1, width))
                            x2 = max(0, min(x2, width))
                            y1 = max(0, min(y1, height))
                            y2 = max(0, min(y2, height))
                            if x2 > x1 and y2 > y1:
                                self._k2_crop_box = (x1, y1, x2, y2)
                            else:
                                self._k2_crop_box = None
                        elif self._k1_crop_center is not None:
                            cx, cy = self._k1_crop_center
                            base_size = 100
                            margin_px = int(min(width, height) * self.cfg.crop_margin_ratio)
                            cw = min(self.cfg.crop_max_size, max(self.cfg.crop_min_size, base_size + margin_px * 2))
                            ch = cw if self.cfg.crop_square else min(self.cfg.crop_max_size, max(self.cfg.crop_min_size, base_size + margin_px * 2))
                            x1 = max(0, cx - cw // 2)
                            y1 = max(0, cy - ch // 2)
                            x2 = min(width, x1 + cw)
                            y2 = min(height, y1 + ch)
                            if x2 > x1 and y2 > y1:
                                self._k2_crop_box = (x1, y1, x2, y2)
                            else:
                                self._k2_crop_box = None
                    else:
                        self._k2_crop_box = None
                except Exception:
                    self._k2_crop_box = None
                self._state = "post_leaving"
                self._post_leaving_frame_counter = 0
            return

        # post_leaving：取 k2 樣本幀
        if self._state == "post_leaving":
            # 在 K2 取樣前，若負向 ROI 觸發則取消事件
            try:
                if self._trigger.is_negative_violated(neg_overlap):
                    self._kbuf.reset()
                    self._state = "idle"
                    self._contact_active = False
                    self._k1_collected = False
                    self._trigger.reset()
                    self._person_missing_frames = 0
                    self._k2_sample_count = 0
                    self._k2_crop_box = None
                    self._current_event_index += 1
                    return
            except Exception:
                pass
            # 修正非法取樣幀位，至少從第 1 幀開始
            try:
                self._k2_sample_frames = [max(1, int(n)) for n in (self._k2_sample_frames or [1, 2])]
            except Exception:
                self._k2_sample_frames = [1, 2]

            self._post_leaving_frame_counter += 1
            if self._k2_sample_count < len(self._k2_sample_frames) and self._post_leaving_frame_counter == self._k2_sample_frames[self._k2_sample_count]:
                try:
                    frame_to_add = frame_bgr
                    if self.cfg.crop_k2 and self._k2_crop_box is not None:
                        eb = self._k2_crop_box
                        # 若是最後一針 K2，依 precheck_scale 放大一次（僅用於此次裁切，不覆寫 _k2_crop_box）
                        try:
                            if (self._k2_sample_count + 1) == len(self._k2_sample_frames):
                                s = float(getattr(self.cfg, "precheck_scale", 1.2))
                                eb = self._expand_box(eb, s, width, height)
                        except Exception:
                            pass
                        cropped = crop_by_box(frame_bgr, eb)
                        self._last_frame_bgr = frame_bgr.copy() if self._last_frame_bgr is None else self._last_frame_bgr
                        if self._k2_sample_count == 0:
                            self._first_frame_bgr_k2 = frame_bgr.copy()
                        self._last_frame_bgr_k2 = frame_bgr.copy()
                        frame_to_add = cropped
                    self._kbuf.add("k2", frame_to_add)
                    self._k2_sample_count += 1
                except Exception:
                    pass
            if self._k2_sample_count >= len(self._k2_sample_frames):
                self._state = "dispatch"
            return

        # dispatch：precheck 與投遞 VLM
        if self._state == "dispatch":
            frames_for_vlm = self._kbuf.collect(max_k1=1, max_k2=3, max_k3=0)

            try:
                frames_for_vlm = self._build_clean_kframes(frames_for_vlm)
            except Exception:
                pass

            try:
                k1_img = frames_for_vlm[0] if len(frames_for_vlm) > 0 else None
                k2_img = frames_for_vlm[1] if len(frames_for_vlm) > 1 else None
                pre_k1 = k1_img
                pre_k2 = k2_img
                try:
                    if self._k1_crop_box is not None and isinstance(self._last_frame_bgr_k1, np.ndarray):
                        eb1 = self._expand_box(self._k1_crop_box, float(getattr(self.cfg, "precheck_scale", 1.2)), width, height)
                        pre_k1 = crop_by_box(self._last_frame_bgr_k1, eb1)
                except Exception:
                    pass
                try:
                    if self._k2_crop_box is not None and isinstance(self._last_frame_bgr_k2, np.ndarray):
                        base_k2 = self._first_frame_bgr_k2 if isinstance(self._first_frame_bgr_k2, np.ndarray) else self._last_frame_bgr_k2
                        if isinstance(base_k2, np.ndarray):
                            eb2 = self._expand_box(self._k2_crop_box, float(getattr(self.cfg, "precheck_scale", 1.2)), width, height)
                            pre_k2 = crop_by_box(base_k2, eb2)
                except Exception:
                    pass

                pre_ok = self._precheck_k1k2(pre_k1, pre_k2)
                try:
                    pc = getattr(self, "_last_precheck", None)
                    if isinstance(pc, dict):
                        _ = save_precheck_log(self._debug_dir, self._current_event_index,
                                              str(pc.get("k1_summary", "")), str(pc.get("k1_decision", "")),
                                              str(pc.get("k2_summary", "")), str(pc.get("k2_decision", "")))
                        _ = save_precheck_images(self._debug_dir, self._current_event_index,
                                                 pc.get("k1_img"), pc.get("k2_img"))
                except Exception:
                    pass

                if not pre_ok:
                    try:
                        frames_for_fail = []
                        if k1_img is not None:
                            frames_for_fail.append(k1_img)
                        if k2_img is not None:
                            frames_for_fail.append(k2_img)
                        if len(frames_for_fail) >= 2:
                            save_kframes(self._debug_dir, self._stem, self._current_event_index, frames_for_fail)
                            from .output import annotate_image_file
                            k1k2_path_fail = os.path.join(self._debug_dir or "", f"{self._stem}_e{self._current_event_index}_k1k2.jpg")
                            if os.path.isfile(k1k2_path_fail):
                                try:
                                    annotate_image_file(k1k2_path_fail, "Filtered", color=(0, 0, 255))
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    self._reset_after_event()
                    return
            except Exception:
                self._reset_after_event()
                return

            try:
                save_kframes(self._debug_dir, self._stem, self._current_event_index, frames_for_vlm)
            except Exception:
                LOGGER.exception("Failed to save k1/k2 debug images")

            prompt = (
                "k1: hand touches ROI. "
                "k2: hand leaves ROI ( consecutive frames after leaving). "
                " check if the hand reaching into ROI and takes a new item from ROI (Observed from k2 frames). "
                "If yes, answer YES. "
                "If not, answer NO. "
                "If unsure, answer UNSURE. "
                "Output one word only."
            )

            try:
                if self.orch is not None:
                    k1k2_path = os.path.join(self._debug_dir or "", f"{self._stem}_e{self._current_event_index}_k1k2.jpg")
                    self.orch._event_queue.put_nowait({
                        "frames": frames_for_vlm,
                        "prompt": prompt,
                        "event_index": int(self._current_event_index),
                        "debug_dir": str(self._debug_dir or ""),
                        "k1k2_path": k1k2_path,
                    })
                    # 立即發出一個 precheck 通過事件
                    self._emitted_events.append(Event(
                        type="precheck_passed",
                        event_index=int(self._current_event_index),
                        decision="",
                        summary="",
                        k1k2_path=k1k2_path,
                        debug_dir=str(self._debug_dir or ""),
                        timestamp_ms=float(time.time() * 1000.0),
                    ))
            except Exception:
                LOGGER.exception("enqueue VLM task failed")

            self._reset_after_event()

    def _reset_after_event(self) -> None:
        self._kbuf.reset()
        self._state = "idle"
        self._contact_active = False
        self._k1_collected = False
        self._trigger.reset()
        self._person_missing_frames = 0
        self._k2_sample_count = 0
        self._k2_crop_box = None
        self._current_event_index += 1
        # 清理對應幀暫存
        self._last_frame_bgr_k1 = None
        self._last_frame_bgr_k2 = None
        self._first_frame_bgr_k2 = None

    # --- 內部：工具方法 ---
    def _build_clean_kframes(self, frames: List[Optional[np.ndarray]]) -> List[Optional[np.ndarray]]:
        try:
            clean_frames = list(frames)
            try:
                if self._k1_crop_box is not None and isinstance(self._last_frame_bgr_k1, np.ndarray):
                    k1_clean = crop_by_box(self._last_frame_bgr_k1, self._k1_crop_box)
                    if len(clean_frames) >= 1:
                        clean_frames[0] = k1_clean
            except Exception:
                pass
            try:
                if self._k2_crop_box is not None and isinstance(self._last_frame_bgr_k2, np.ndarray) and len(clean_frames) > 1:
                    k2_clean = crop_by_box(self._last_frame_bgr_k2, self._k2_crop_box)
                    clean_frames[1] = k2_clean
            except Exception:
                pass
            return clean_frames
        except Exception:
            return frames

    def _expand_box(self, box_xyxy: Tuple[int, int, int, int], scale: float, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
        try:
            x1, y1, x2, y2 = box_xyxy
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            cx = x1 + bw / 2.0
            cy = y1 + bh / 2.0
            s = max(1.0, float(scale))
            new_w = int(round(bw * s))
            new_h = int(round(bh * s))
            nx1 = int(round(cx - new_w / 2.0))
            ny1 = int(round(cy - new_h / 2.0))
            nx2 = nx1 + new_w
            ny2 = ny1 + new_h
            nx1 = max(0, min(nx1, frame_w - 1))
            ny1 = max(0, min(ny1, frame_h - 1))
            nx2 = max(0, min(nx2, frame_w))
            ny2 = max(0, min(ny2, frame_h))
            if nx2 <= nx1:
                nx2 = min(frame_w, nx1 + 1)
            if ny2 <= ny1:
                ny2 = min(frame_h, ny1 + 1)
            return (nx1, ny1, nx2, ny2)
        except Exception:
            return box_xyxy

    def _precheck_k1k2(self, k1_img: Optional[np.ndarray], k2_img: Optional[np.ndarray]) -> bool:
        try:
            if not self.cfg.precheck_enabled:
                return True
            if self.orch is None or getattr(self.orch, "reasoning", None) is None:
                return True
            if k1_img is None or k2_img is None:
                try:
                    self._last_precheck = {
                        "k1_summary": "",
                        "k1_decision": "",
                        "k2_summary": "",
                        "k2_decision": "",
                        "k1_img": k1_img,
                        "k2_img": k2_img,
                    }
                except Exception:
                    pass
                return False

            pre_prompt = (
                "任務：判斷這兩張圖像（K1=觸發瞬間，K2=解除瞬間）是否顯示「此人明確與ROI櫃子互動」（例如開門、拿取、放置）。"
                "條件："
                "- 必須看到手明確接觸或操作cabinet，且K1→K2能解釋為一次連續行為。"
                "- 只要沒看到手伸向roi都算no。"
                "- 不確定回答no。"
                "只允許輸出一個詞："
                "yes 或 no"
            )

            r = self.orch.reasoning.analyze_keyframes([k1_img, k2_img], prompt=pre_prompt)
            summary = str(r.get("summary", ""))
            overall = self._parse_decision(summary)

            try:
                self._last_precheck = {
                    "k1_summary": summary,
                    "k1_decision": overall,
                    "k2_summary": summary,
                    "k2_decision": overall,
                    "k1_img": k1_img,
                    "k2_img": k2_img,
                }
            except Exception:
                pass

            return (overall == "yes")
        except Exception:
            return False


# ---- 函式式 SDK 封裝 ----
class SessionHandle:
    def __init__(self, pipeline: StreamingPipeline) -> None:
        self._p = pipeline

    @property
    def pipeline(self) -> StreamingPipeline:
        return self._p


def create_session(cfg: PipelineConfig, enable_video_output: bool = False) -> SessionHandle:
    p = StreamingPipeline(cfg, enable_video_output=enable_video_output)
    return SessionHandle(p)


def process_frame(handle: SessionHandle, frame_bgr: np.ndarray, ts_ms: Optional[float] = None, fps_hint: Optional[float] = None) -> ProcessOutput:
    return handle.pipeline.process(frame_bgr, ts_ms=ts_ms, fps_hint=fps_hint)


def register_event_callback(handle: SessionHandle, cb: Optional[Callable[[Event], None]]) -> None:
    handle.pipeline.on_event(cb)


def set_should_stop(handle: SessionHandle, fn: Optional[Callable[[], bool]]) -> None:
    handle.pipeline.set_should_stop(fn)


def close_session(handle: SessionHandle) -> Dict[str, Any]:
    return handle.pipeline.close()


