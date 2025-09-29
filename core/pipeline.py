

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Callable
import os
import time
import logging
import json

import cv2
import numpy as np

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


LOGGER = logging.getLogger("core.pipeline")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s")


@dataclass
class PipelineConfig:
    input_path: str
    output_dir: str
    points: List[Tuple[int, int]]
    labels: List[int]
    iou_threshold: float = 0.0
    trigger_frames: int = 15
    leave_patience: int = 30
    show_window: bool = False
    yolo_weights: Optional[str] = None
    tracker_cfg: Optional[str] = None
    track_conf: float = 0.35
    batch_size: int = 0
    no_track: bool = False
    track_interval: int = 1
    crop_k2: bool = True
    crop_margin_ratio: float = 0.2
    crop_min_size: int = 300
    crop_max_size: int = 300
    crop_square: bool = False
    vlm_backend: str = "qwen"
    person_mask: bool = False
    mask_mode: str = "keep"
    draw_sam: bool = False
    roi_poly_norm: Optional[List[Tuple[float, float]]] = None
    # 負向 ROI：若在 K1 之後到 K2 之前與負向 ROI 有重疊，則取消此次事件
    neg_roi_poly_norm: Optional[List[Tuple[float, float]]] = None
    neg_iou_threshold: float = 0.0
    # K1/K2 前置檢查（以 VLM 單張判斷手是否進入/離開 ROI）
    precheck_enabled: bool = True
    # Precheck 影像裁切相對於 dispatch 放大比例（例如 1.2 表示放大 20%）
    precheck_scale: float = 2
    # Actor 追蹤（以 tracking ID 為主）離開判定設定
    actor_leave_enabled: bool = True
    actor_leave_patience: int = 3


class VisualMonitoringPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config
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
        self.use_fixed_neg_roi_mask: bool = bool(self.cfg.neg_roi_poly_norm and len(self.cfg.neg_roi_poly_norm) >= 3)

        # UI 事件回呼（事件級更新）
        self._ui_event_cb: Optional[Callable[[Dict[str, Any]], None]] = None
        self._last_event_index: int = 0
        # 外部停止旗標（可由 UI 設定）
        self._should_stop: Callable[[], bool] = (lambda: False)
        # UI 心跳節流
        self._last_ui_emit_ts: float = 0.0

        # 保存 crop box 信息，用於生成沒有 mask 的版本
        self._k1_crop_box: Optional[Tuple[int, int, int, int]] = None
        self._k2_crop_box: Optional[Tuple[int, int, int, int]] = None
        self._last_frame_bgr: Optional[np.ndarray] = None
        # 分別記錄 k1 / k2 對應幀（供無 mask 裁切）
        self._last_frame_bgr_k1: Optional[np.ndarray] = None
        self._last_frame_bgr_k2: Optional[np.ndarray] = None
        # 記錄第一針 k2 對應幀（供 precheck 使用第一針）
        self._first_frame_bgr_k2: Optional[np.ndarray] = None
        
        # 事件彙總日誌相關
        self._debug_dir: Optional[str] = None
        self._current_event_index: int = 0
        # Actor 追蹤 ID 與離開計數器（簡化：僅追蹤 ID，不追 centroid/area）
        self._actor_track_id: Optional[int] = None
        self._actor_leave_counter: int = 0

    # --- UI 面板 ---
    def _set_panel(self, mode: str, message: str) -> None:
        try:
            if mode not in ("orange", "green", "red", "none"):
                return
            self.panel_mode = mode
            self.panel_message = str(message or "")
            # 事件級更新：當面板狀態改變時發送事件（僅文字/顏色，不含影格）
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
                # 橙色代表：VLM 判定物品被拿走
                self._set_panel("orange", "Item taken (VLM)")
            elif decision == "no":
                self._set_panel("green", "SAFE")
            
            # 標註 k1k2 合併圖：YES/NO/UNSURE
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

            # 寫入事件彙總日誌（k_img 目錄下）
            if self._debug_dir and self._current_event_index > 0:
                try:
                    append_event_summary(self._debug_dir, self._current_event_index, decision, summary)
                except Exception:
                    LOGGER.exception("Failed to write event summary")
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

    # --- YOLOE / Tracking 準備（可選）---
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
            m = UltralyticsYOLOE(self.cfg.yolo_weights or os.getenv("YOLOE_WEIGHTS", "yoloe-11s-seg.pt"))
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

    # --- YOLOE 結果解析 ---
    def _extract_person_bbox_from_yoloe_result(self, r: Any, fallback_img_shape: Optional[Tuple[int, int]] = None) -> Optional[Tuple[int, int, int, int]]:
        try:
            if r is None:
                return None
            boxes = getattr(r, "boxes", None)
            masks = getattr(r, "masks", None)
            if masks is not None:
                xy = getattr(masks, "xy", None)
                if isinstance(xy, list) and len(xy) > 0:
                    try:
                        poly = xy[0]
                        arr = np.asarray(poly)
                        if arr.ndim == 2 and arr.shape[1] >= 2:
                            xs = arr[:, 0]; ys = arr[:, 1]
                            x1 = int(max(0, np.min(xs)))
                            y1 = int(max(0, np.min(ys)))
                            x2 = int(np.max(xs))
                            y2 = int(np.max(ys))
                            if x2 > x1 and y2 > y1:
                                return (x1, y1, x2, y2)
                    except Exception:
                        pass
            if boxes is not None:
                xyxy = getattr(boxes, "xyxy", None)
                if xyxy is not None:
                    try:
                        arr = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.asarray(xyxy)
                    except Exception:
                        arr = np.asarray(xyxy)
                    if arr.size >= 4:
                        b = arr[0]
                        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                        if x2 > x1 and y2 > y1:
                            return (x1, y1, x2, y2)
        except Exception:
            return None
        return None

    def _run_sam_on_bbox(self, frame_bgr: np.ndarray, bbox_xyxy: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        try:
            if self.orch is None:
                return None
            orig_points = list(self.cfg.points)
            orig_labels = list(self.cfg.labels)
            self.orch.set_sam_prompts(points=[], labels=[], roi_box=bbox_xyxy)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            _ = self.orch.preview_sam(frame_rgb)
            m = self.orch.perception.get_last_sam_mask()
            self.orch.set_sam_prompts(points=orig_points, labels=orig_labels, roi_box=None)
            try:
                _ = self.orch.preview_sam(frame_rgb)
            except Exception:
                pass
            return m
        except Exception:
            return None

    def _apply_mask_to_crop(self, crop_bgr: np.ndarray, full_mask_bin: np.ndarray, crop_box: Tuple[int, int, int, int], mode: str = "keep") -> np.ndarray:
        try:
            x1, y1, x2, y2 = crop_box
            h, w = crop_bgr.shape[:2]
            local = full_mask_bin[y1:y2, x1:x2]
            if local.shape[0] != h or local.shape[1] != w:
                local = cv2.resize(local, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = (local > 0).astype(np.uint8)
            if mode == "overlay":
                colored = np.zeros_like(crop_bgr)
                colored[:, :, 1] = (mask * 255)
                out = cv2.addWeighted(crop_bgr, 0.7, colored, 0.3, 0)
                return out
            bg = (crop_bgr * 0.3).astype(np.uint8)
            mask3 = np.repeat(mask[:, :, None], 3, axis=2)
            out = np.where(mask3 > 0, crop_bgr, bg)
            return out
        except Exception:
            return crop_bgr

    def _build_clean_kframes(self, frames: List[Optional[np.ndarray]]) -> List[Optional[np.ndarray]]:
        try:
            clean_frames = list(frames)
            # 使用對應幀 + 對應裁切框，生成無 mask 版本
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

    def _bbox_iou(self, a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        try:
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0:
                return 0.0
            area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
            area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
            denom = float(area_a + area_b - inter) if (area_a + area_b - inter) > 0 else 1.0
            return float(inter) / denom
        except Exception:
            return 0.0

    # --- K1/K2 前置檢查（使用同一個 prompt 檢查 k1 與 k2；兩者皆 YES 才通過） ---
    def _precheck_k1(self, k1_img: Optional[np.ndarray]) -> bool:
        try:
            if not self.cfg.precheck_enabled:
                return True
            if self.orch is None or getattr(self.orch, "reasoning", None) is None:
                return True  # 若無法檢查，放行以不阻塞流程
            
            # 若有 k1 裁切框，優先使用裁切後影像；否則回退到輸入或全圖
            k1_in = None
            try:
                if self._k1_crop_box is not None and isinstance(self._last_frame_bgr, np.ndarray):
                    k1_in = crop_by_box(self._last_frame_bgr, self._k1_crop_box)
            except Exception:
                k1_in = None
            if k1_in is None:
                if k1_img is not None:
                    k1_in = k1_img
                else:
                    base = self._last_frame_bgr if isinstance(self._last_frame_bgr, np.ndarray) else None
                    if base is not None:
                        k1_in = base
            if k1_in is None:
                return False

            k1_prompt = (
                "You are a surveillance system looking. The cabinet is the ROI. "
                "If a person is interacting  with the  ROI, answer YES; "
                "otherwise answer NO. Output YES or NO only."
            )

            r1 = self.orch.reasoning.analyze_keyframes([k1_in], prompt=k1_prompt)
            d1 = self._parse_decision(str(r1.get("summary", "")))
            # 簡潔記錄（僅在啟用 precheck 時）
            try:
                stem = "precheck"
                # 在 dispatch 階段會以同一個 debug_dir 寫 k1/k2，這裡不建立檔案，僅暫存於實例屬性
                self._last_precheck = {
                    "k1_summary": str(r1.get("summary", "")),
                    "k1_decision": d1,
                    "k2_summary": "",
                    "k2_decision": "",
                    "k1_img": k1_in,
                    "k2_img": None,
                }
            except Exception:
                pass
            return (d1 == "yes")
        except Exception:
            return False

    def _precheck_k1k2(self, k1_img: Optional[np.ndarray], k2_img: Optional[np.ndarray]) -> bool:
        try:
            if not self.cfg.precheck_enabled:
                return True
            if self.orch is None or getattr(self.orch, "reasoning", None) is None:
                return True  # 若無法檢查，放行以不阻塞流程

            # 直接使用傳入的影像（需由外部以相同清理邏輯產生）
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
                "- 必須看到手明確接觸或操作ROI，且K1→K2能解釋為一次連續行為。"
                "- 單純路過、擦到、掠過、模糊不清都算no。"
                "- 不確定回答yes。"
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

    # --- 公開執行介面 ---
    def run(self) -> None:
        in_path = self.cfg.input_path
        if os.path.isdir(in_path):
            vids: List[str] = []
            for root, _, files in os.walk(in_path):
                for fn in files:
                    if fn.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".flv", ".webm")):
                        vids.append(os.path.join(root, fn))
            vids.sort()
            LOGGER.info("found %d videos in %s", len(vids), in_path)

            self.orch = Orchestrator(iou_threshold=self.cfg.iou_threshold, vlm_backend=self.cfg.vlm_backend)
            try:
                self.orch.initialize_models()
                self.orch.set_result_callback(self._result_callback)
                self.orch.set_prompts(visual_template_image=None, text_prompt="person")
                for vp in vids:
                    try:
                        self.process_one_video_with_orchestrator(vp)
                    except Exception:
                        LOGGER.exception("failed on %s", vp)
            finally:
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
        else:
            self.process_one_video(self.cfg.input_path)

    # --- UI 事件回呼（供外部 UI 訂閱事件級狀態）---
    def set_ui_event_callback(self, cb: Optional[Callable[[Dict[str, Any]], None]]) -> None:
        """設定 UI 事件回呼。回呼將收到 dict：
        {
          "color": "green|orange|red|none",
          "message": str,
          "vlm_text": str,
          "event_index": int,
          "timestamp_ms": float
        }
        """
        self._ui_event_cb = cb

    def _emit_ui_event(self, color: str, message: str, vlm_text: str) -> None:
        try:
            if self._ui_event_cb is None:
                return
            evt = {
                "color": str(color or "none"),
                "message": str(message or ""),
                "vlm_text": str(vlm_text or ""),
                "event_index": int(self._last_event_index),
                "frame_index": int(self._last_event_index),  # 使用相同的 frame_index
                "timestamp_ms": float(time.time() * 1000.0),
            }
            self._ui_event_cb(evt)
        except Exception:
            pass

    def set_should_stop(self, fn: Optional[Callable[[], bool]]) -> None:
        try:
            if fn is None:
                self._should_stop = (lambda: False)
            else:
                self._should_stop = fn
        except Exception:
            self._should_stop = (lambda: False)

    def process_one_video(self, video_path: str) -> Optional[str]:
        self.current_vlm_text = ""
        self.vlm_has_output = False
        self.panel_mode = "none"
        self.panel_message = ""

        self.orch = Orchestrator(iou_threshold=self.cfg.iou_threshold, vlm_backend=self.cfg.vlm_backend)
        try:
            self.orch.initialize_models()
            self.orch.set_result_callback(self._result_callback)
            self.orch.set_prompts(visual_template_image=None, text_prompt="person")
            return self.process_one_video_with_orchestrator(video_path)
        finally:
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

    # --- 單影片主流程（依照 auto_pipeline_v2 對齊）---
    def process_one_video_with_orchestrator(self, video_path: str) -> Optional[str]:
        if not os.path.isfile(video_path):
            LOGGER.error("not a file: %s", video_path)
            return None

        self.current_vlm_text = ""
        self.vlm_has_output = False
        self.panel_mode = "none"
        self.panel_message = ""

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            LOGGER.error("failed to open: %s", video_path)
            return None

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        size = (width, height)
        ensure_dir(self.cfg.output_dir)
        stem = os.path.splitext(os.path.basename(video_path))[0]
        # 依需求建立 outputs/<影片名>/{video,k_img}
        video_dir = os.path.join(self.cfg.output_dir, stem, "video")
        kimg_dir = os.path.join(self.cfg.output_dir, stem, "k_img")
        ensure_dir(video_dir)
        ensure_dir(kimg_dir)
        out_path = os.path.join(video_dir, f"{stem}_auto_out_v2.mp4")
        writer = make_video_writer(out_path, fps, size)
        LOGGER.info("processing: %s -> %s", video_path, out_path)

        # 準備追蹤器
        self._prepare_tracker()

        # 讀第一幀：準備 SAM 或固定 ROI 遮罩
        ok, first_bgr = cap.read()
        if not ok or first_bgr is None:
            LOGGER.error("empty video: %s", video_path)
            cap.release(); writer.release()
            return None
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
                try:
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
                except Exception:
                    LOGGER.exception("failed to build fixed ROI mask; fallback to SAM")
                    self.use_fixed_roi_mask = False
            # 構建負向 ROI 遮罩（若有）：同樣支援像素自動 normalize
            if self.use_fixed_neg_roi_mask and self.cfg.neg_roi_poly_norm is not None:
                try:
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

                    # 構建負向 ROI mask（lazy 構建：一次性）
                    if not hasattr(self, "_neg_roi_mask_bin") or getattr(self, "_neg_roi_mask_bin") is None:
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
                    LOGGER.exception("failed to build NEG-ROI mask; disable neg mask")
                    self._neg_roi_mask_bin = None
                    self.use_fixed_neg_roi_mask = False

            if not self.use_fixed_roi_mask and self.orch is not None:
                if len(self.cfg.points) > 0:
                    self.orch.set_sam_prompts(points=self.cfg.points, labels=(self.cfg.labels or [1] * len(self.cfg.points)), roi_box=None)
                    first_rgb = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB)
                    _ = self.orch.preview_sam(first_rgb)
                    LOGGER.info("SAM preview done with %d points", len(self.cfg.points))
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

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        start_time = time.time()

        trigger = InteractionTrigger(iou_threshold=self.cfg.iou_threshold, consecutive_frames=self.cfg.trigger_frames, neg_iou_threshold=self.cfg.neg_iou_threshold)
        state = "idle"
        contact_active = False
        kbuf = ReasoningKeyframeBuffer()
        k1_collected = False
        k2_sample_frames = [1,8,16]
        k2_sample_count = 0
        post_leaving_frame_counter = 0
        k1_crop_center = None
        k2_crop_box: Optional[Tuple[int, int, int, int]] = None
        # 修正非法取樣幀位（如 0 或負數），至少從第 1 幀開始
        try:
            k2_sample_frames = [max(1, int(n)) for n in (k2_sample_frames or [1, 2])]
        except Exception:
            k2_sample_frames = [1, 2]

        person_missing_frames = 0
        person_missing_threshold = 10

        frame_index = 0
        debug_dir = kimg_dir
        ensure_dir(debug_dir)
        event_index = 1
        
        # 設定事件彙總日誌相關屬性
        self._debug_dir = debug_dir
        self._current_event_index = event_index

        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    break
                # 停止檢查
                try:
                    if self._should_stop():
                        break
                except Exception:
                    pass
                frame_index += 1
                # 提供事件序號給事件回呼（UI 可用於顯示/分組）
                self._last_event_index = int(event_index)

                annotated_bgr = None
                r = None
                if self.track_model is not None:
                    try:
                        results = self.track_model.track(frame_bgr, persist=True, tracker=self.cfg.tracker_cfg, conf=self.cfg.track_conf)
                        r = results[0] if isinstance(results, (list, tuple)) and len(results) > 0 else results
                        annotated_bgr = r.plot()
                    except Exception:
                        LOGGER.exception("YOLOE.track failed; skip annotate")

                person_present = self._is_person_present_from_result(r)
                if not person_present:
                    person_missing_frames += 1
                else:
                    person_missing_frames = 0
                if self.panel_mode == "orange" and person_missing_frames >= person_missing_threshold:
                    self._set_panel("red", "person leave with item")

                # SAM mask
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

                # 主要 ROI 的 overlap
                max_overlap = 0.0
                best_poly = None
                best_idx = -1
                polys = None
                if r is not None and sam_bin is not None:
                    polys = yolo_polys_from_result(r)
                    for i, poly in enumerate(polys):
                        ov = overlap_ratio_poly_with_mask(poly, sam_bin, width, height)
                        if ov > max_overlap:
                            max_overlap = ov
                            best_poly = poly
                            best_idx = i

                # 負向 ROI 的 overlap（若提供）
                neg_overlap = 0.0
                if self.use_fixed_neg_roi_mask and self.cfg.neg_roi_poly_norm is not None:
                    try:
                        # 構建負向 ROI mask（lazy 構建：重用 _neg_roi_mask_bin）
                        if not hasattr(self, "_neg_roi_mask_bin"):
                            poly_px: List[Tuple[int, int]] = []
                            for xn, yn in (self.cfg.neg_roi_poly_norm or []):
                                x = int(round(float(xn) * float(width)))
                                y = int(round(float(yn) * float(height)))
                                x = max(0, min(width - 1, x))
                                y = max(0, min(height - 1, y))
                                poly_px.append((x, y))
                            mask0 = np.zeros((height, width), dtype=np.uint8)
                            if len(poly_px) >= 3:
                                pts = np.array(poly_px, dtype=np.int32).reshape((-1, 1, 2))
                                cv2.fillPoly(mask0, [pts], 255)
                                setattr(self, "_neg_roi_mask_bin", (mask0 > 0).astype(np.uint8))
                            else:
                                setattr(self, "_neg_roi_mask_bin", None)
                        neg_mask = getattr(self, "_neg_roi_mask_bin", None)
                        if neg_mask is not None and r is not None:
                            polys2 = polys if polys is not None else yolo_polys_from_result(r)
                            for poly in polys2:
                                ov2 = overlap_ratio_poly_with_mask(poly, neg_mask, width, height)
                                if ov2 > neg_overlap:
                                    neg_overlap = ov2
                    except Exception:
                        neg_overlap = 0.0

                event = trigger.update(max_overlap)
                if state == "idle":
                    if event and not contact_active:
                        contact_active = True
                        kbuf.reset()
                        state = "contacting"
                elif state == "contacting":
                    # 若在 contacting 階段（已收 K1 前/後）偵測到負向 ROI 觸發，取消此次事件
                    try:
                        if trigger.is_negative_violated(neg_overlap):
                            kbuf.reset(); state = "idle"; contact_active = False; k1_collected = False
                            trigger.reset(); person_missing_frames = 0; k2_sample_count = 0; k2_crop_box = None
                            event_index += 1; self._current_event_index = event_index
                            continue
                    except Exception:
                        pass
                    if not k1_collected:
                        # 先將 SAM mask 畫到 frame_bgr 上
                        frame_with_sam = frame_bgr
                        if sam_bin is not None:
                            try:
                                frame_with_sam = overlay_sam_mask(frame_bgr, sam_bin, color=(0, 255, 0), alpha=0.35)
                            except Exception:
                                pass
                        kbuf.add("k1", frame_with_sam)
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
                                        # 保存 crop box 信息，用於後續生成沒有 mask 的版本
                                        self._k1_crop_box = eb
                                        self._last_frame_bgr = frame_bgr.copy()
                                        self._last_frame_bgr_k1 = frame_bgr.copy()
                                        # 不應用 person mask，直接使用 crop 後的圖片
                                        kbuf.replace_last("k1", cropped)
                                        k1_crop_center = (cx, cy)
                                        # 在 K1 當下鎖定 Actor tracking ID（若可用）
                                        try:
                                            if self.cfg.actor_leave_enabled and r is not None and best_idx >= 0:
                                                boxes = getattr(r, "boxes", None)
                                                ids = getattr(boxes, "id", None) if boxes is not None else None
                                                if ids is not None:
                                                    try:
                                                        ids_arr = ids.cpu().numpy() if hasattr(ids, "cpu") else np.asarray(ids)
                                                    except Exception:
                                                        ids_arr = np.asarray(ids)
                                                    if ids_arr is not None and ids_arr.size > best_idx:
                                                        try:
                                                            self._actor_track_id = int(ids_arr[best_idx])
                                                            self._actor_leave_counter = 0
                                                        except Exception:
                                                            self._actor_track_id = None
                                                            self._actor_leave_counter = 0
                                            else:
                                                self._actor_track_id = None
                                                self._actor_leave_counter = 0
                                        except Exception:
                                            self._actor_track_id = None
                                            self._actor_leave_counter = 0
                        except Exception:
                            pass
                        k1_collected = True

                    # 使用 tracking ID 的 Actor 離開判定（即使仍有人在 ROI 也可觸發 K2）
                    try:
                        if self.cfg.actor_leave_enabled and k1_collected:
                            actor_overlap = None
                            if self._actor_track_id is not None and r is not None and sam_bin is not None:
                                try:
                                    boxes = getattr(r, "boxes", None)
                                    ids = getattr(boxes, "id", None) if boxes is not None else None
                                    if ids is not None:
                                        try:
                                            ids_arr = ids.cpu().numpy() if hasattr(ids, "cpu") else np.asarray(ids)
                                        except Exception:
                                            ids_arr = np.asarray(ids)
                                        idx_list = []
                                        try:
                                            # 支援多個同 ID（理論上只取第一個）
                                            for i in range(int(ids_arr.shape[0])):
                                                try:
                                                    if int(ids_arr[i]) == int(self._actor_track_id):
                                                        idx_list.append(i)
                                                except Exception:
                                                    continue
                                        except Exception:
                                            idx_list = []
                                        target_idx = idx_list[0] if len(idx_list) > 0 else -1
                                        if target_idx >= 0:
                                            # 取該 idx 的 polygon（若無 mask，改用 bbox 近似多邊形）
                                            poly_pts = None
                                            try:
                                                masks = getattr(r, "masks", None)
                                                xy = getattr(masks, "xy", None) if masks is not None else None
                                                if isinstance(xy, list) and target_idx < len(xy):
                                                    arr = np.asarray(xy[target_idx])
                                                    if arr.ndim == 2 and arr.shape[1] >= 2:
                                                        poly_pts = [(int(p[0]), int(p[1])) for p in arr]
                                            except Exception:
                                                poly_pts = None
                                            if poly_pts is None:
                                                try:
                                                    xyxy = getattr(boxes, "xyxy", None)
                                                    if xyxy is not None:
                                                        arr = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.asarray(xyxy)
                                                        if arr.size >= (target_idx + 1) * 4:
                                                            b = arr[target_idx]
                                                            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                                                            poly_pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                                                except Exception:
                                                    poly_pts = None
                                            if isinstance(poly_pts, list) and len(poly_pts) >= 3:
                                                actor_overlap = overlap_ratio_poly_with_mask(poly_pts, sam_bin, width, height)
                                except Exception:
                                    pass
                            # 若無法取得 actor_overlap，視為 0
                            if actor_overlap is None:
                                actor_overlap = 0.0
                            # 低於門檻則累計離開計數，反之歸零
                            if actor_overlap < max(0.0, float(self.cfg.iou_threshold)):
                                self._actor_leave_counter += 1
                            else:
                                self._actor_leave_counter = 0
                            # 連續不足門檻達到耐心值：進入 post_leaving，提前觸發 K2
                            if self._actor_leave_counter >= int(max(1, self.cfg.actor_leave_patience)):
                                contact_active = False
                                k2_sample_count = 0
                                # 優先沿用 k1 的裁切框
                                try:
                                    if self.cfg.crop_k2:
                                        if self._k1_crop_box is not None:
                                            x1, y1, x2, y2 = self._k1_crop_box
                                            x1 = max(0, min(x1, width)); x2 = max(0, min(x2, width))
                                            y1 = max(0, min(y1, height)); y2 = max(0, min(y2, height))
                                            if x2 > x1 and y2 > y1:
                                                k2_crop_box = (x1, y1, x2, y2)
                                            else:
                                                k2_crop_box = None
                                        elif k1_crop_center is not None:
                                            cx, cy = k1_crop_center
                                            base_size = 100
                                            margin_px = int(min(width, height) * self.cfg.crop_margin_ratio)
                                            cw = min(self.cfg.crop_max_size, max(self.cfg.crop_min_size, base_size + margin_px * 2))
                                            ch = cw if self.cfg.crop_square else min(self.cfg.crop_max_size, max(self.cfg.crop_min_size, base_size + margin_px * 2))
                                            x1 = max(0, cx - cw // 2)
                                            y1 = max(0, cy - ch // 2)
                                            x2 = min(width, x1 + cw)
                                            y2 = min(height, y1 + ch)
                                            if x2 > x1 and y2 > y1:
                                                k2_crop_box = (x1, y1, x2, y2)
                                            else:
                                                k2_crop_box = None
                                    else:
                                        k2_crop_box = None
                                except Exception:
                                    k2_crop_box = None
                                state = "post_leaving"
                                post_leaving_frame_counter = 0
                                # 清理 actor 狀態
                                self._actor_leave_counter = 0
                                # 保持 _actor_track_id 以便除錯；可選擇清空
                    except Exception:
                        pass

                    if not event and contact_active:
                        contact_active = False
                        k2_sample_count = 0
                        try:
                            if self.cfg.crop_k2:
                                # 優先沿用 k1 的裁切框，確保 k1/k2 一致
                                if self._k1_crop_box is not None:
                                    x1, y1, x2, y2 = self._k1_crop_box
                                    # clamp 到畫面內
                                    x1 = max(0, min(x1, width))
                                    x2 = max(0, min(x2, width))
                                    y1 = max(0, min(y1, height))
                                    y2 = max(0, min(y2, height))
                                    if x2 > x1 and y2 > y1:
                                        k2_crop_box = (x1, y1, x2, y2)
                                    else:
                                        k2_crop_box = None
                                elif k1_crop_center is not None:
                                    cx, cy = k1_crop_center
                                    base_size = 100
                                    margin_px = int(min(width, height) * self.cfg.crop_margin_ratio)
                                    cw = min(self.cfg.crop_max_size, max(self.cfg.crop_min_size, base_size + margin_px * 2))
                                    ch = cw if self.cfg.crop_square else min(self.cfg.crop_max_size, max(self.cfg.crop_min_size, base_size + margin_px * 2))
                                    x1 = max(0, cx - cw // 2)
                                    y1 = max(0, cy - ch // 2)
                                    x2 = min(width, x1 + cw)
                                    y2 = min(height, y1 + ch)
                                    if x2 > x1 and y2 > y1:
                                        k2_crop_box = (x1, y1, x2, y2)
                                    else:
                                        k2_crop_box = None
                            else:
                                k2_crop_box = None
                        except Exception:
                            k2_crop_box = None
                        state = "post_leaving"
                        post_leaving_frame_counter = 0
                elif state == "post_leaving":
                    # 在 K2 採樣前，如負向 ROI 被觸發，取消此次事件
                    try:
                        if trigger.is_negative_violated(neg_overlap):
                            kbuf.reset(); state = "idle"; contact_active = False; k1_collected = False
                            trigger.reset(); person_missing_frames = 0; k2_sample_count = 0; k2_crop_box = None
                            self._actor_track_id = None; self._actor_leave_counter = 0
                            event_index += 1; self._current_event_index = event_index
                            continue
                    except Exception:
                        pass
                    post_leaving_frame_counter += 1
                    if k2_sample_count < len(k2_sample_frames) and post_leaving_frame_counter == k2_sample_frames[k2_sample_count]:
                        try:
                            frame_to_add = frame_bgr
                            if self.cfg.crop_k2 and k2_crop_box is not None:
                                eb = k2_crop_box
                                cropped = crop_by_box(frame_bgr, eb)
                                # 保存 crop box 信息，用於後續生成沒有 mask 的版本
                                self._k2_crop_box = eb
                                if self._last_frame_bgr is None:
                                    self._last_frame_bgr = frame_bgr.copy()
                                # 若是第一針 k2，額外保存供 precheck 使用
                                if k2_sample_count == 0:
                                    self._first_frame_bgr_k2 = frame_bgr.copy()
                                self._last_frame_bgr_k2 = frame_bgr.copy()
                                # 不應用 person mask，直接使用 crop 後的圖片
                                frame_to_add = cropped
                            kbuf.add("k2", frame_to_add)
                            k2_sample_count += 1
                        except Exception:
                            pass
                    if k2_sample_count >= len(k2_sample_frames):
                        state = "dispatch"
                elif state == "dispatch":
                    frames_for_vlm = kbuf.collect(max_k1=1, max_k2=2, max_k3=0)
                    # 產生與 precheck 一致的乾淨 k1/k2（使用相同函式）
                    try:
                        frames_for_vlm = self._build_clean_kframes(frames_for_vlm)
                    except Exception:
                        pass
                    # 前置檢查：使用放大裁切（precheck_scale）檢查互動；dispatch 保持原裁切
                    try:
                        k1_img = frames_for_vlm[0] if len(frames_for_vlm) > 0 else None
                        k2_img = frames_for_vlm[1] if len(frames_for_vlm) > 1 else None
                        # 建立 precheck 專用較大裁切
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
                        # 若有 precheck 結果，寫入簡潔 JSON
                        try:
                            pc = getattr(self, "_last_precheck", None)
                            if isinstance(pc, dict):
                                _ = save_precheck_log(debug_dir, event_index,
                                                      str(pc.get("k1_summary", "")), str(pc.get("k1_decision", "")),
                                                      str(pc.get("k2_summary", "")), str(pc.get("k2_decision", "")))
                                _ = save_precheck_images(debug_dir, event_index,
                                                         pc.get("k1_img"), pc.get("k2_img"))
                        except Exception:
                            pass
                        if not pre_ok:
                            # precheck 未通過：仍保存 k1/k2 合併圖，並以紅字標註 "Filtered"
                            try:
                                frames_for_fail = []
                                if k1_img is not None:
                                    frames_for_fail.append(k1_img)
                                # 取當前收集到的 k2（若存在）
                                if k2_img is not None:
                                    frames_for_fail.append(k2_img)
                                if len(frames_for_fail) >= 2:
                                    save_kframes(debug_dir, stem, event_index, frames_for_fail)
                                    from .output import annotate_image_file
                                    k1k2_path_fail = os.path.join(debug_dir, f"{stem}_e{event_index}_k1k2.jpg")
                                    if os.path.isfile(k1k2_path_fail):
                                        try:
                                            annotate_image_file(k1k2_path_fail, "Filtered", color=(0, 0, 255))
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                            # 濾掉此次事件並重置狀態
                            kbuf.reset()
                            state = "idle"; contact_active = False; k1_collected = False
                            trigger.reset()
                            person_missing_frames = 0
                            k2_sample_count = 0
                            k2_crop_box = None
                            self._actor_track_id = None; self._actor_leave_counter = 0
                            event_index += 1
                            self._current_event_index = event_index
                            continue
                    except Exception:
                        # 任意錯誤也視為不通過，避免誤報
                        kbuf.reset()
                        state = "idle"; contact_active = False; k1_collected = False
                        trigger.reset()
                        person_missing_frames = 0
                        k2_sample_count = 0
                        k2_crop_box = None
                        self._actor_track_id = None; self._actor_leave_counter = 0
                        event_index += 1
                        self._current_event_index = event_index
                        continue

                    try:
                        save_kframes(debug_dir, stem, event_index, frames_for_vlm)
                    except Exception:
                        LOGGER.exception("Failed to save k1/k2 debug images")
                    prompt = (
                        "k1: hand touches cabinet. "
                        "k2: hand withdraw cabinet ( consecutive frames after withdraw). "
                        "From your observation from k1, k2, check if the hand reaching into cabinet and takes a new item from cabinet(check if there is item holded in any of the k2 frames). "
                        "If yes, answer YES. "
                        "If not, answer NO. "
                        "If unsure, answer UNSURE. "
                        "Output one word only."
                    )
                    try:
                        if self.orch is not None:
                            # 附加 k1k2 合併圖的路徑，供 callback 標註
                            k1k2_path = os.path.join(debug_dir, f"{stem}_e{event_index}_k1k2.jpg")
                            self.orch._event_queue.put_nowait({
                                "frames": frames_for_vlm,
                                "prompt": prompt,
                                "event_index": int(event_index),
                                "debug_dir": str(debug_dir),
                                "k1k2_path": k1k2_path,
                            })
                    except Exception:
                        LOGGER.exception("enqueue VLM task failed")
                    # 事件完成後統一重置
                    kbuf.reset()
                    state = "idle"; contact_active = False; k1_collected = False
                    trigger.reset()
                    person_missing_frames = 0
                    k2_sample_count = 0
                    k2_crop_box = None
                    self._actor_track_id = None; self._actor_leave_counter = 0
                    event_index += 1
                    self._current_event_index = event_index
                    # 清理對應幀暫存
                    self._last_frame_bgr_k1 = None
                    self._last_frame_bgr_k2 = None
                    self._first_frame_bgr_k2 = None

                text = self.current_vlm_text or ""
                base_bgr = annotated_bgr if annotated_bgr is not None else frame_bgr
                if self.cfg.draw_sam and self._initial_sam_mask_bin is not None:
                    try:
                        base_bgr = overlay_sam_mask(base_bgr, self._initial_sam_mask_bin, color=(0, 255, 0), alpha=0.35)
                    except Exception:
                        pass
                base_bgr = draw_status_panel(base_bgr, self.panel_mode, self.panel_message)
                out_frame = put_text_bottom_right(cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB), text)
                out_frame_bgr = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
                writer.write(out_frame_bgr)
                # 心跳事件：每 ~0.5 秒推送一次目前面板狀態，避免 UI 無事件時不更新
                try:
                    now_ts = time.time()
                    if (now_ts - self._last_ui_emit_ts) >= 0.5:
                        # 更新 frame_index 為當前處理的幀數
                        self._last_event_index = frame_index
                        self._emit_ui_event(self.panel_mode, self.panel_message, self.current_vlm_text)
                        self._last_ui_emit_ts = now_ts
                except Exception:
                    pass
                if self.cfg.show_window:
                    try:
                        cv2.imshow("auto_v2", out_frame_bgr)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    except Exception:
                        pass
                # 迴圈底部停止檢查（以縮短響應延遲）
                try:
                    if self._should_stop():
                        break
                except Exception:
                    pass
        finally:
            cap.release()
            writer.release()
            try:
                if self.cfg.show_window:
                    cv2.destroyAllWindows()
            except Exception:
                pass

        elapsed = time.time() - start_time
        LOGGER.info("done: %s (%.2fs)", out_path, elapsed)
        return out_path





