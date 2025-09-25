from typing import Any, Dict, List, Optional, Tuple, Union

import os
import numpy as np
import logging


class PerceptionEngine:
    """
    YOLOE 封裝（介面層）

    目標：
    - 支援 visual prompt（以範例影像作為模板，使用模板比對或模型特徵相似度）
    - 支援 text prompt（以關鍵字/類別名稱引導檢測過濾）

    設計說明：
    - initialize(): 嘗試載入 YOLOE；若環境無對應相依，保持 _model=None，並以簡易模板匹配作為退場機制。
    - set_prompts()/set_prompts_v2(): 設定文字與影像提示。
    - detect(frame_bgr): 回傳 (det_a, det_b)，供 orchestrator 使用（兩個來源的檢測，用於計算 IOU）。
    - detect_for_ui(frame_bgr): 回傳 list[dict] 供 UI 繪製（label/box/confidence）。

    注意：
    - 真正的 YOLOE 推論需整合對應框架（如 MMYOLO）。此處以接口穩定與退場機制為主，
      若 _model 不可用，則以 visual prompt 的模板匹配提供 det_a，text prompt 則無模型時回傳空。
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        use_image_prompts: bool = True,
        conf_threshold: float = 0.8,
        debug: bool = True,
        **_: Any,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.use_image_prompts = use_image_prompts
        self.conf_threshold = conf_threshold
        self._debug = debug

        self._initialized: bool = False
        self._model: Optional[Any] = None  # Ultralytics YOLOE 實例
        self._evp_predictor_cls: Optional[Any] = None  # YOLOEVPSegPredictor

        self._text_prompt: str = ""
        self._visual_prompts: List[np.ndarray] = []  # BGR 影像列表
        self._visual_bbox_list: List[Tuple[int, int, int, int]] = []  # 對應 visual_prompts 的 bbox

        self._logger = logging.getLogger("perception.yoloe")
        if not self._logger.handlers:
            logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s")

    # --- lifecycle ---
    def initialize(self) -> None:
        if self._initialized:
            return
        # 嘗試載入 Ultralytics YOLOE；若套件或權重缺失則保留 None，使用退場機制
        model = None
        evp_cls = None
        try:
            from ultralytics import YOLOE
            from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

            # 找權重：優先 self.model_path → 環境變數 YOLOE_WEIGHTS → 專案常見路徑
            weight_candidates: List[str] = []
            if isinstance(self.model_path, str) and self.model_path.strip():
                weight_candidates.append(self.model_path)
            env_w = os.getenv("YOLOE_WEIGHTS", "").strip()
            if env_w:
                weight_candidates.append(env_w)
            # 專案路徑
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            weight_candidates.extend([
                os.path.join(base_dir, "yoloe-11l-seg.pt"),
                os.path.join(base_dir, "weights", "yoloe-11l-seg.pt"),
                "yoloe-11l-seg.pt",
            ])

            weight_path: Optional[str] = None
            for p in weight_candidates:
                try:
                    if p and os.path.isfile(p):
                        weight_path = p
                        break
                except Exception:
                    continue

            if weight_path:
                model = YOLOE(weight_path)
                evp_cls = YOLOEVPSegPredictor
        except Exception as e:
            self._log(f"YOLOE initialize failed: {repr(e)}")
            model = None
            evp_cls = None

        self._model = model
        self._evp_predictor_cls = evp_cls

        # SAM2 初始化（可選）
        self._sam_model = None
        try:
            from ultralytics import SAM
            # 權重檔可透過環境變數覆寫，Ultralytics 官方示例：sam2.1_b.pt / sam2_b.pt 等
            sam2_weights = os.getenv("SAM2_WEIGHTS", "sam2.1_b.pt")
            self._sam_model = SAM(sam2_weights)
            self._log("SAM model initialized")
        except Exception as e:
            self._log(f"SAM init skipped/failed: {repr(e)}")
            self._sam_model = None

        self._initialized = True
        # 互動式狀態
        self._sam_prev_logits = None  # 上一輪遮罩 logits（若模型支援）

    def reset(self) -> None:
        self._text_prompt = ""
        self._visual_prompts = []
        self._visual_bbox_list = []

    def set_debug(self, enabled: bool) -> None:
        self._debug = bool(enabled)

    # --- prompts ---
    def set_prompts(self, visual_template_image: Optional[np.ndarray], text_prompt: str) -> None:
        self._text_prompt = text_prompt or ""
        # 不再使用視覺提示，只保留文字提示
        self._visual_prompts = []
        self._visual_bbox_list = []

    def set_prompts_v2(self, texts: Optional[List[str]] = None, images: Optional[List[np.ndarray]] = None) -> None:
        self._text_prompt = (texts[0] if texts else "") or ""
        # 不再使用視覺提示，只保留文字提示
        self._visual_prompts = []
        self._visual_bbox_list = []

    # --- core detect wrappers ---
    def detect(self, frame_bgr: np.ndarray) -> Tuple[Optional[List[Tuple[int, int, int, int]]], Optional[List[Tuple[int, int, int, int]]]]:
        """
        回傳：
        - det_a: 針對 visual prompt 的偵測框列表（供 IOU 計算）- 已停用
        - det_b: 針對 text prompt 的偵測框列表（供 IOU 計算）

        若任一來源無可用結果，對應值可為 None。
        """
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
            return None, None

        # 不再使用視覺提示，只使用文字提示
        det_a: Optional[List[Tuple[int, int, int, int]]] = None
        det_b: Optional[List[Tuple[int, int, int, int]]] = self._detect_by_text_prompt(frame_bgr)
        return det_a, det_b

    def detect_batch(self, frames_bgr: List[np.ndarray]) -> List[Optional[List[Tuple[int, int, int, int]]]]:
        """
        批次版：只回傳 det_b（text prompt 的 bbox 列表），長度與輸入影格一致。
        - 若無模型或無 text prompt，回傳 [None] * len(frames)
        - 若推論失敗，對應索引回傳 None
        """
        try:
            num_frames = len(frames_bgr)
        except Exception:
            return []

        if num_frames == 0:
            return []

        text = (self._text_prompt or "").strip()
        if self._model is None or not text:
            return [None] * num_frames

        # 前處理：過濾無效影格，保留索引以便復原順序
        valid_indices: List[int] = []
        valid_frames: List[np.ndarray] = []
        for i, f in enumerate(frames_bgr):
            if isinstance(f, np.ndarray) and f.size > 0:
                valid_indices.append(i)
                valid_frames.append(f)

        outputs: List[Optional[List[Tuple[int, int, int, int]]]] = [None] * num_frames
        if not valid_frames:
            return outputs

        try:
            names = self._parse_text_prompt_to_names(text)
            if names:
                txt_pe = self._model.get_text_pe(names)
                self._model.set_classes(names, txt_pe)

            # Ultralytics 支援 list[np.ndarray] 做批次推論
            results = self._model.predict(valid_frames, device=self.device, conf=self.conf_threshold)

            # 將每個結果解析為 bbox 列表
            per_frame_boxes: List[List[Tuple[int, int, int, int]]] = []
            try:
                for r in results:
                    per_frame_boxes.append(self._extract_xyxy([r]))
            except Exception:
                # 某些版本 results 不是可疊代，退回單結果
                per_frame_boxes = [self._extract_xyxy(results)]

            # 回填到對應位置
            for local_i, gi in enumerate(valid_indices):
                try:
                    boxes = per_frame_boxes[local_i] if local_i < len(per_frame_boxes) else []
                    outputs[gi] = boxes or None
                except Exception:
                    outputs[gi] = None
        except Exception as e:
            self._log(f"detect_batch error: {repr(e)}")
            # 保持 None 輸出
        return outputs

    def detect_for_ui(self, frame_bgr: np.ndarray) -> List[Dict[str, Union[str, float, Tuple[int, int, int, int]]]]:
        """
        UI 友善版本：回傳 list[dict]，每個 dict 需包含：
        - label: str
        - box: (x1, y1, x2, y2)
        - confidence: float
        """
        results: List[Dict[str, Union[str, float, Tuple[int, int, int, int]]]] = []
        det_a, det_b = self.detect(frame_bgr)
        # 只使用文字提示的偵測結果
        if det_b:
            for box in det_b:
                results.append({"label": (self._text_prompt or "text"), "box": box, "confidence": 1.0})
        return results

    def detect_for_ui_with_masks(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        回傳包含可選 'mask' 的偵測結果，供 UI 覆蓋分割遮罩。
        結構：[{label, box, confidence, mask(optional poly list)}]
        """
        # 只使用文字提示進行推論
        if self._model is not None:
            try:
                results = None
                if (self._text_prompt or "").strip():
                    self._log(f"text_prompt infer: names={self._parse_text_prompt_to_names(self._text_prompt)} conf={self.conf_threshold}")
                    names = self._parse_text_prompt_to_names(self._text_prompt)
                    if names:
                        txt_pe = self._model.get_text_pe(names)
                        self._model.set_classes(names, txt_pe)
                    results = self._model.predict(frame_bgr, device=self.device, conf=self.conf_threshold)
                if results is None:
                    return self.detect_for_ui(frame_bgr)
                dets = self._extract_ui_detections(results)
                # 若無任何偵測，退回 bbox-only 流程
                if not dets:
                    self._log("mask path yielded 0 dets, fallback to bbox-only path")
                    return self.detect_for_ui(frame_bgr)
                return dets
            except Exception as e:
                self._log(f"detect_for_ui_with_masks error: {repr(e)}")
        # 退回 bbox-only
        return self.detect_for_ui(frame_bgr)

    # --- internal helpers ---

    def _detect_by_text_prompt(self, frame_bgr: np.ndarray) -> Optional[List[Tuple[int, int, int, int]]]:
        text = (self._text_prompt or "").strip()
        if not text:
            return None

        # 若有 YOLOE，使用 text prompt 設定類別後推論
        if self._model is None:
            return None

        try:
            names = self._parse_text_prompt_to_names(text)
            if names:
                txt_pe = self._model.get_text_pe(names)
                self._model.set_classes(names, txt_pe)
            results = self._model.predict(frame_bgr, device=self.device, conf=self.conf_threshold)
            boxes = self._extract_xyxy(results)
            return boxes or None
        except Exception as e:
            self._log(f"text path error: {repr(e)}")
            return None

    # --- utils ---
    @staticmethod
    def _match_label_with_text(label: str, text: str) -> bool:
        label = (label or "").lower().strip()
        text = (text or "").lower().strip()
        if not label or not text:
            return False
        return label in text or text in label

    @staticmethod
    def _to_xyxy(box: Union[Tuple[int, int, int, int], Tuple[float, float, float, float]]) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        return int(x1), int(y1), int(x2), int(y2)

    def _extract_xyxy(self, results: Any) -> List[Tuple[int, int, int, int]]:
        boxes_out: List[Tuple[int, int, int, int]] = []
        try:
            if not results:
                return boxes_out
            res0 = results[0]
            boxes = getattr(res0, "boxes", None)
            if boxes is None:
                return boxes_out
            xyxy = getattr(boxes, "xyxy", None)
            if xyxy is None:
                return boxes_out
            try:
                arr = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.asarray(xyxy)
            except Exception:
                arr = np.asarray(xyxy)
            for b in arr:
                if len(b) >= 4:
                    x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                    boxes_out.append((int(x1), int(y1), int(x2), int(y2)))
        except Exception as e:
            self._log(f"_extract_xyxy error: {repr(e)}")
            return boxes_out
        return boxes_out

    @staticmethod
    def _parse_text_prompt_to_names(text: str) -> List[str]:
        # 以逗號/換行/分號切分；空白去除
        raw = [s.strip() for s in (text or "").replace("\n", ",").replace(";", ",").split(",")]
        return [s for s in raw if s]

    def _extract_ui_detections(self, results: Any) -> List[Dict[str, Any]]:
        """將 Ultralytics Results 轉為 UI 可用的 detection 結構，含可選的 masks(poly)。"""
        out: List[Dict[str, Any]] = []
        try:
            if not results:
                return out
            res0 = results[0]
            boxes = getattr(res0, "boxes", None)
            masks = getattr(res0, "masks", None)
            labels = None
            scores = None
            if boxes is not None:
                xyxy = getattr(boxes, "xyxy", None)
                conf = getattr(boxes, "conf", None)
                cls = getattr(boxes, "cls", None)
                try:
                    xyxy = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.asarray(xyxy)
                except Exception:
                    xyxy = np.asarray(xyxy)
                try:
                    scores = conf.cpu().numpy() if hasattr(conf, "cpu") else np.asarray(conf)
                except Exception:
                    scores = None
                try:
                    labels = cls.cpu().numpy().astype(int) if hasattr(cls, "cpu") else np.asarray(cls).astype(int)
                except Exception:
                    labels = None

                polys = None
                if masks is not None:
                    # 取得多邊形 xy（每個為 Nx2）
                    polys = getattr(masks, "xy", None)
                for i, b in enumerate(xyxy or []):
                    try:
                        x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                        det: Dict[str, Any] = {
                            "label": str(labels[i]) if (labels is not None and i < len(labels)) else (self._text_prompt or "obj"),
                            "box": (int(x1), int(y1), int(x2), int(y2)),
                            "confidence": float(scores[i]) if (scores is not None and i < len(scores)) else 1.0,
                        }
                        if polys is not None and i < len(polys) and isinstance(polys[i], list) and len(polys[i]) > 0:
                            # 取第一段多邊形
                            poly0 = polys[i][0]
                            try:
                                poly_pts = [(int(p[0]), int(p[1])) for p in poly0]
                                det["mask"] = poly_pts
                            except Exception:
                                pass
                        out.append(det)
                    except Exception:
                        continue
        except Exception as e:
            self._log(f"_extract_ui_detections error: {repr(e)}")
            return out
        return out

    def _log(self, msg: str) -> None:
        if self._debug:
            try:
                self._logger.info(msg)
            except Exception:
                print(msg)

    # SAM2 互動式提示
    def set_sam_prompts(self, points: List[Tuple[int, int]], labels: List[int], roi_box: Optional[Tuple[int, int, int, int]] = None) -> None:
        """設定 SAM2 的互動點與 ROI。labels: 1=正、0=負。"""
        self._sam_points = [(int(x), int(y)) for (x, y) in (points or [])]
        self._sam_labels = [int(v) for v in (labels or [])]
        self._sam_roi = tuple(int(v) for v in roi_box) if isinstance(roi_box, (tuple, list)) and len(roi_box) == 4 else None
        self._log(f"SAM2 prompts set: points={self._sam_points} labels={self._sam_labels} roi={self._sam_roi}")

    def preview_sam(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        """對單張影像做 SAM 分割預覽，輸出 masks 與可視化（RGB）。"""
        out: Dict[str, Any] = {"masks": None, "vis": None}
        if self._sam_model is None or image_rgb is None or not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
            self._log("preview_sam: SAM model not available or invalid input image")
            return out
        
        self._log(f"preview_sam: input image shape={image_rgb.shape}, dtype={image_rgb.dtype}")
        try:
            import cv2
            src = image_rgb  # 偏好 RGB
            pts = list(self._sam_points) if hasattr(self, "_sam_points") else []
            lbl = list(self._sam_labels) if hasattr(self, "_sam_labels") else []
            roi = self._sam_roi if hasattr(self, "_sam_roi") else None
            kwargs: Dict[str, Any] = {}
            if pts and lbl and len(pts) == len(lbl):
                # 使用雙層 list（每張圖一組點），以符合 Ultralytics SAM 慣例
                kwargs["points"] = [pts]
                kwargs["labels"] = [lbl]
            if roi is not None:
                # Ultralytics SAM 參數為 bboxes（xyxy）
                kwargs["bboxes"] = [list(roi)]
            # Ultralytics SAM 不支援 multimask_output，移除該參數
            # 嘗試帶入上一輪 logits（若支援且尺寸相容）
            try:
                if getattr(self, "_sam_prev_logits", None) is not None:
                    kwargs["mask_input"] = self._sam_prev_logits
            except Exception:
                pass
            # 使用 predict 以便傳 numpy 陣列
            self._log(f"preview_sam: calling SAM predict with kwargs={kwargs}")
            res = self._sam_model.predict(source=src, **kwargs)
            r = res[0] if isinstance(res, (list, tuple)) and len(res) > 0 else res
            masks = getattr(r, "masks", None)
            mask_data = getattr(masks, "data", None)
            self._log(f"preview_sam: SAM result masks={masks is not None}, mask_data shape={getattr(mask_data, 'shape', None) if mask_data is not None else None}")

            # Ultralytics SAM 預設只回傳一個遮罩，直接使用
            if mask_data is not None and hasattr(mask_data, 'shape'):
                if len(mask_data.shape) >= 3 and mask_data.shape[0] > 1:
                    # 若有多個遮罩，取第一個
                    out["masks"] = mask_data[0:1]
                    self._log(f"SAM preview: using first mask from {mask_data.shape[0]} candidates")
                else:
                    out["masks"] = mask_data
                    self._log(f"SAM preview: single mask output")
            else:
                out["masks"] = mask_data
                
            try:
                # 手動繪製單一遮罩，避免重疊
                if out["masks"] is not None and hasattr(out["masks"], 'shape'):
                    vis = src.copy()
                    mask = out["masks"][0] if len(out["masks"].shape) >= 3 else out["masks"]
                    if hasattr(mask, 'cpu'):
                        mask = mask.cpu().numpy()
                    
                    # 確保遮罩與原始影像尺寸完全匹配
                    if mask.shape[:2] != vis.shape[:2]:
                        self._log(f"Resizing mask from {mask.shape[:2]} to {vis.shape[:2]}")
                        mask = cv2.resize(mask, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST)
                    
                    # 使用更靈活的閾值處理
                    if mask.dtype != np.uint8:
                        mask = (mask > 0.5).astype(np.uint8) * 255
                    else:
                        mask = (mask > 127).astype(np.uint8) * 255
                    
                    # 創建彩色遮罩 - 確保尺寸完全匹配
                    colored_mask = np.zeros_like(vis)
                    if mask.shape[:2] == vis.shape[:2]:
                        colored_mask[:, :, 1] = mask  # 綠色遮罩
                        # 疊加遮罩
                        vis = cv2.addWeighted(vis, 0.7, colored_mask, 0.3, 0)
                    else:
                        self._log(f"Mask size mismatch after resize: mask={mask.shape[:2]}, vis={vis.shape[:2]}")
                        # 如果尺寸還是不匹配，直接使用原始影像
                        vis = src.copy()
                    
                    out["vis"] = vis
                else:
                    vis = r.plot()
                    out["vis"] = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            except Exception as e:
                self._log(f"custom mask visualization failed: {repr(e)}")
                try:
                    vis = r.plot()
                    out["vis"] = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                except Exception:
                    out["vis"] = src
            # 嘗試保存最佳遮罩作為下一輪的 mask_input
            try:
                if out["masks"] is not None:
                    self._sam_prev_logits = out["masks"]  # 作為簡化的 logits 輸入
            except Exception:
                self._sam_prev_logits = None

            self._log(f"SAM preview: masks={None if out['masks'] is None else getattr(out['masks'], 'shape', None)}")
        except Exception as e:
            self._log(f"preview_sam error: {repr(e)}")
        return out

    def get_last_sam_mask(self) -> Optional[np.ndarray]:
        """回傳上一輪 SAM 遮罩的二值圖 (uint8 0/255)；若無則回傳 None。"""
        try:
            m = getattr(self, "_sam_prev_logits", None)
            if m is None:
                return None
            if hasattr(m, 'cpu'):
                m = m.cpu().numpy()
            if not isinstance(m, np.ndarray) or m.size == 0:
                return None
            mask = m[0] if (m.ndim == 3 and m.shape[0] >= 1) else m
            
            # 改進的遮罩處理邏輯
            if mask.dtype != np.uint8:
                # 使用更靈活的閾值處理
                if mask.max() <= 1.0:
                    mask = (mask > 0.5).astype(np.uint8) * 255
                else:
                    mask = (mask > 127).astype(np.uint8) * 255
            else:
                # 確保是二值遮罩
                mask = (mask > 127).astype(np.uint8) * 255
            
            # 添加調試信息
            self._log(f"get_last_sam_mask: shape={mask.shape}, dtype={mask.dtype}, unique_values={np.unique(mask)}")
            return mask
        except Exception:
            return None


