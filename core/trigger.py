from typing import Optional, Tuple
import numpy as np


class InteractionTrigger:
    def __init__(self, iou_threshold: float = 0.0, consecutive_frames: int = 10) -> None:
        self.iou_threshold = iou_threshold
        self.consecutive_frames = max(1, int(consecutive_frames))
        self._last_iou: Optional[float] = None
        self._consecutive_count: int = 0

    @staticmethod
    def compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area
        if union == 0:
            return 0.0
        return inter_area / union

    @staticmethod
    def compute_mask_bbox_iou(mask: np.ndarray, box: Tuple[int, int, int, int]) -> float:
        """
        計算二值遮罩與單一 bbox 的 IoU。遮罩需為 uint8 或 bool，非零即視為前景。
        - 若遮罩與 bbox 任一無效，回傳 0.0。
        - 將 bbox 區域裁切至遮罩尺寸內再計算。
        """
        if mask is None or not isinstance(mask, np.ndarray) or mask.size == 0:
            return 0.0
        if mask.ndim != 2:
            # 若是 3D（如 1xHxW），嘗試 squeeze
            try:
                mask = np.squeeze(mask)
            except Exception:
                return 0.0
            if mask.ndim != 2:
                return 0.0
        h, w = mask.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        # 規範化座標並裁切至影像內
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return 0.0

        # 轉為 bool
        m = mask.astype(bool) if mask.dtype != bool else mask

        # bbox 範圍內的遮罩
        crop = m[y1:y2, x1:x2]
        inter = int(np.count_nonzero(crop))
        if inter == 0:
            return 0.0

        box_area = int((x2 - x1) * (y2 - y1))
        if box_area <= 0:
            return 0.0

        # 以 bbox 像素數作為分母（overlap ratio），或可改為 IoU：
        # iou = inter / (box_area + mask_area - inter)
        # 但此處採用對觸發更靈敏的 overlap ratio（相對 bbox）
        overlap_ratio = inter / float(box_area)
        return float(overlap_ratio)

    def update(self, iou: Optional[float]) -> bool:
        self._last_iou = iou
        # 僅當 overlap 嚴格大於門檻才累計；門檻 0.0 代表需有正交集
        if iou is None or iou <= self.iou_threshold:
            self._consecutive_count = 0
            return False
        self._consecutive_count += 1
        return self._consecutive_count >= self.consecutive_frames

    def reset(self) -> None:
        self._last_iou = None
        self._consecutive_count = 0

    # 供 UI 顯示與外部調整
    def get_consecutive_count(self) -> int:
        return int(self._consecutive_count)

    def set_consecutive_frames(self, n: int) -> None:
        try:
            self.consecutive_frames = max(1, int(n))
        except Exception:
            self.consecutive_frames = 10
