from typing import Tuple, Optional, List

import numpy as np
import cv2


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def _center_crop_box(
    img_w: int,
    img_h: int,
    margin_ratio: float = 0.15,
    min_size_px: int = 300,
    max_size_px: int = 300,
    make_square: bool = False,
) -> Tuple[int, int, int, int]:
    """
    Calculate crop box based on image center with min/max size constraints.

    Rules:
    - Use image center as reference point.
    - Base size is 100, margin is min(W,H) * margin_ratio.
    - Target size = clamp(base_size + 2*margin, [min_size_px, max_size_px]).
    - If make_square=True, force equal width/height; otherwise use same calculation for both.
    """
    cx = img_w // 2
    cy = img_h // 2

    base_size = 100
    margin_px = int(min(img_w, img_h) * float(margin_ratio))
    target_w = max(min_size_px, min(max_size_px, base_size + margin_px * 2))
    target_h = target_w if make_square else max(min_size_px, min(max_size_px, base_size + margin_px * 2))

    half_w = target_w // 2
    half_h = target_h // 2

    x1 = _clamp(cx - half_w, 0, max(0, img_w - 1))
    y1 = _clamp(cy - half_h, 0, max(0, img_h - 1))
    x2 = _clamp(cx + half_w, x1 + 1, img_w)
    y2 = _clamp(cy + half_h, y1 + 1, img_h)

    # Final safety check to avoid zero/negative area
    if x2 <= x1:
        x2 = _clamp(x1 + 1, 1, img_w)
    if y2 <= y1:
        y2 = _clamp(y1 + 1, 1, img_h)

    return int(x1), int(y1), int(x2), int(y2)


def _apply_box_crop(img_bgr: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    if img_bgr is None or not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0:
        return img_bgr
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = _clamp(x1, 0, max(0, w - 1))
    y1 = _clamp(y1, 0, max(0, h - 1))
    x2 = _clamp(x2, x1 + 1, w)
    y2 = _clamp(y2, y1 + 1, h)
    return img_bgr[y1:y2, x1:x2].copy()


def crop_k1_k2(
    k1: np.ndarray,
    k2: np.ndarray,
    roi_poly: List[Tuple[int, int]],
    yolo_poly: List[Tuple[int, int]],
    *,
    margin_ratio: float = 0.15,
    min_size_px: int = 300,
    max_size_px: int = 300,
    make_square: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    Crop k1 and k2 using the same crop box derived from the intersection of ROI and YOLO polygons.

    Fallback: if intersection is empty, use ROI's bounding box.

    Parameters:
    - k1, k2: Image arrays (H,W,C)
    - roi_poly: list of (x,y) tuples in pixel coordinates
    - yolo_poly: list of (x,y) tuples in pixel coordinates
    - margin_ratio: expansion ratio relative to min(W,H)
    - min_size_px, max_size_px: min/max side length constraints
    - make_square: force square crop if True

    Returns:
    - (k1_crop, k2_crop, box_xyxy)
    """
    if not isinstance(k1, np.ndarray) or k1.size == 0:
        raise ValueError("k1 is not a valid image array")
    if not isinstance(k2, np.ndarray) or k2.size == 0:
        raise ValueError("k2 is not a valid image array")

    h, w = k1.shape[:2]

    def _poly_to_mask(poly: List[Tuple[int, int]], width: int, height: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        if not isinstance(poly, (list, tuple)) or len(poly) < 3:
            return mask
        try:
            pts = np.array([(int(px), int(py)) for (px, py) in poly], dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)
        except Exception:
            return np.zeros((height, width), dtype=np.uint8)
        return mask

    def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        try:
            if mask is None or not isinstance(mask, np.ndarray) or mask.size == 0:
                return None
            ys, xs = np.where(mask > 0)
            if ys.size == 0 or xs.size == 0:
                return None
            x1 = int(np.min(xs)); y1 = int(np.min(ys))
            x2 = int(np.max(xs) + 1); y2 = int(np.max(ys) + 1)
            if x2 <= x1 or y2 <= y1:
                return None
            return (x1, y1, x2, y2)
        except Exception:
            return None

    def _build_box_from_base(base_box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        bx1, by1, bx2, by2 = [int(v) for v in base_box]
        bw = max(1, bx2 - bx1)
        bh = max(1, by2 - by1)
        cx = (bx1 + bx2) // 2
        cy = (by1 + by2) // 2
        margin_px = int(min(w, h) * float(margin_ratio))
        base_size = max(bw, bh)
        target_w = min(int(max_size_px), max(int(min_size_px), int(base_size + margin_px * 2)))
        target_h = target_w if make_square else min(int(max_size_px), max(int(min_size_px), int(base_size + margin_px * 2)))
        half_w = target_w // 2
        half_h = target_h // 2
        x1 = max(0, cx - half_w)
        y1 = max(0, cy - half_h)
        x2 = min(w, x1 + target_w)
        y2 = min(h, y1 + target_h)
        if x2 <= x1:
            x2 = min(w, x1 + 1)
        if y2 <= y1:
            y2 = min(h, y1 + 1)
        return (int(x1), int(y1), int(x2), int(y2))

    # 1) 交集外接框；2) 無交集則使用 ROI 外接框
    roi_mask = _poly_to_mask(roi_poly or [], w, h)
    yolo_mask = _poly_to_mask(yolo_poly or [], w, h)
    inter_mask = ((roi_mask > 0) & (yolo_mask > 0)).astype(np.uint8)
    base_box = _bbox_from_mask(inter_mask)
    if base_box is None:
        base_box = _bbox_from_mask(roi_mask)

    if base_box is None:
        # 保底改用影像中心邏輯，避免失敗
        base_box = _center_crop_box(
            img_w=w,
            img_h=h,
            margin_ratio=float(margin_ratio),
            min_size_px=int(min_size_px),
            max_size_px=int(max_size_px),
            make_square=bool(make_square),
        )

    box = _build_box_from_base(base_box)
    k1_crop = _apply_box_crop(k1, box)
    k2_crop = _apply_box_crop(k2, box)
    return k1_crop, k2_crop, box


__all__ = ["crop_k1_k2"]


