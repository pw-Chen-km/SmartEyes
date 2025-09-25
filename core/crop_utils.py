from typing import List, Tuple, Optional
import numpy as np
import cv2


def _polys_to_mask(polys: List[List[Tuple[int, int]]], img_w: int, img_h: int) -> np.ndarray:
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if not isinstance(polys, list) or len(polys) == 0:
        return mask
    try:
        for poly in polys:
            try:
                if not isinstance(poly, (list, tuple)) or len(poly) < 3:
                    continue
                pts = np.array([(int(p[0]), int(p[1])) for p in poly], dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 255)
            except Exception:
                continue
    except Exception:
        pass
    return mask


def mask_bounding_box(mask_bin: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    try:
        if mask_bin is None or not isinstance(mask_bin, np.ndarray) or mask_bin.size == 0:
            return None
        ys, xs = np.where(mask_bin > 0)
        if ys.size == 0 or xs.size == 0:
            return None
        x1 = int(np.min(xs)); y1 = int(np.min(ys))
        x2 = int(np.max(xs) + 1); y2 = int(np.max(ys) + 1)
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)
    except Exception:
        return None


def compute_overlap_bbox(
    sam_mask_bin: Optional[np.ndarray],
    yolo_polys: Optional[List[List[Tuple[int, int]]]],
    img_w: int,
    img_h: int,
) -> Optional[Tuple[int, int, int, int]]:
    """
    以 SAM 二值遮罩與 YOLO 多邊形遮罩的交集求外接框。
    若多邊形不存在，退回 SAM 外接框。
    若兩者交集為空，退回 None。
    """
    try:
        if sam_mask_bin is None or not isinstance(sam_mask_bin, np.ndarray) or sam_mask_bin.size == 0:
            return None
        sam_bin = (sam_mask_bin > 0).astype(np.uint8)
        if yolo_polys and len(yolo_polys) > 0:
            y_mask = _polys_to_mask(yolo_polys, img_w, img_h)
            inter = ((sam_bin > 0) & (y_mask > 0)).astype(np.uint8)
            box = mask_bounding_box(inter)
            if box is not None:
                return box
        # fallback: 僅以 SAM
        return mask_bounding_box(sam_bin)
    except Exception:
        return None


def intersect_boxes(
    a: Optional[Tuple[int, int, int, int]],
    b: Optional[Tuple[int, int, int, int]],
) -> Optional[Tuple[int, int, int, int]]:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1); y1 = max(ay1, by1)
    x2 = min(ax2, bx2); y2 = min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def expand_and_clamp_box(
    box: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    margin_ratio: float = 0.18,
    base_min_margin_px: int = 12,
    min_size_px: int = 160,
    max_size_px: int = 512,
    make_square: bool = False,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in box]
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    # 計算外擴邊界
    margin = max(int(base_min_margin_px), int(float(margin_ratio) * float(max(w, h))))
    nx1 = max(0, x1 - margin)
    ny1 = max(0, y1 - margin)
    nx2 = min(img_w, x2 + margin)
    ny2 = min(img_h, y2 + margin)

    # 調整為接近正方（可選）
    if make_square:
        cw = nx2 - nx1; ch = ny2 - ny1
        side = max(cw, ch)
        # 嘗試以中心為基準擴充較短邊
        cx = (nx1 + nx2) // 2
        cy = (ny1 + ny2) // 2
        half = side // 2
        nx1 = max(0, cx - half)
        ny1 = max(0, cy - half)
        nx2 = min(img_w, nx1 + side)
        ny2 = min(img_h, ny1 + side)

    # 最小尺寸保障（盡量向外補齊）
    cw = nx2 - nx1; ch = ny2 - ny1
    if cw < min_size_px or ch < min_size_px:
        need_w = max(0, min_size_px - cw)
        need_h = max(0, min_size_px - ch)
        nx1 = max(0, nx1 - need_w // 2)
        nx2 = min(img_w, nx2 + (need_w - need_w // 2))
        ny1 = max(0, ny1 - need_h // 2)
        ny2 = min(img_h, ny2 + (need_h - need_h // 2))

    # 最大尺寸限制（不強制縮小內容，只裁到邊界）
    cw = nx2 - nx1; ch = ny2 - ny1
    if cw > max_size_px or ch > max_size_px:
        # 以中心裁到最大尺寸
        cx = (nx1 + nx2) // 2
        cy = (ny1 + ny2) // 2
        half_w = min(max_size_px // 2, img_w // 2)
        half_h = min(max_size_px // 2, img_h // 2)
        nx1 = max(0, cx - half_w)
        nx2 = min(img_w, cx + half_w)
        ny1 = max(0, cy - half_h)
        ny2 = min(img_h, cy + half_h)

    # 最終保險
    nx1 = max(0, min(nx1, img_w - 1))
    ny1 = max(0, min(ny1, img_h - 1))
    nx2 = max(nx1 + 1, min(nx2, img_w))
    ny2 = max(ny1 + 1, min(ny2, img_h))
    return (int(nx1), int(ny1), int(nx2), int(ny2))


def crop_by_box(frame_bgr: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    try:
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
            return frame_bgr
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, min(x1, w - 1))
        x2 = max(x1 + 1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(y1 + 1, min(y2, h))
        return frame_bgr[y1:y2, x1:x2].copy()
    except Exception:
        return frame_bgr


