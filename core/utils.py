from typing import List, Tuple, Optional, Any
import os
import json
import numpy as np
import cv2


def parse_points_from_str(s: Optional[str]) -> Tuple[List[Tuple[int, int]], List[int]]:
    points: List[Tuple[int, int]] = []
    labels: List[int] = []
    if not isinstance(s, str) or not s.strip():
        return points, labels
    try:
        items = [x.strip() for x in s.split(";") if x.strip()]
        for it in items:
            parts = [p.strip() for p in it.split(",")]
            if len(parts) >= 2:
                x = int(float(parts[0])); y = int(float(parts[1]))
                points.append((x, y))
                labels.append(1)
    except Exception:
        # 靜默失敗，回傳目前已解析內容
        pass
    return points, labels


def parse_points_from_json(path: Optional[str]) -> Tuple[List[Tuple[int, int]], List[int]]:
    points: List[Tuple[int, int]] = []
    labels: List[int] = []
    if not isinstance(path, str) or not path.strip() or not os.path.isfile(path):
        return points, labels
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pts = data.get("points") or []
        lbs = data.get("labels") or []
        for i, p in enumerate(pts):
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                x = int(p[0]); y = int(p[1])
                points.append((x, y))
                lab = int(lbs[i]) if (isinstance(lbs, list) and i < len(lbs)) else 1
                labels.append(1 if lab != 0 else 0)
    except Exception:
        pass
    return points, labels


def parse_roi_poly_from_str(s: Optional[str]) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    if not isinstance(s, str) or not s.strip():
        return points
    try:
        tokens = [t for t in s.replace(",", " ").split() if t]
        # 若為 YOLO segmentation 格式，首個可能是類別 id（整數）
        if len(tokens) % 2 == 1 and len(tokens) >= 7:
            tokens = tokens[1:]
        vals = [float(t) for t in tokens]
        if len(vals) % 2 != 0:
            vals = vals[:-1]
        for i in range(0, len(vals), 2):
            x = float(vals[i]); y = float(vals[i + 1])
            points.append((x, y))
    except Exception:
        pass
    return points


def parse_roi_poly_from_file(path: Optional[str]) -> List[Tuple[float, float]]:
    if not isinstance(path, str) or not path.strip() or not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            line = ""
            for ln in f:
                if ln.strip():
                    line = ln.strip()
                    break
        return parse_roi_poly_from_str(line)
    except Exception:
        return []


def yolo_polys_from_result(result: Any) -> List[List[Tuple[int, int]]]:
    polys: List[List[Tuple[int, int]]] = []
    try:
        masks = getattr(result, "masks", None)
        xy = getattr(masks, "xy", None) if masks is not None else None
        if isinstance(xy, list):
            for poly in xy:
                try:
                    arr = np.asarray(poly)
                    pts = [(int(p[0]), int(p[1])) for p in arr]
                    if len(pts) >= 3:
                        polys.append(pts)
                except Exception:
                    continue
    except Exception:
        pass
    return polys


def overlap_ratio_poly_with_mask(poly_pts: List[Tuple[int, int]], sam_mask_bin: np.ndarray, img_w: int, img_h: int) -> float:
    try:
        if sam_mask_bin is None or not isinstance(sam_mask_bin, np.ndarray) or sam_mask_bin.size == 0:
            return 0.0
        yolo_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        pts = np.array(poly_pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(yolo_mask, [pts], 255)
        inter = int(np.count_nonzero((yolo_mask > 0) & (sam_mask_bin > 0)))
        y_area = int(np.count_nonzero(yolo_mask))
        if y_area <= 0:
            return 0.0
        return float(inter) / float(y_area)
    except Exception:
        return 0.0









