from typing import Tuple, List, Dict, Optional
import os
import cv2
import numpy as np
import json


def ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def make_video_writer(dst_path: str, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(dst_path, fourcc, fps, size)
    return writer


def put_text_bottom_right(image: np.ndarray, text: str, font_scale: float = 0.8,
                          max_width_ratio: float = 0.6, max_height_ratio: float = 0.8,
                          autoshrink: bool = True) -> np.ndarray:
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return image
    try:
        img = image.copy()
        h, w = img.shape[:2]
        margin = 10
        raw_text = (text or "").replace("\t", "    ")
        if not raw_text.strip():
            return img

        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2

        max_block_w = max(50, int(w * float(max_width_ratio)))
        max_block_h = max(40, int(h * float(max_height_ratio)))

        def wrap_lines(current_scale: float):
            max_w = max_block_w - margin * 2
            paragraphs = [p for p in raw_text.split("\n") if p.strip()]
            wrapped: List[str] = []
            for para in paragraphs:
                tokens = para.split(" ")
                line = ""
                for tk in tokens:
                    candidate = (line + (" " if line else "") + tk) if tk else line
                    size, _ = cv2.getTextSize(candidate, font, current_scale, thickness)
                    if size[0] <= max_w:
                        line = candidate
                    else:
                        if line:
                            wrapped.append(line)
                            line = ""
                        if tk:
                            acc = ""
                            for ch in tk:
                                cand2 = (acc + ch)
                                size2, _ = cv2.getTextSize(cand2, font, current_scale, thickness)
                                if size2[0] <= max_w:
                                    acc = cand2
                                else:
                                    if acc:
                                        wrapped.append(acc)
                                    acc = ch
                            if acc:
                                line = acc
                if line:
                    wrapped.append(line)
            if not wrapped:
                wrapped = [""]
            line_height = int(20 * current_scale)
            max_line_w = 0
            for ln in wrapped:
                sz, _ = cv2.getTextSize(ln, font, current_scale, thickness)
                if sz[0] > max_line_w:
                    max_line_w = sz[0]
            block_w = max_line_w + margin * 2
            block_h = int(line_height * len(wrapped) + margin * 2)
            return wrapped, block_w, block_h

        lines, block_w, block_h = wrap_lines(font_scale)
        if autoshrink:
            scale = float(font_scale)
            tries = 0
            while (block_w > max_block_w or block_h > max_block_h) and scale > 0.3 and tries < 10:
                scale *= 0.9
                lines, block_w, block_h = wrap_lines(scale)
                tries += 1
            font_scale = scale

        line_height = int(20 * font_scale)
        max_lines_fit = max(1, (max_block_h - margin * 2) // max(1, line_height))
        if len(lines) > max_lines_fit:
            lines = lines[:max_lines_fit]
            if lines:
                ell = "..."
                base = lines[-1]
                while True:
                    test = (base + ell)
                    sz, _ = cv2.getTextSize(test, font, font_scale, thickness)
                    if sz[0] <= (max_block_w - margin * 2) or not base:
                        lines[-1] = test if base else ell
                        break
                    base = base[:-1]
        max_line_w = 0
        for ln in lines:
            sz, _ = cv2.getTextSize(ln, font, font_scale, thickness)
            if sz[0] > max_line_w:
                max_line_w = sz[0]
        block_w = min(max_block_w, max_line_w + margin * 2)
        block_h = min(max_block_h, int(line_height * len(lines) + margin * 2))

        x1 = max(margin, w - block_w - margin)
        y1 = max(margin, h - block_h - margin)
        x2 = min(w - margin, x1 + block_w)
        y2 = min(h - margin, y1 + block_h)

        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
        img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

        y = y1 + margin + line_height
        for ln in lines:
            cv2.putText(img, ln, (x1 + margin, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            y += line_height
        return img
    except Exception:
        return image


def overlay_sam_mask(frame_bgr: np.ndarray, mask_bin: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), alpha: float = 0.35) -> np.ndarray:
    try:
        if frame_bgr is None or mask_bin is None:
            return frame_bgr
        if not isinstance(frame_bgr, np.ndarray) or not isinstance(mask_bin, np.ndarray):
            return frame_bgr
        if frame_bgr.size == 0 or mask_bin.size == 0:
            return frame_bgr
        h, w = frame_bgr.shape[:2]
        mb = mask_bin
        if mb.shape[0] != h or mb.shape[1] != w:
            mb = cv2.resize(mb, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = (mb > 0).astype(np.uint8)
        if np.count_nonzero(mask) == 0:
            return frame_bgr
        colored = np.zeros_like(frame_bgr)
        colored[:, :] = color
        blended = cv2.addWeighted(frame_bgr, 1.0, colored, alpha, 0)
        mask3 = np.repeat(mask[:, :, None], 3, axis=2)
        out = np.where(mask3 > 0, blended, frame_bgr)
        return out
    except Exception:
        return frame_bgr


def concat_images_no_resize(img1: np.ndarray, img2: np.ndarray, axis: int = 1, pad_color: Tuple[int, int, int] = (0, 0, 0)) -> Optional[np.ndarray]:
    try:
        if img1 is None or img2 is None:
            return None
        if not isinstance(img1, np.ndarray) or not isinstance(img2, np.ndarray):
            return None
        if img1.ndim != 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        if img2.ndim != 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        if axis == 1:
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            max_h = max(h1, h2)
            if h1 != max_h:
                pad1 = np.full((max_h - h1, w1, 3), pad_color, dtype=img1.dtype)
                img1p = np.vstack([img1, pad1])
            else:
                img1p = img1
            if h2 != max_h:
                pad2 = np.full((max_h - h2, w2, 3), pad_color, dtype=img2.dtype)
                img2p = np.vstack([img2, pad2])
            else:
                img2p = img2
            return np.hstack([img1p, img2p])
        else:
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            max_w = max(w1, w2)
            if w1 != max_w:
                pad1 = np.full((h1, max_w - w1, 3), pad_color, dtype=img1.dtype)
                img1p = np.hstack([img1, pad1])
            else:
                img1p = img1
            if w2 != max_w:
                pad2 = np.full((h2, max_w - w2, 3), pad_color, dtype=img2.dtype)
                img2p = np.hstack([img2, pad2])
            else:
                img2p = img2
            return np.vstack([img1p, img2p])
    except Exception:
        return None


def draw_status_panel(frame_bgr: np.ndarray, panel_mode: str, panel_message: str) -> np.ndarray:
    try:
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
            return frame_bgr
        img = frame_bgr.copy()
        h, w = img.shape[:2]

        margin = 14
        panel_w = int(max(260, min(420, w * 0.32)))
        panel_h = int(max(180, min(320, h * 0.42)))
        x1 = w - panel_w - margin
        y1 = margin
        x2 = w - margin
        y2 = y1 + panel_h

        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
        img = cv2.addWeighted(img, 0.65, overlay, 0.35, 0)

        title_color = (0, 255, 255)
        text_color = (220, 220, 220)
        red = (0, 0, 255)
        green = (0, 200, 0)
        orange = (0, 165, 255)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale_title = 1.0
        scale_text = 0.9
        thick = 2

        cv2.putText(img, "SYSTEM STATUS", (x1 + 10, y1 + 26), font, scale_title, title_color, thick, cv2.LINE_AA)
        cv2.line(img, (x1 + 10, y1 + 34), (x2 - 10, y1 + 34), (90, 90, 90), 1, cv2.LINE_AA)

        ty = y1 + 58
        lh = 24
        lines: List[Tuple[str, Tuple[int, int, int]]] = []  # type: ignore[assignment]
        if panel_mode == "red":
            lines.append(("RED ALERT!", red))
            lines.append((panel_message or "person leave with item", (255, 255, 255)))
        elif panel_mode == "orange":
            lines.append(("ALERT", orange))
            lines.append((panel_message or "Item being taken", (255, 255, 255)))
        elif panel_mode == "green":
            lines.append((panel_message or "SAFE", green))
        else:
            lines.append(("Monitoring...", text_color))

        for text, color in lines:
            cv2.putText(img, text, (x1 + 12, ty), font, scale_text, color, thick, cv2.LINE_AA)
            ty += lh

        sq = int(min(panel_h - 56, panel_w * 0.48))
        sq = max(60, min(180, sq))
        sq = max(30, sq // 2)
        sx2 = x2 - 12
        sx1 = sx2 - sq
        sy1 = y1 + 44
        sy2 = sy1 + sq
        color = (40, 40, 40)
        if panel_mode == "red":
            color = red
        elif panel_mode == "orange":
            color = orange
        elif panel_mode == "green":
            color = green
        cv2.rectangle(img, (sx1, sy1), (sx2, sy2), color, thickness=-1)
        return img
    except Exception:
        return frame_bgr


def save_kframes(debug_dir: str, stem: str, event_index: int, frames_for_vlm: List[np.ndarray]) -> Dict[str, List[str]]:
    paths: Dict[str, List[str]] = {"k1k2": []}
    try:
        ensure_dir(debug_dir)
        k1_frame = frames_for_vlm[0] if len(frames_for_vlm) > 0 else None
        k2_frame = frames_for_vlm[1] if len(frames_for_vlm) > 1 else None
        if k1_frame is not None and k2_frame is not None:
            combined = concat_images_no_resize(k1_frame, k2_frame, axis=1)
            if combined is None:
                try:
                    h2 = k2_frame.shape[0]
                    k1r = cv2.resize(k1_frame, (int(k1_frame.shape[1] * (h2 / k1_frame.shape[0])), h2))
                    combined = np.hstack([k1r, k2_frame])
                except Exception:
                    try:
                        w2 = k2_frame.shape[1]
                        k1r = cv2.resize(k1_frame, (w2, int(k1_frame.shape[0] * (w2 / k1_frame.shape[1]))))
                        combined = np.vstack([k1r, k2_frame])
                    except Exception:
                        combined = None
            if combined is not None:
                p3 = os.path.join(debug_dir, f"{stem}_e{event_index}_k1k2.jpg")
                cv2.imwrite(p3, combined)
                paths["k1k2"].append(p3)
    except Exception:
        pass
    return paths






def save_precheck_log(debug_dir: str, event_index: int,
                      k1_summary: str, k1_decision: str,
                      k2_summary: str, k2_decision: str) -> Optional[str]:
    try:
        ensure_dir(debug_dir)
        data: Dict[str, object] = {
            "event_index": int(event_index),
            "k1": {"summary": str(k1_summary or ""), "decision": str(k1_decision or "")},
            "k2": {"summary": str(k2_summary or ""), "decision": str(k2_decision or "")},
            "passed": bool((k1_decision == "yes") and (k2_decision == "yes")),
        }
        path = os.path.join(debug_dir, f"precheck_log_{event_index:03d}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path
    except Exception:
        return None


def save_precheck_images(debug_dir: str, event_index: int,
                         k1_img: Optional[np.ndarray], k2_img: Optional[np.ndarray]) -> Dict[str, Optional[str]]:
    paths: Dict[str, Optional[str]] = {"k1k2": None, "k1": None, "k2": None}
    try:
        ensure_dir(debug_dir)
        has_k1 = isinstance(k1_img, np.ndarray) and k1_img.size > 0
        has_k2 = isinstance(k2_img, np.ndarray) and k2_img.size > 0

        if has_k1 and has_k2:
            combined = concat_images_no_resize(k1_img, k2_img, axis=1)
            if combined is None:
                try:
                    h2 = k2_img.shape[0]
                    k1r = cv2.resize(k1_img, (int(k1_img.shape[1] * (h2 / k1_img.shape[0])), h2))
                    combined = np.hstack([k1r, k2_img])
                except Exception:
                    try:
                        w2 = k2_img.shape[1]
                        k1r = cv2.resize(k1_img, (w2, int(k1_img.shape[0] * (w2 / k1_img.shape[1]))))
                        combined = np.vstack([k1r, k2_img])
                    except Exception:
                        combined = None
            if combined is not None:
                p = os.path.join(debug_dir, f"precheck_k1k2_{event_index:03d}.jpg")
                cv2.imwrite(p, combined)
                paths["k1k2"] = p
        elif has_k1 and not has_k2:
            p1 = os.path.join(debug_dir, f"precheck_k1_{event_index:03d}.jpg")
            cv2.imwrite(p1, k1_img)
            paths["k1"] = p1
        elif has_k2 and not has_k1:
            p2 = os.path.join(debug_dir, f"precheck_k2_{event_index:03d}.jpg")
            cv2.imwrite(p2, k2_img)
            paths["k2"] = p2
    except Exception:
        pass
    return paths



def append_event_summary(debug_dir: str, event_index: int, decision: str, summary: str) -> Optional[str]:
    """將每個事件（e1/e2/e3…）的最終 VLM 判斷附加寫入單一檔案（JSONL）。"""
    try:
        ensure_dir(debug_dir)
        path = os.path.join(debug_dir, "events_summary.jsonl")
        entry: Dict[str, object] = {
            "event": f"e{int(event_index)}",
            "event_index": int(event_index),
            "decision": str(decision or ""),
            "summary": str(summary or ""),
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return path
    except Exception:
        return None


# 簡易頂左角徽章標註（含半透明背景）
def _draw_badge_top_left(image: np.ndarray, text: str,
                         fg_color: Tuple[int, int, int] = (0, 0, 255),
                         bg_color: Tuple[int, int, int] = (0, 0, 0),
                         alpha: float = 0.35) -> np.ndarray:
    try:
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return image
        img = image.copy()
        h, w = img.shape[:2]
        margin = 12
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 4.0  # 放大四倍
        thick = 8  # 對應放大厚度
        label = (text or "").strip()
        if not label:
            return img
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)
        bw = tw + margin * 2
        bh = th + baseline + margin * 2
        x1, y1 = margin, margin
        x2, y2 = min(w - margin, x1 + bw), min(h - margin, y1 + bh)
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, thickness=-1)
        img = cv2.addWeighted(img, 1.0, overlay, float(alpha), 0)
        tx = x1 + margin
        ty = y1 + margin + th
        cv2.putText(img, label, (tx, ty), font, scale, fg_color, thick, cv2.LINE_AA)
        return img
    except Exception:
        return image


def annotate_image_file(src_path: str, text: str,
                        color: Tuple[int, int, int] = (0, 0, 255),
                        out_path: Optional[str] = None) -> Optional[str]:
    """在檔案上方左側加上徽章文字。預設就地覆寫。"""
    try:
        if not isinstance(src_path, str) or not os.path.isfile(src_path):
            return None
        img = cv2.imread(src_path)
        if img is None or img.size == 0:
            return None
        annotated = _draw_badge_top_left(img, str(text or ""), fg_color=color)
        dst = out_path if (isinstance(out_path, str) and out_path.strip()) else src_path
        ok = cv2.imwrite(dst, annotated)
        return dst if ok else None
    except Exception:
        return None
