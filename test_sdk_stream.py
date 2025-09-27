import argparse
import time
import logging
import os

import cv2

from core.sdk import (
    create_session,
    process_frame,
    register_event_callback,
    close_session,
)
from core.pipeline import PipelineConfig
from core.utils import parse_roi_poly_from_str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test streaming SDK by feeding frames from a video file",
    )
    parser.add_argument(
        "--video",
        type=str,
        default="/Users/patrick/Downloads/SmartEyes_v2/Usman.mp4",
        help="Path to the test video file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/jj/PW/SmartEyes_v2/outputs",
        help="Directory to store outputs (k_img/, video/)",
    )
    parser.add_argument(
        "--write_video",
        action="store_true",
        help="Write annotated output video via SDK",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live annotated frames in a window",
    )
    parser.add_argument(
        "--roi_fullframe",
        action="store_true",
        help="Use full-frame ROI polygon (1.0 coverage). Recommended if you didn't set SAM points",
    )
    parser.add_argument(
        "--roi_px",
        type=str,
        default="",
        help="Pixel ROI polygon points. Accepts JSON like '[[x,y],...]' or flat 'x1 y1 x2 y2 ...'",
    )
    parser.add_argument(
        "--neg_roi_px",
        type=str,
        default="",
        help="Pixel NEG-ROI polygon points. Same format as --roi_px",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s")

    if not os.path.isfile(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    print(f"Video opened: {args.video} ({width}x{height} @ {fps:.2f} FPS)")

    # Prepare config for streaming mode
    roi_poly_norm = None
    neg_roi_poly_norm = None
    if args.roi_fullframe:
        # Full-frame ROI; if tracker可用，person與ROI重疊將觸發事件
        roi_poly_norm = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    # 支援從 CLI 直接輸入像素點（首幀會自動normalize）
    if isinstance(args.roi_px, str) and args.roi_px.strip():
        try:
            roi_poly_norm = parse_roi_poly_from_str(args.roi_px)
        except Exception:
            roi_poly_norm = roi_poly_norm
    if isinstance(args.neg_roi_px, str) and args.neg_roi_px.strip():
        try:
            neg_roi_poly_norm = parse_roi_poly_from_str(args.neg_roi_px)
        except Exception:
            neg_roi_poly_norm = neg_roi_poly_norm

    cfg = PipelineConfig(
        input_path=args.video,  # streaming 模式下不使用，但欄位必填
        output_dir=args.output_dir,
        # 使用你提供的 SAM 提示點與標籤
        points=[(280, 259)],
        labels=[1],
        iou_threshold=0.0,
        trigger_frames=5,  # 與 test_core.py 一致
        leave_patience=30,
        show_window=False,
        yolo_weights=None,  # 如需指定權重，可在此傳入或設定環境變數 YOLOE_WEIGHTS
        tracker_cfg=None,
        track_conf=0.35,
        batch_size=0,
        no_track=True,  # 若無YOLOE環境，SDK會自動降級為不追蹤（則通常不會觸發事件）
        track_interval=1,
        crop_k2=True,
        crop_margin_ratio=0.17,
        crop_min_size=100,
        crop_max_size=512,
        crop_square=False,
        vlm_backend="qwen",
        person_mask=False,
        mask_mode="keep",
        draw_sam=True,
        roi_poly_norm=roi_poly_norm,
        neg_roi_poly_norm=neg_roi_poly_norm,
        precheck_enabled=True,
        precheck_scale=2.0,
    )

    handle = create_session(cfg, enable_video_output=bool(args.write_video))

    def on_evt(evt):
        # evt.type: "status" | "precheck_passed" | "vlm_decision"
        print(f"[EVENT] type={evt.type} idx={evt.event_index} decision={evt.decision} summary={evt.summary}")
        if evt.type == "vlm_decision":
            print(f"  -> VLM 決策: {evt.decision.upper()}")
            if evt.k1k2_path:
                print(f"  -> k1k2 圖: {evt.k1k2_path}")

    register_event_callback(handle, on_evt)

    frame_count = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break
            frame_count += 1
            ts_ms = time.time() * 1000.0
            out = process_frame(handle, frame_bgr, ts_ms=ts_ms, fps_hint=fps)
            
            # 顯示當前面板狀態（每 30 幀顯示一次避免刷屏）
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: panel={out.panel_mode} msg='{out.panel_message}' vlm='{out.vlm_text[:50]}...'")
            
            # 顯示本幀產生的事件
            for evt in out.events:
                print(f"[FRAME {frame_count}] {evt.type} idx={evt.event_index} decision={evt.decision}")

            if args.show:
                try:
                    cv2.imshow("sdk_stream", out.overlay_bgr if out.overlay_bgr is not None else frame_bgr)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception:
                    pass
    finally:
        cap.release()
        try:
            if args.show:
                cv2.destroyAllWindows()
        except Exception:
            pass

    summary = close_session(handle)
    print("Done.")
    print(f"Summary: {summary}")
    print(f"Frames processed: {frame_count}")


if __name__ == "__main__":
    main()


