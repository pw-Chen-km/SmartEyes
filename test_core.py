from typing import List, Tuple, Optional
import argparse
import os
import logging

from core.pipeline import VisualMonitoringPipeline, PipelineConfig
from core.utils import (
    parse_points_from_str,
    parse_points_from_json,
    parse_roi_poly_from_str,
    parse_roi_poly_from_file,
)


LOGGER = logging.getLogger("test_core")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s")


DEFAULT_FIXED_ROI_POLY_NORM: List[Tuple[float, float]] = [
    (0.223087, 0.078426),
    (0.219978, 0.322132),
    (0.419772, 0.471985),
    (0.730337, 0.474794),
    (0.942543, 0.379559),
    (0.951859, 0.075632),
]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Modular visual monitoring pipeline (core version)")
    p.add_argument("--input", required=True, help="影片檔或資料夾路徑")
    p.add_argument("--output-dir", default="outputs", help="輸出資料夾")
    p.add_argument("--points", default="", help="SAM 點座標，格式 'x1,y1;x2,y2;...'，label 預設為 1")
    p.add_argument("--points-json", default="", help="JSON 檔，包含 {points:[[x,y],...], labels:[1,0,...]}（優先於 --points）")
    p.add_argument("--region", choices=["camera1", "camera2", "camera2_up", "camera3", "camera4", "camera4_up", "camera5", "test", "perfume"], default="", help="預設區域名稱，覆寫自訂點")
    p.add_argument("--iou-th", type=float, default=0.0, help="觸發重疊門檻（相對 bbox 面積）")
    p.add_argument("--trigger-frames", type=int, default=5, help="連續重疊幀數觸發")
    p.add_argument("--leave-patience", type=int, default=15, help="追蹤對象缺失多少幀視為離開")
    p.add_argument("--show", action="store_true", help="顯示即時視窗（按 q 離開）")
    p.add_argument("--yolo-weights", default=os.getenv("YOLOE_WEIGHTS", "yoloe-11l-seg.pt"), help="Ultralytics YOLOE 權重檔案")
    p.add_argument("--tracker-cfg", default=os.getenv("YOLOE_TRACKER_CFG", "bytetrack.yaml"), help="追蹤器設定檔（如 bytetrack.yaml）")
    p.add_argument("--track-conf", type=float, default=0.25, help="追蹤/偵測的信心閾值")
    p.add_argument("--batch-size", type=int, default=16, help="批次處理大小（>1 啟用批次模式）")
    p.add_argument("--no-track", action="store_true", help="停用追蹤（僅偵測）")
    p.add_argument("--track-interval", type=int, default=1, help="每 N 幀執行一次追蹤")
    p.add_argument("--vlm-backend", choices=["qwen", "gemini"], default="qwen", help="選擇視覺語言模型後端")
    p.add_argument("--crop-k2", action="store_true", help="啟用 k1/k2 影格裁切（以 ROI∩分割外擴）")
    p.add_argument("--crop-margin", type=float, default=0.17, help="裁切外擴像素數（例如 0.5 = 50px 外擴）")
    p.add_argument("--crop-min-size", type=int, default=50, help="裁切最小邊長（像素）")
    p.add_argument("--crop-max-size", type=int, default=1000, help="裁切最大邊長（像素）")
    p.add_argument("--crop-square", action="store_true", help="裁切時嘗試趨近正方形")
    p.add_argument("--person-mask", action="store_true", help="以 YOLOE bbox 觸發 SAM 生成人遮罩，套用於 k1/k2 裁切圖")
    p.add_argument("--mask-mode", choices=["keep", "overlay"], default="keep", help="人遮罩模式：keep=背景變暗, overlay=加半透明色")
    p.add_argument("--draw-sam", action="store_true", help="在輸出影片上疊加初始 SAM 遮罩")
    p.add_argument("--roi-poly", default="", help="以空白或逗號分隔的 YOLO 多邊形座標")
    p.add_argument("--roi-poly-file", default="", help="包含單行 YOLO 多邊形座標的文字檔路徑")
    # 前置檢查（預設啟用，可用 --no-precheck 關閉）
    p.add_argument("--no-precheck", action="store_true", help="關閉 K1/K2 前置檢查（由 VLM 單張確認手/ROI 關係）")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)

    pts: List[Tuple[int, int]] = []
    lbs: List[int] = []
    if isinstance(args.points_json, str) and args.points_json.strip():
        pts, lbs = parse_points_from_json(args.points_json)
    if not pts:
        p2, l2 = parse_points_from_str(args.points)
        if p2:
            pts, lbs = p2, l2

    if isinstance(args.region, str) and args.region.strip():
        reg = args.region.strip().lower()
        if reg == "camera1":
            pts = [(280, 259)]
            lbs = [1]
        elif reg == "camera2":
            pts =[(207, 230)] 
            lbs = [1]
        elif reg == "camera2_up":
            pts =[(119, 177), (117, 137), (91, 158)]
            lbs = [1, 1, 1]
        elif reg == "camera3":
            pts =[(160, 262), (135, 187), (138, 239)] 
            lbs = [1, 0, 0]
        elif reg == "camera4":
            pts = [(282, 267)]  # type: ignore[list-item]
            lbs = [1]
        elif reg == "camera4_up":
            pts =[(141, 160), (133, 124), (121, 145)]
            lbs = [1, 1, 1]
        elif reg == "camera5":
            pts =[(208, 284)]
            lbs = [1]
        elif reg == "test":
            pts = [[496, 438], [417, 113], [932, 100], [424, 335], [979, 337], [758, 475], [1268, 486], [1074, 283], [1701, 120], [1607, 389], [1620, 298], [1002, 484], [1067, 88], [918, 267], [1148, 375], [697, 197], [1516, 195], [877, 470], [588, 384], [1290, 368]]  # type: ignore[list-item]
            lbs = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        elif reg == "perfume":
            pts = [[459, 362], [427, 357]]  # type: ignore[list-item]
            lbs = [1, 0]
    LOGGER.info("SAM points: %d", len(pts))

    roi_poly_norm: Optional[List[Tuple[float, float]]] = None
    try:
        if isinstance(args.roi_poly_file, str) and args.roi_poly_file.strip():
            roi_poly_norm = parse_roi_poly_from_file(args.roi_poly_file)
        elif isinstance(args.roi_poly, str) and args.roi_poly.strip():
            roi_poly_norm = parse_roi_poly_from_str(args.roi_poly)
    except Exception:
        roi_poly_norm = None
    if not roi_poly_norm and not pts:
        roi_poly_norm = DEFAULT_FIXED_ROI_POLY_NORM

    cfg = PipelineConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        points=pts,
        labels=lbs,
        iou_threshold=args.iou_th,
        trigger_frames=args.trigger_frames,
        leave_patience=args.leave_patience,
        show_window=bool(args.show),
        yolo_weights=args.yolo_weights,
        tracker_cfg=args.tracker_cfg,
        track_conf=args.track_conf,
        batch_size=args.batch_size,
        no_track=bool(args.no_track),
        track_interval=args.track_interval,
        crop_k2=bool(args.crop_k2),
        crop_margin_ratio=args.crop_margin,
        crop_min_size=args.crop_min_size,
        crop_max_size=args.crop_max_size,
        crop_square=bool(args.crop_square),
        vlm_backend=str(args.vlm_backend),
        person_mask=bool(args.person_mask),
        mask_mode=str(args.mask_mode),
        draw_sam=bool(args.draw_sam),
        roi_poly_norm=roi_poly_norm,
        precheck_enabled=(not bool(args.no_precheck)),
    )

    pipeline = VisualMonitoringPipeline(cfg)
    pipeline.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


