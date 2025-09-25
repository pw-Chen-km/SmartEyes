from typing import Tuple, Any, Optional

try:
	import cv2  # type: ignore
	import numpy as np  # type: ignore
except Exception:  # pragma: no cover
	cv2 = None  # type: ignore
	np = None  # type: ignore


def _resolve_video_path(video_input: Any) -> Optional[str]:
	"""將 Gradio Video 可能的回傳型別解析為檔案路徑字串。

	可能型別：
	- str: 直接是檔案路徑
	- dict: 可能包含 'name' 或 'path' 欄位
	"""
	if isinstance(video_input, str):
		return video_input
	if isinstance(video_input, dict):
		# Gradio 可能提供 {'name': '/path/to/file', ...} 或 {'path': '/path/to/file', ...}
		return video_input.get('path') or video_input.get('name')
	return None


def extract_first_frame(video_input: Any) -> Any:
	"""讀取影片並回傳第一幀畫面（RGB，numpy.ndarray）。讀取失敗時回傳 None。

	- video_input: 可為影片路徑或 Gradio Video 的回傳物件
	"""
	if cv2 is None or np is None:
		return None
	path = _resolve_video_path(video_input)
	if not path:
		return None
	cap = cv2.VideoCapture(path)
	try:
		ok, frame_bgr = cap.read()
		if not ok or frame_bgr is None:
			return None
		frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
		return frame_rgb
	finally:
		cap.release()


def crop_by_bbox(image: Any, bbox: Tuple[int, int, int, int]) -> Any:
	"""依 bbox 在影像上裁切區域，回傳裁切結果（RGB ndarray）。

	bbox: (x1, y1, x2, y2)
	"""
	if image is None or np is None:
		return None
	x1, y1, x2, y2 = bbox
	# 安全界限
	h, w = image.shape[:2]
	x1 = max(0, min(int(x1), w - 1))
	y1 = max(0, min(int(y1), h - 1))
	x2 = max(0, min(int(x2), w))
	y2 = max(0, min(int(y2), h))
	if x2 <= x1 or y2 <= y1:
		return None
	crop = image[y1:y2, x1:x2]
	return crop
