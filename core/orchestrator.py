from typing import Optional, List, Tuple, Any, Dict, Callable
from queue import Queue, Full
from threading import Thread

from .perception import PerceptionEngine
from .reasoning import ReasoningEngine, GeminiReasoningEngine


class Orchestrator:
    def __init__(self, iou_threshold: float = 0.0, vlm_backend: str = "qwen") -> None:
        self.perception = PerceptionEngine()
        self.reasoning = None  # 將於 initialize_models 依 backend 建立
        self._vlm_backend = (vlm_backend or "qwen").strip().lower()

        # --- 事件佇列與推理執行緒 ---
        self._event_queue: "Queue[Dict[str, Any]]" = Queue(maxsize=4)
        self._result_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self._worker_started: bool = False
        self._pending_tasks: int = 0  # 尚未完成的 VLM 任務數

    def initialize_models(self) -> None:
        self.perception.initialize()
        # 依據 backend 建立推理引擎
        if self._vlm_backend in ("gemini", "google", "g"):
            self.reasoning = GeminiReasoningEngine()
        else:
            self.reasoning = ReasoningEngine()
        self.reasoning.initialize()
        self._start_worker_if_needed()

    def reset(self) -> None:
        # 不清空佇列，避免丟失已產生的事件；如需清空可提供專用 API
        pass

    def set_result_callback(self, fn: Optional[Callable[[Dict[str, Any]], None]]) -> None:
        """設定推理結果回呼（含一次性與未來的串流版本）。"""
        self._result_callback = fn

    def _start_worker_if_needed(self) -> None:
        if self._worker_started:
            return
        def _worker() -> None:
            while True:
                task = self._event_queue.get()
                try:
                    frames_for_vlm = task.get("frames", [])
                    prompt = task.get("prompt", "")
                    task_type = str(task.get("task_type", "vlm") or "vlm").lower()

                    if task_type == "precheck":
                        # 背景 precheck：通過才轉投 VLM 任務
                        result = self.reasoning.analyze_keyframes(frames_for_vlm, prompt=prompt)
                        # 解析決策（僅取首詞）
                        try:
                            summary = str((result or {}).get("summary", ""))
                            token = (summary.strip().lower().split()[0]) if summary else ""
                        except Exception:
                            token = ""
                            summary = str((result or {}).get("summary", ""))

                        if token in ("yes", "y", "true"):  # precheck 通過
                            # 先回報一個 precheck_passed 事件給上層
                            cb = self._result_callback
                            if cb is not None:
                                try:
                                    cb({
                                        "type": "precheck_passed",
                                        "event_index": task.get("event_index"),
                                        "debug_dir": task.get("debug_dir"),
                                        "k1k2_path": task.get("k1k2_path"),
                                        "summary": summary,
                                    })
                                except Exception:
                                    pass
                            # 再把同一批 frames 投遞為 VLM 任務
                            try:
                                self._event_queue.put_nowait({
                                    "task_type": "vlm",
                                    "frames": frames_for_vlm,
                                    "prompt": task.get("vlm_prompt", ""),
                                    "event_index": task.get("event_index"),
                                    "debug_dir": task.get("debug_dir"),
                                    "k1k2_path": task.get("k1k2_path"),
                                })
                            except Exception:
                                pass
                        else:
                            # precheck 未通過：不再投遞 VLM；可選擇於上層標記為 Filtered
                            pass
                    else:
                        # VLM 任務（原行為）
                        result = self.reasoning.analyze_keyframes(frames_for_vlm, prompt=prompt)
                        # 夾帶任務的 metadata（如 event_index, debug_dir）回傳給 callback
                        try:
                            if isinstance(result, dict):
                                if "event_index" not in result and "event_index" in task:
                                    result["event_index"] = task.get("event_index")
                                if "debug_dir" not in result and "debug_dir" in task:
                                    result["debug_dir"] = task.get("debug_dir")
                                if "k1k2_path" not in result and "k1k2_path" in task:
                                    result["k1k2_path"] = task.get("k1k2_path")
                        except Exception:
                            pass
                        cb = self._result_callback
                        if cb is not None:
                            try:
                                cb(result)
                            except Exception:
                                pass
                except Exception:
                    pass
                finally:
                    try:
                        self._event_queue.task_done()
                    except Exception:
                        pass
                    # 完成一個任務
                    try:
                        self._pending_tasks = max(0, int(self._pending_tasks) - 1)
                    except Exception:
                        self._pending_tasks = 0
        t = Thread(target=_worker, daemon=True)
        t.start()
        self._worker_started = True

    def wait_all(self, timeout_sec: Optional[float] = None) -> None:
        """等待所有已送出的 VLM 任務完成。可選 timeout。"""
        import time
        start = time.time()
        while True:
            try:
                if int(self._pending_tasks) <= 0 and self._event_queue.unfinished_tasks == 0:
                    break
            except Exception:
                break
            if timeout_sec is not None and (time.time() - start) > timeout_sec:
                break
            time.sleep(0.05)

    def set_prompts(self, visual_template_image, text_prompt: str) -> None:
        self.perception.set_prompts(visual_template_image=visual_template_image, text_prompt=text_prompt)

    # SAM2 互動：設定點/ROI
    def set_sam_prompts(self, points: List[Tuple[int, int]], labels: List[int], roi_box: Optional[Tuple[int, int, int, int]] = None) -> None:
        self.perception.set_sam_prompts(points=points, labels=labels, roi_box=roi_box)

    # SAM2 單張預覽
    def preview_sam(self, image_rgb: Any) -> Dict[str, Any]:
        return self.perception.preview_sam(image_rgb)

    # 簡化的 orchestrator，只負責模型管理和 VLM 佇列
    # 觸發邏輯已移至 pipeline.py
