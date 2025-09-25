from typing import List, Any, Dict, Optional
import os
import tempfile
import shutil
from pathlib import Path

import torch
import numpy as np
from PIL import Image

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
except Exception as import_error:  # 延遲到 initialize 再報錯
    Qwen2_5_VLForConditionalGeneration = None  # type: ignore
    AutoProcessor = None  # type: ignore
    process_vision_info = None  # type: ignore
    _IMPORT_ERROR = import_error
else:
    _IMPORT_ERROR = None

# --- Google Gemini SDK ---
try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None  # type: ignore


class ReasoningEngine:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", max_new_tokens: int = 128) -> None:
        self._initialized = False
        self._model_name = model_name
        self._max_new_tokens = max_new_tokens
        self._device: Optional[str] = None
        self._model = None
        self._processor = None
        self._prompt: str = "請描述這段影片中發生的事件與關鍵物件。"
        # 下採樣參數
        self._source_fps: Optional[float] = None
        self._target_sampling_fps: Optional[float] = None
        self._input_frame_stride: int = 1
        self._max_vlm_frames: Optional[int] = None

    def set_prompt(self, text_prompt: str) -> None:
        self._prompt = text_prompt

    def initialize(self) -> None:
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                f"缺少必要套件：{_IMPORT_ERROR}. 請先安裝 transformers、accelerate 與 qwen-vl-utils。"
            )

        has_cuda = torch.cuda.is_available()
        self._device = "cuda" if has_cuda else "cpu"

        # 載入模型與處理器
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self._model_name,
            torch_dtype="auto",
            device_map="auto" if has_cuda else None,
        )
        # 可選擇影像縮放範圍，維持合理的 VRAM 需求
        self._processor = AutoProcessor.from_pretrained(
            self._model_name,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )

        self._initialized = True

    def _ensure_initialized(self) -> None:
        if not self._initialized or self._model is None or self._processor is None:
            raise RuntimeError("ReasoningEngine 尚未初始化，請先呼叫 initialize()。")

    # 供外部設定來源影片 fps（若可得），以及目標下採樣 fps
    def set_source_fps(self, fps: float) -> None:
        if fps > 0:
            self._source_fps = float(fps)

    def set_sampling_fps(self, fps: float) -> None:
        if fps > 0:
            self._target_sampling_fps = float(fps)

    # 以固定步長下採樣（若未提供 fps 時使用）
    def set_sampling_stride(self, stride: int) -> None:
        self._input_frame_stride = max(1, int(stride))

    # 限制最終餵給 VLM 的影格上限
    def set_max_vlm_frames(self, max_frames: int) -> None:
        self._max_vlm_frames = max(1, int(max_frames))

    def _save_frames_to_temp_images(self, frames: List[Any]) -> List[str]:
        temp_dir = tempfile.mkdtemp(prefix="qwen_vl_frames_")
        uris: List[str] = []

        for idx, frame in enumerate(frames):
            try:
                img = self._to_pil_image(frame)
                img_path = os.path.join(temp_dir, f"frame_{idx:04d}.jpg")
                img.save(img_path, format="JPEG", quality=90)
                uris.append(Path(img_path).absolute().as_uri())
            except Exception:
                continue

        # 將暫存目錄路徑作為屬性，供清理使用
        self._last_temp_dir = temp_dir  # type: ignore[attr-defined]
        return uris

    def _cleanup_temp_dir(self) -> None:
        temp_dir = getattr(self, "_last_temp_dir", None)
        if isinstance(temp_dir, str) and os.path.isdir(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass

    def _to_pil_image(self, frame: Any) -> Image.Image:
        if isinstance(frame, Image.Image):
            return frame
        if isinstance(frame, np.ndarray):
            array = frame
            if array.dtype != np.uint8:
                # 嘗試將 [0,1] 浮點或其他 dtype 正規化到 uint8
                try:
                    if array.dtype in (np.float32, np.float64):
                        array = np.clip(array, 0.0, 1.0)
                        array = (array * 255.0).astype(np.uint8)
                    else:
                        array = array.astype(np.uint8)
                except Exception:
                    array = array.astype(np.uint8, copy=False)
            # 若是灰階或單通道，轉為 RGB
            if array.ndim == 2:
                array = np.stack([array] * 3, axis=-1)
            if array.ndim == 3 and array.shape[2] == 4:
                # RGBA → RGB
                pil = Image.fromarray(array, mode="RGBA").convert("RGB")
            else:
                pil = Image.fromarray(array)
            return pil.convert("RGB")
        raise TypeError("不支援的影格格式：需要 PIL.Image 或 numpy.ndarray。")

    def _downsample_frames(self, frames: List[Any]) -> (List[Any], float):
        # 優先使用 fps 轉換；否則使用 stride；最後維持原樣
        effective_stride = 1
        effective_fps = 1.0

        if self._source_fps and self._source_fps > 0:
            if self._target_sampling_fps and self._target_sampling_fps > 0:
                # 透過 fps 決定步長
                ratio = max(1.0, self._source_fps / self._target_sampling_fps)
                effective_stride = max(1, int(round(ratio)))
                effective_fps = float(self._source_fps / effective_stride)
            else:
                # 僅已知來源 fps，使用 stride（若已設定）
                effective_stride = max(1, int(self._input_frame_stride))
                effective_fps = float(self._source_fps / effective_stride)
        else:
            # 未知來源 fps，退回使用 stride，下游先以 1.0 告知
            effective_stride = max(1, int(self._input_frame_stride))
            effective_fps = 1.0

        sampled = frames[::effective_stride] if effective_stride > 1 else list(frames)

        if self._max_vlm_frames is not None and len(sampled) > self._max_vlm_frames:
            # 均勻再取樣至上限
            idxs = np.linspace(0, len(sampled) - 1, num=self._max_vlm_frames, dtype=int).tolist()
            sampled = [sampled[i] for i in idxs]

        return sampled, effective_fps

    def analyze(self, frames: List[Any]) -> Dict[str, Any]:
        self._ensure_initialized()

        if not frames:
            return {"summary": "沒有可分析的影格。", "num_frames": 0}

        # 先依設定進行下採樣
        frames_to_use, message_fps = self._downsample_frames(frames)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        # 使用影格清單（file:// URI）傳入
                        "video": self._save_frames_to_temp_images(frames_to_use),
                        "fps": float(message_fps),
                        "max_pixels": 360 * 420,
                    },
                    {"type": "text", "text": self._prompt},
                ],
            }
        ]

        try:
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                fps=float(message_fps),
                padding=True,
                return_tensors="pt",
            )

            if self._device == "cuda":
                inputs = inputs.to("cuda")

            generated_ids = self._model.generate(
                **inputs, max_new_tokens=self._max_new_tokens
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self._processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            summary = output_text[0] if isinstance(output_text, list) and output_text else ""
            return {
                "summary": summary,
                "num_frames": len(frames_to_use),
                "model": self._model_name,
            }
        except Exception as e:
            return {
                "summary": f"推論失敗：{e}",
                "num_frames": len(frames_to_use),
                "model": self._model_name,
                "error": True,
            }
        finally:
            self._cleanup_temp_dir()

    def analyze_keyframes(self, frames: List[Any], prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        多圖（image 列表）推論：使用少量關鍵影格（k1/k2/k3）。
        """
        self._ensure_initialized()

        if not frames:
            return {"summary": "沒有可分析的影格。", "num_frames": 0}

        uris = self._save_frames_to_temp_images(frames)
        # 兩張圖各附上 ID（k1 / k2），不足兩張則依序命名
        contents = []
        for idx, uri in enumerate(uris, start=1):
            if idx == 1:
                contents.append({"type": "image", "image": uri, "id": "k1"})
            elif idx in (2, 3):
                contents.append({"type": "image", "image": uri, "id": "k2"})
            else:
                contents.append({"type": "image", "image": uri, "id": f"k2_2"})
        # 附上提示詞
        contents.append({"type": "text", "text": prompt or self._prompt})

        messages = [
            {
                "role": "user",
                "content": contents,
            }
        ]

        try:
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # 僅處理多張 image，不傳 videos 也不傳 fps
            image_inputs, _ = process_vision_info(messages)
            inputs = self._processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )

            if self._device == "cuda":
                inputs = inputs.to("cuda")

            generated_ids = self._model.generate(
                **inputs, max_new_tokens=self._max_new_tokens
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self._processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            summary = output_text[0] if isinstance(output_text, list) and output_text else ""
            return {
                "summary": summary,
                "num_frames": len(frames),
                "model": self._model_name,
            }
        except Exception as e:
            return {
                "summary": f"推論失敗：{e}",
                "num_frames": len(frames),
                "model": self._model_name,
                "error": True,
            }
        finally:
            self._cleanup_temp_dir()


class GeminiReasoningEngine:
    def __init__(self, model_name: str = "gemini-2.5-flash-lite", max_output_tokens: int = 256) -> None:
        self._initialized = False
        self._model_name = model_name
        self._max_output_tokens = int(max_output_tokens)
        self._model = None
        self._prompt: str = "請描述這段影片中發生的事件與關鍵物件。"

    def set_prompt(self, text_prompt: str) -> None:
        self._prompt = text_prompt

    def initialize(self) -> None:
        if genai is None:
            raise RuntimeError("缺少 google-generativeai，請先安裝該套件。")

        # 使用使用者要求的硬編碼 API Key
        try:
            genai.configure(api_key="AIzaSyADOwZofHKf7REXInTUV7PcjDS5tk9Fdkw")
        except Exception as e:
            raise RuntimeError(f"配置 Gemini API Key 失敗：{e}")

        try:
            self._model = genai.GenerativeModel(model_name=self._model_name)
        except Exception as e:
            raise RuntimeError(f"建立 Gemini 模型失敗：{e}")

        self._initialized = True

    def _ensure_initialized(self) -> None:
        if not self._initialized or self._model is None:
            raise RuntimeError("GeminiReasoningEngine 尚未初始化，請先呼叫 initialize()。")

    def _to_pil_image(self, frame: Any) -> Image.Image:
        if isinstance(frame, Image.Image):
            return frame.convert("RGB")
        if isinstance(frame, np.ndarray):
            array = frame
            if array.dtype != np.uint8:
                try:
                    if array.dtype in (np.float32, np.float64):
                        array = np.clip(array, 0.0, 1.0)
                        array = (array * 255.0).astype(np.uint8)
                    else:
                        array = array.astype(np.uint8)
                except Exception:
                    array = array.astype(np.uint8, copy=False)
            if array.ndim == 2:
                array = np.stack([array] * 3, axis=-1)
            if array.ndim == 3 and array.shape[2] == 4:
                pil = Image.fromarray(array, mode="RGBA").convert("RGB")
            else:
                pil = Image.fromarray(array).convert("RGB")
            return pil
        raise TypeError("不支援的影格格式：需要 PIL.Image 或 numpy.ndarray。")

    def analyze_keyframes(self, frames: List[Any], prompt: Optional[str] = None) -> Dict[str, Any]:
        self._ensure_initialized()

        if not frames:
            return {"summary": "沒有可分析的影格。", "num_frames": 0}

        try:
            # Gemini 支援多段輸入，將提示文字與多張影像一起傳入
            parts: List[Any] = []
            # 先附上提示
            parts.append(prompt or self._prompt)
            # 兩張圖各自加上 ID 說明（文字 + 影像）
            for idx, f in enumerate(frames, start=1):
                try:
                    tag = "k1" if idx == 1 else ("k2" if idx in (2, 3) else f"img_{idx}")
                    parts.append(f"[{tag}] below image")
                    img = self._to_pil_image(f)
                    parts.append(img)
                except Exception:
                    continue

            response = self._model.generate_content(parts, generation_config={
                "max_output_tokens": self._max_output_tokens,
            })
            text = getattr(response, "text", None)
            summary = str(text) if text is not None else ""

            if not summary:
                # 某些版本可能需透過候選取得輸出
                try:
                    candidates = getattr(response, "candidates", None)
                    if candidates and len(candidates) > 0:
                        content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
                        if content and isinstance(content, dict):
                            parts_out = content.get("parts")
                            if isinstance(parts_out, list) and parts_out:
                                summary = str(parts_out[0].get("text", ""))
                except Exception:
                    pass

            return {
                "summary": summary or "",
                "num_frames": len(frames),
                "model": self._model_name,
            }
        except Exception as e:
            return {
                "summary": f"推論失敗：{e}",
                "num_frames": len(frames),
                "model": self._model_name,
                "error": True,
            }
