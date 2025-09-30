from typing import Any, Dict, List, Optional

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
except Exception as import_error:
    Qwen2_5_VLForConditionalGeneration = None  # type: ignore
    AutoProcessor = None  # type: ignore
    process_vision_info = None  # type: ignore
    _IMPORT_ERROR = import_error
else:
    _IMPORT_ERROR = None


class VLMRunner:
    """
    Synchronous VLM inference wrapper using Qwen only. Self-contained, no project-internal imports.

    - model_name: default "Qwen/Qwen2.5-VL-7B-Instruct"
    - prompt: if not specified, uses the same decision prompt as in SDK
    - Input images can be numpy.ndarray (BGR/RGB/GRAY/RGBA) or PIL.Image.Image
    - Returns dict with at least: summary (raw text), decision ("yes"/"no"/"unsure"/empty), num_frames, model
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_new_tokens: int = 128,
        default_prompt: Optional[str] = None,
    ) -> None:
        self._initialized = False
        self._device: Optional[str] = None
        self._model_name = model_name or "Qwen/Qwen2.5-VL-7B-Instruct"
        self._max_new_tokens = int(max_new_tokens)
        self._model = None
        self._processor = None
        self._default_prompt = (
            default_prompt
            or (
                "k1: hand touches ROI. "
                "k2: hand leaves ROI ( consecutive frames after leaving). "
                " check if the hand reaching into ROI and takes a new item from ROI (Observed from k2 frames). "
                "If yes, answer YES. "
                "If not, answer NO. "
                "If unsure, answer UNSURE. "
                "Output one word only."
            )
        )

        self._initialize()

    # --- public API ---
    def set_prompt(self, prompt: str) -> None:
        self._default_prompt = prompt or self._default_prompt

    def infer(self, k1: Any, k2_list: List[Any], prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform VLM inference with k1 (single) and k2 (multiple) images.

        - k1: single image (np.ndarray or PIL.Image)
        - k2_list: multiple images (list)
        - prompt: override default prompt (optional)
        """
        frames: List[Any] = []
        if k1 is not None:
            frames.append(k1)
        if isinstance(k2_list, list) and len(k2_list) > 0:
            frames.extend(k2_list)
        out = self.forward(frames, prompt=(prompt or self._default_prompt))
        if isinstance(out, dict):
            # error path or already-structured
            summary = str(out.get("summary", ""))
            result: Dict[str, Any] = {
                "summary": summary,
                "num_frames": out.get("num_frames", len(frames)),
                "model": out.get("model", self._model_name),
            }
        else:
            summary = str(out or "")
            result = {
                "summary": summary,
                "num_frames": len(frames),
                "model": self._model_name,
            }
        result["decision"] = self.parse_decision(summary)
        return result

    def precheck(self, k1: Any, k2: Any, prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Lightweight precheck using two keyframes (K1 trigger, K2 release).

        Returns a dict containing:
        - k1_summary, k1_decision
        - k2_summary, k2_decision
        - overall ("yes" or "no")
        - passed (bool)
        """
        frames: List[Any] = []
        if k1 is not None:
            frames.append(k1)
        if k2 is not None:
            frames.append(k2)
        if len(frames) < 2:
            return {
                "k1_summary": "",
                "k1_decision": "",
                "k2_summary": "",
                "k2_decision": "",
                "overall": "no",
                "passed": False,
            }

        pre_prompt = (
                "任務：判斷這兩張圖像（K1=觸發瞬間，K2=解除瞬間）是否顯示「此人明確與ROI櫃子互動」（例如開門、拿取、放置）。"
                "條件："
                "- 必須看到手明確接觸或操作ROI，且K1→K2能解釋為一次連續行為。"
                "- 只要沒有手伸向cabinet的都算no。"
                "只允許輸出一個詞："
                "yes 或 no"
            )

        out = self.forward(frames, prompt=pre_prompt)
        summary = str(out.get("summary", "")) if isinstance(out, dict) else str(out or "")
        overall = self.parse_decision(summary)
        # Precheck expects strict yes/no; treat non-yes as no
        overall = "yes" if overall == "yes" else "no"
        return {
            "k1_summary": summary,
            "k1_decision": overall,
            "k2_summary": summary,
            "k2_decision": overall,
            "overall": overall,
            "passed": bool(overall == "yes"),
        }

    def precheck_infer(self, k1: Any, k2_list: List[Any], prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Combined helper: same input/output as infer(), but runs precheck first.

        - Uses the first image in k2_list as the K2 frame for precheck
        - If precheck passes, returns infer(k1, k2_list, prompt)
        - If precheck fails, returns an infer-shaped dict with empty decision
        """
        k2 = k2_list[0] if isinstance(k2_list, list) and len(k2_list) > 0 else None
        pr = self.precheck(k1, k2)
        if bool(pr.get("passed")):
            return self.infer(k1, k2_list, prompt=prompt)

        frames_count = 1 + (len(k2_list) if isinstance(k2_list, list) else 0)
        return {
            "summary": f"precheck failed: {pr.get('overall', 'no')}",
            "num_frames": frames_count,
            "model": self._model_name,
            "decision": "",
        }


    @staticmethod
    def parse_decision(text: str) -> str:
        try:
            s = (text or "").strip().lower()
            token = s.split()[0] if s else ""
            if token in ("yes", "y", "true"):
                return "yes"
            if token in ("no", "n", "false"):
                return "no"
            if token in ("unsure", "unknown", "maybe"):
                return "unsure"
        except Exception:
            pass
        return ""

    # --- internals ---
    def _initialize(self) -> None:
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                f"Missing required packages: {_IMPORT_ERROR}. Please install transformers, accelerate and qwen-vl-utils."
            )
        has_cuda = torch.cuda.is_available()
        self._device = "cuda" if has_cuda else "cpu"

        # load model and processor
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self._model_name,
            torch_dtype="auto",
            device_map="auto" if has_cuda else None,
        )
        self._processor = AutoProcessor.from_pretrained(
            self._model_name,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )
        self._initialized = True

    def _ensure_initialized(self) -> None:
        if not self._initialized or self._model is None or self._processor is None:
            raise RuntimeError("VLMRunner not initialized.")

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
        raise TypeError("Unsupported frame type: require PIL.Image or numpy.ndarray.")

    def _save_frames_to_temp_images(self, frames: List[Any]) -> List[str]:
        temp_dir = tempfile.mkdtemp(prefix="qwen_vl_frames_")
        uris: List[str] = []
        for idx, frame in enumerate(frames):
            try:
                img = self._to_pil_image(frame)
                img_path = os.path.join(temp_dir, f"frame_{idx:04d}.jpg")
                img.save(img_path, format="JPEG", quality=90)
                uris.append(os.path.abspath(img_path))
            except Exception:
                continue
        self._last_temp_dir = temp_dir  # type: ignore[attr-defined]
        return uris

    def _cleanup_temp_dir(self) -> None:
        temp_dir = getattr(self, "_last_temp_dir", None)
        if isinstance(temp_dir, str) and os.path.isdir(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass

    def forward(self, frames: List[Any], prompt: Optional[str] = None) -> Dict[str, Any]:
        self._ensure_initialized()
        if not frames:
            return {"summary": "", "num_frames": 0, "model": self._model_name}

        uris = self._save_frames_to_temp_images(frames)
        contents = []
        for idx, uri in enumerate(uris, start=1):
            if idx == 1:
                contents.append({"type": "image", "image": uri, "id": "k1"})
            elif idx in (2, 3):
                contents.append({"type": "image", "image": uri, "id": "k2"})
            else:
                contents.append({"type": "image", "image": uri, "id": f"k2_2"})
        contents.append({"type": "text", "text": (prompt or self._default_prompt)})

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
                "summary": f"inference failed: {e}",
                "num_frames": len(frames),
                "model": self._model_name,
                "error": True,
            }
        finally:
            self._cleanup_temp_dir()


