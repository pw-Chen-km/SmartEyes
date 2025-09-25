from typing import List, Any


class FrameBuffer:
    def __init__(self, sample_rate: int = 2, max_frames: int = 12) -> None:
        self.sample_rate = max(1, sample_rate)
        self.max_frames = max(1, max_frames)
        self._frames: List[Any] = []
        self._counter = 0

    def reset(self) -> None:
        self._frames.clear()
        self._counter = 0

    def add_frame(self, frame: Any) -> None:
        self._frames.append(frame)
        if len(self._frames) > self.max_frames:
            self._frames.pop(0)

    def maybe_sample(self, frame: Any) -> None:
        self._counter += 1
        if self._counter % self.sample_rate == 0:
            self.add_frame(frame)

    def is_ready(self) -> bool:
        return len(self._frames) >= self.max_frames

    def flush(self) -> List[Any]:
        frames = list(self._frames)
        self.reset()
        return frames


class ReasoningKeyframeBuffer:
    def __init__(self) -> None:
        # 僅收集關鍵幀：k1(接觸)、k2(離開)、k3(離開後)
        self._k1: List[Any] = []
        self._k2: List[Any] = []
        self._k3: List[Any] = []

    def reset(self) -> None:
        self._k1.clear()
        self._k2.clear()
        self._k3.clear()

    def add(self, tag: str, frame: Any) -> None:
        if tag == "k1":
            self._k1.append(frame)
        elif tag == "k2":
            self._k2.append(frame)
        elif tag == "k3":
            self._k3.append(frame)

    def replace_last(self, tag: str, frame: Any) -> None:
        """替換指定標籤的最後一個幀"""
        if tag == "k1" and self._k1:
            self._k1[-1] = frame
        elif tag == "k2" and self._k2:
            self._k2[-1] = frame
        elif tag == "k3" and self._k3:
            self._k3[-1] = frame

    def collect(self, max_k1: int = 1, max_k2: int = 1, max_k3: int = 2) -> List[Any]:
        """
        回傳關鍵幀序列：k1(最多 max_k1)、k2(最多 max_k2)、k3(最多 max_k3)。
        為了相容舊邏輯，預設仍是各取 1 張 k1/k2 與最多 2 張 k3。
        """
        frames: List[Any] = []
        if self._k1:
            frames.extend(self._k1[:max_k1])
        if self._k2:
            frames.extend(self._k2[:max_k2])
        if self._k3:
            frames.extend(self._k3[:max_k3])
        return frames
