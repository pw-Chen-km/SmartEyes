# Visual Monitoring POC

一個以 YOLO-World 作為感知層、Gemini Pro Vision 作為推理層的互動式影片監控概念驗證專案。GUI 採用 Gradio，支援上傳影片、視覺與文字提示、觸發事件緩衝後送雲分析並回傳說明文字。

## 快速開始

1. 建立虛擬環境並安裝套件
   ```bash
   cd visual_monitoring_poc
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. 設定環境變數
   ```bash
   cp .env .env.local  # 或直接編輯 .env
   # 編輯 GEMINI_API_KEY=your_key
   ```
3. 啟動
   ```bash
   python main.py
   ```

## 架構說明

- `core/orchestrator.py`: 管理各模組資料流與業務邏輯
- `core/perception.py`: 封裝 YOLO-World 輸入/輸出介面
- `core/reasoning.py`: 封裝 Gemini API 呼叫
- `core/trigger.py`: IOU 計算與觸發邏輯
- `core/buffer.py`: 影格採樣與累積
- `utils/video_processing.py`: 影片與影像處理工具
- `app_ui.py`: Gradio 介面
- `main.py`: 程式入口

## 後續開發建議

- 串接 YOLO-World demo 的推論介面至 `PerceptionEngine.detect`
- 於 `ReasoningEngine.analyze` 串接 Gemini Pro Vision 多張影像輸入
- 在 `utils/video_processing.py` 實作 OpenCV 讀取與裁切
- 在 `app_ui.py` 增加 bbox 繪製互動（可用 Gradio Canvas 或自訂前端）
