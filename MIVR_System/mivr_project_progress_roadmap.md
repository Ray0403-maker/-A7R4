# MIVR-CEIQ 專案進度與未來規劃說明書 (Handover Document for Claude)

本文件整理了 **MIVR-CEIQ 系統 MVP Demo (多人沉浸式VR職涯探索系統)** 的目前開發進度、現行專案架構以及接下來的實作規劃，以利 Claude 接續進行開發與調試。

---

## 📌 專案概述 (Project Overview)
本專案旨在驗證使用 **Sony A7R IV 相機 + ArUco 標記定位**之高精度追蹤，整合 **ESP32-S3 IMU 數據與 EKF 濾波器**實現低延遲定位，並串接 **Google AI (Gemini / STT / TTS) 與 Firebase** 提供沉浸式 VR 教師廣播與 AI 生成場景雙模式的 Holland 職涯行為量化探索。

*   **開發環境**: Windows (PowerShell)
*   **Python 版本**: 3.14.5
*   **關鍵套件**: `opencv-python`, `opencv-contrib-python`, `scipy`, `numpy`, `filterpy`, `google-generativeai`, `firebase-admin`, `websockets`, `psycopg2-binary` (套件均已透過 `--user` 成功安裝完成)

---

## 📂 目前專案目錄結構 (Current Workspace Structure)
專案根目錄為 `d:/MIVR_System/`，目前已建立的目錄結構與核心檔案如下：
```text
d:/MIVR_System/
├── .env                              # 環境變數設定檔 (已建立，待填入金鑰)
├── test cam.py                       # 原廠/基礎相機 Live View 測試腳本
├── config/                           # 設定檔與產生之圖檔目錄
│   ├── camera_calibration.json       # 相機校正檔案 (待生成)
│   ├── helmet_marker_00~09.png       # 產生的 ArUco 頭盔 Marker (DICT_4X4_50)
│   └── floor_anchor_10~17.png        # 產生的地板錨點 Marker
├── data/                             # 數據收集與分析結果
│   ├── trajectories/                 # 軌跡數據
│   ├── sessions/                     # 學習階段歷程紀錄
│   └── static_accuracy_test.csv      # 靜態精度測試數據 (待由 validator 寫入)
├── logs/                             # 系統運行日誌
└── src/                              # 原始碼主目錄
    ├── positioning/                  # 空間定位與傳感器融合
    │   └── aruco_validator.py        # [NEW] ArUco 即時識別與抖動精度檢測程式
    ├── vr_engine/                    # VR 通訊與教學引擎 (待開發)
    ├── ai_agents/                    # Google AI 與多智慧體協調 (待開發)
    └── data_pipeline/                # 數據管道與 Holland 行為分析 (待開發)
```

---

## 📈 目前開發進度表 (Current Progress Checklist)

### Phase 1: 基礎環境與 AI SDK 依賴安裝 `[100% COMPLETED]`
*   [x] 檢查 Python 版本及現有 OpenCV 安裝。
*   [x] 安裝定位核心套件及 Google AI SDK、Firebase、websockets。
*   [x] 建立專案目錄結構。
*   [x] 產生預設 [.env](file:///d:/MIVR_System/.env) 環境變數範本。

### Phase 2: 相機校正與 ArUco 識別驗證 `[50% IN-PROGRESS]`
*   [x] 產生 ArUco Marker 貼紙（頭盔 0-9，地板 10-17）至 `config/` 目錄下。
*   [x] 實作並部署 [aruco_validator.py](file:///d:/MIVR_System/src/positioning/aruco_validator.py) 精度檢測程式。
*   [ ] 拍攝 9x6 棋盤格圖片至少 5 張（建議 10 張）放至 `config/calibration/`，並執行相機校正。
*   [ ] 執行 [aruco_validator.py](file:///d:/MIVR_System/src/positioning/aruco_validator.py) 並錄製/儲存 `data/static_accuracy_test.csv`。
*   [ ] 執行靜態定位精度分析（計算平均誤差是否 < 5cm）。

---

## 🔮 未來規劃與待開發任務 (Future Roadmap & Tasks)

### Phase 3: IMU 數據接收與 EKF 融合 (定位核心升級)
1.  **實作 ESP32-S3 串口數據讀取**：
    *   撰寫腳本讀取 IMU 6軸加速度與角速度數據 (波特率 115200)。
2.  **實作 EKF (Extended Kalman Filter) 融合定位算法**：
    *   於 `src/positioning/vio_fusion.py` 實作視覺 (ArUco 位姿) 與慣性 (IMU) 數據卡爾曼濾波融合。
3.  **多目標追蹤與路徑交叉測試**：
    *   撰寫 `multi_person_tracker.py` 支援 2 人以上交叉移動時不發生 ID 交換與追蹤丟失。

### Phase 4: Google AI 語音與雲端整合
1.  **意圖分類器**：串接 Gemini 2.0 Flash，將學生的語音意圖分類為 concept_request, career_link 等。
2.  **語音互動模組 (STT / TTS)**：
    *   實作 Speech-to-Text 接收學生提問。
    *   實作 Text-to-Speech 合成 AI 導師聲音 (`cmn-TW-Wavenet-A`)。
3.  **Firebase Realtime Database 串接**：將實時定位座標、對話歷史與系統狀態同步上雲。

### Phase 5: VR 雙模式教學引擎
1.  **WebSocket 伺服器**：建立 `ws_server.py`，提供高頻率 (60Hz+) 定位數據廣播給 Unity 頭盔。
2.  **模式一：教師主導廣播控制器**：透過 JSON 設定檔同步所有頭盔至同一 VR 實驗場景。
3.  **模式二：AI 生成式場景引擎**：Gemini 根據學生意圖動態修改/推薦 VR 虛擬實境物件。
4.  **多智慧體協調器 (Multi-Agent Orchestrator)**：協調 AI 導師、Holland 分析師與場景引擎。

### Phase 6: Holland 行為量化與職涯報告
1.  **行為量化模型 (holland_analyzer.py)**：
    *   根據學生在 VR 中對不同職業物件（如馬達、電路板、藝術模型）的停留時間、注視次數與互動對話，量化 Holland 六維度分數 (R, I, A, S, E, C)。
2.  **AI 職涯探索報告生成**：
    *   Gemini 讀取 Holland 分數，自動生成繁體中文個人化探索建議報告與推薦證照路徑。

### Phase 7: Demo 驗收與影片標記
1.  **軌跡視覺化 (visualize_trajectories.py)**：將動態定位數據繪製並疊加至示範影片上。
2.  **自動化生成驗收 JSON 報告** (`demo_validation_report.json`)。

---

## 💡 給 Claude 的接手引導提示 (Context Prompts for Claude)
當您（Claude）接手本專案時，建議遵循以下步驟：
1.  請使用者提供棋盤格圖片，並將其放入 `config/calibration/` 下，完成 Phase 2 的校正動作。
2.  請先在主控台執行：
    ```bash
    python src/positioning/aruco_validator.py --camera 0
    ```
    確認相機 live 畫面能正常偵測出 ArUco Marker（會顯示 ID、距離與 Jitter）。
3.  接下來著手開發 **Phase 3** 中的 EKF 融合代碼（`src/positioning/vio_fusion.py`）與 **Phase 5** 的 WebSocket 伺服器。
