# Changelog

## 0.1.0 - 2026-08-19

### Added

- 建立 `game_meeting_recorder` 外接工具。
- 新增 Windows loopback 音訊裝置列舉與設定。
- 新增 RMS 語音切段、pre-roll、靜音結束與最大片段時間。
- 新增 WAV 原始片段保留與 faster-whisper 背景辨識。
- 新增固定聊天室 ROI screenshot、畫面差異檢查、RapidOCR 與 fuzzy dedup。
- 新增 `status`、`list_audio_devices`、`configure_audio`、`configure_chat`、`configure_output`、`test_chat_capture`、`start_session`、`stop_session`、`get_transcript` actions。
- 新增 JSONL 事件時間軸與 Session metadata。
- 新增 voice/OCR confidence、辨識錯誤紀錄與 `incomplete` 狀態。
- 新增 summary-ready transcript、`summary_instruction` 與 `summary_channel_id` 輸出。
- 新增核心 `publish_final_reply` post-processing：Gemini 完成摘要後，由墨雪核心驗證同 Guild 目標頻道並發布，外接工具不取得 Discord Token / Bot instance。
- 跨頻道發布停用 Discord mentions，長內容自動分割成 Discord-safe chunks。
- 新增 `meeting` optional dependencies 與 Windows dependency smoke test。
- Runtime config 移至 `data/game-meeting-recorder/config.json`，避免修改 Git 追蹤中的預設模板。
- 擴充 Tool discovery 關鍵字，涵蓋會議摘要、摘要頻道、音訊裝置、loopback 與 OCR 校正等設定操作。
- 新增 per-action `requires_dj` 權限；除 `status` 外，會議錄製、設定、校正、逐字稿與音訊裝置 actions 皆限 DJ／管理員。
- 非 DJ 的 Gemini request 會隱藏受限 action，Tool Client 與 Gateway Registry 執行時也會再次拒絕未授權呼叫。
- Tool trigger matching 新增 Unicode/CJK 字形、空白與標點正規化，並補上自然語句觸發詞，避免 mention 模式誤判成一般聊天。
- 新增 action-specific `trigger_keywords`；狀態、音訊裝置、OCR、開始/停止 Session 等意圖只曝光相關 function schema，避免單一會議關鍵字一次送出全部 actions。
- 新增 Discord 原生 `/meeting` Slash Command Group；控制操作直接呼叫 Tool Gateway，不依賴 Gemini function selection，只有停止後的內容摘要需要 Gemini。
- 新增獨立 `Moxue Capture Agent`、DXcam 優先螢幕擷取、凍結畫面框選 GUI、normalized ROI 與 per-device Profile。
- 新增 standalone `MoxueCapture.exe` build boundary，PyInstaller 模式將裝置 ID、ROI Profile 與配對資料永久保存在 `%LOCALAPPDATA%\MoxueCapture`。
- 新增 `Moxue Capture Hub` outbound WebSocket 架構：遠端 Agent 主動連回 Server，使用 6 碼配對碼與持久 token 授權。
- 新增 `/meeting agents`、`/meeting pair`、`/meeting select`；`agent/calibrate/profiles/profile/ocr` 在已選遠端 Agent 在線時透過 Hub RPC 執行，否則 fallback 本機 Agent。
- 遠端 Hub 目前只建議使用可信任 LAN / VPN；公開 Internet transport 等待 TLS/WSS 階段。
