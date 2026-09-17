# Changelog

## 0.4.1 - 2026-08-26

### Improved

- 週會報人工修正改為「提交整篇修正版」：審核者不需要記住每個錯詞或手動逐條列出差異，可直接上傳完整 UTF-8 `.md` / `.txt`，或在內容足夠短時直接貼整篇文字。
- 按下修正按鈕後，墨雪會在同一 Guild／使用者／頻道等待 10 分鐘，下一份符合條件的完整修正版自動綁定目前草稿，不需要人工輸入 review ID。
- 完整人工修正版會另存為 `human_revision_<id>.md`，不覆寫 AI 原草稿；墨雪同時比較原始逐字稿、AI 原草稿與人工完整修正版，再提出可重用的 `asr_alias`、`domain_term`、`report_preference` 候選。
- 短訊息不會被 pending review 誤當成整篇修正版；附件限制為 `.md` / `.txt`、UTF-8、最大 2 MiB，全文最多 50,000 字元。
- 後續重新整理會把整篇人工修正版視為本次最高優先級人工指導，但 Knowledge DB 仍只有在使用者再次確認「套用學習並重整」後才會更新。

### Tests

- 新增 `tests/test_meeting_full_article.py`，驗證全文正規化、短訊息拒絕、原稿／人工修正版 unified diff、10 分鐘 pending expiry，以及所有審核回合均改用全文修正版入口。

## 0.4.0 - 2026-08-25

### Added

- 週會報人工審核新增「修正並重新整理」流程；DJ／管理員可直接貼修正後段落或 `誤辨識 -> 正確詞 | 簡短解釋`，墨雪會先建立 Feedback revision，不會直接發布或覆寫原草稿。
- 新增兩階段學習確認：墨雪先列出可重用的 `ASR alias`、領域專有名詞與報告格式偏好候選，人工可選「套用學習並重整」或「只重整，不學習」。AI 建議永遠只以 pending candidate 保存，未經人工核准不得寫入 Knowledge DB。
- 新增 SQLite `meeting_feedback_*`、`meeting_asr_training_examples`、`meeting_report_preferences`、`meeting_report_examples` 資料；保存人工修正、學習候選、核准週報範例與可追溯 ASR 校正資料。
- 已核准的 ASR alias 會回寫目前 Knowledge Profile，並利用原 `transcript.json` 的 segment 時間碼建立 `audio_sha256 + start/end + 原辨識 + 人工修正` 訓練樣本。
- 已核准週會報會保存為同一 Knowledge Profile 的格式範例；之後生成只可參考結構與寫法，Prompt 明確禁止沿用舊報告事實。
- 新增 `/meeting_training_stats` 與 `/meeting_training_export`，可查看或匯出未來本地 Whisper fine-tune 所需的 JSONL manifest。
- 修訂版會另外保存 `report_revision_*`、`transcript_reviewed_*`，不破壞原始 Whisper transcript，方便比較與回溯。

### Safety / Validation

- 一次性的會議事實、數字、負責人、時間與決議不允許成為學習 candidate kind；模型只能提出 `asr_alias`、`domain_term`、`report_preference`，且仍需人工核准。
- 目前只蒐集／匯出本地 fine-tune 訓練資料，**不會**在 Discord Bot 執行期間自動訓練或替換 Whisper 模型。
- 新增 `tests/test_meeting_feedback.py`，驗證人工 mapping 優先、AI candidate 白名單、Profile 隔離、時間段 ASR training sample 與 manifest export。

## 0.3.0 - 2026-08-25

### Added

- 上傳錄音產生的週會報改為「草稿 → 人工審核 → 核准發布」流程；只有 DJ／管理員按下「核准並發布」後，墨雪才會把週會報送到 `/meeting output` 已設定的 `summary_channel_id`。
- 草稿新增「退回／不發布」按鈕；審核狀態、核准者、時間與實際發布頻道會寫入工作目錄的 `review_*.json`，方便稽核。
- 週會報固定加入週會報日期、產生時間、錄音長度與使用中的 Knowledge Profile；日期時間由 Bot 設定時區產生，不由 LLM 猜測。
- 新增 guild-scoped SQLite App Knowledge Database，可建立不同遊戲、App 或情境的 Knowledge Profile，並為每個 Profile 儲存標準專有名詞、簡短解釋與別名／常見 ASR 誤辨識。
- 新增 `/knowledge profiles`、`/knowledge profile_set`、`/knowledge use`、`/knowledge term_set`、`/knowledge terms`、`/knowledge term_remove` 管理介面。
- Whisper 只取得所選 Knowledge Profile 的標準詞與 aliases 做辨識 bias；Gemini 則額外取得簡短解釋協助理解詞義，並明確禁止把詞庫內容當成本次會議事實。
- 內建《明日之後》Profile 作為初始 seed；seed 只補缺漏，不覆蓋使用者後續人工修改的解釋、aliases 或 Profile 設定。

### Changed

- 詞庫從單一《明日之後》靜態校正表擴充為可重用的 App Knowledge Profile 架構；既有靜態 glossary 保留為無資料庫時的 fallback。
- Knowledge Profile 由會議標題／hint 自動選擇，無法判斷時使用該 Discord Guild 設定的預設 Profile。
- 逐字稿 cache fingerprint 納入會影響 Whisper 的 Profile、標準詞與 aliases；純語意解釋修改不必強迫重新跑 ASR。
- 發布時會重新讀取目前的 `/meeting output` 設定，而不是沿用產生草稿時的舊頻道值。

### Tests

- 新增 `tests/test_app_knowledge.py`，驗證 Guild 隔離、Profile 選擇、LifeAfter seed、專有名詞解釋與使用者資料優先。
- 新增 `tests/test_meeting_knowledge.py`，驗證 Whisper prompt bias、aliases 精確校正、歧義詞保留與 Discord 長文分段。

## 0.2.1 - 2026-08-25

### Improved

- 新增《明日之後》週會專有名詞詞庫；標題可辨識為《明日之後》時，會將已確認詞彙加入 faster-whisper initial prompt，降低中文／英文混講與遊戲術語誤辨識。
- 新增保守的轉錄後校正，只處理人工確認過的誤辨識（例如海域 3、浴血、肉魚、屑骨人、晶蝶無人機、黑馬、新興、迷霧的 boss、三王）；仍有歧義的詞不會自動改寫。
- 轉錄工作目錄新增 `glossary.json`，保存詞庫版本、套用 profile 與實際替換紀錄，方便回查自動校正。
- 訊息右鍵整理錄音時，若訊息本身有文字，優先用訊息文字作為會議名稱，讓詞庫能從「明日之後週會」等上下文自動啟用。
- 詞庫版本納入逐字稿 cache fingerprint，詞庫更新後不會錯誤重用舊的未校正逐字稿。

### Tests

- 將 `tests/test_meeting_upload.py` 從 pytest-style 裸函式改為 `unittest.TestCase`；原 CI 使用 `python -m unittest discover`，先前這批測試實際未被執行。
- 新增 `tests/test_meeting_glossary.py`，驗證詞庫選擇、Whisper prompt bias、人工確認替換、歧義詞保留與 `glossary.json` audit。

## 0.2.0 - 2026-08-25

### Added

- 新增 Discord `/meeting_report`：可直接上傳 MP3、WAV、M4A、FLAC、OGG、OPUS、WEBM 或 MP4 會議錄音，由 Bot 主機上的 `faster-whisper` 本地轉錄後交給墨雪整理週會報。
- 新增訊息右鍵選單「墨雪：整理錄音成週會報」，可直接對已上傳的錄音附件執行同一流程。
- 匯入音檔、逐字稿 JSON/TXT 與 Markdown 週會報保存在 `data/game-meeting-recorder/imports/<sha256>/`；同一音檔且 Whisper 設定未變時會重用逐字稿快取，避免重跑 ASR。
- 逐字稿保留 Whisper segment 起迄時間與 heuristic confidence；低信心片段會明確標記，方便人工回查原錄音。
- 週會報提示詞新增防誤報規則：不得自行補負責人、日期、數字或決議；必須區分已完成／已測試、進行中、提案／討論、決議與尚未驗證。
- 音訊不會送往 Gemini；Gemini 僅接收本地轉錄後的文字逐字稿。
- 新增 `tests/test_meeting_upload.py`，涵蓋支援格式／大小限制、時間碼、低信心標記、防臆測提示詞與本地 Whisper segment 保留。

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
- 新增 standalone `MoxueCapture.exe` build boundary，PyInstaller 模式將裝置 ID、ROI Profile 與配對資料永久保存在 `%LOCALAPPDATA%\\MoxueCapture`。
- 新增 `Moxue Capture Hub` outbound WebSocket 架構：遠端 Agent 主動連回 Server，使用 6 碼配對碼與持久 token 授權。
- 新增 `/meeting agents`、`/meeting pair`、`/meeting select`；`agent/calibrate/profiles/profile/ocr` 在已選遠端 Agent 在線時透過 Hub RPC 執行，否則 fallback 本機 Agent。
- 遠端 Hub 目前只建議使用可信任 LAN / VPN；公開 Internet transport 等待 TLS/WSS 階段。