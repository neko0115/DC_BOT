# Game Meeting Recorder

## 目的

把「遊戲中定期的語音狀況講解」與「固定聊天室區域中的玩家打字討論」收進同一個 Session 時間軸，保留來源、時間、辨識信心與原始語音，再交給墨雪整理會議重點。

這個工具刻意不做 DCS 即時戰術事件解析，也不直接讀遊戲記憶體或封包。第一版只依賴玩家本來就看得到／聽得到的畫面與音訊。

## 資料流

```text
Windows 遊戲輸出
  -> WASAPI loopback (SoundCard)
  -> RMS 分段 + pre-roll
  -> WAV 保留
  -> faster-whisper + VAD
  -> voice event

聊天室固定 ROI
  -> Windows screenshot
  -> resize + autocontrast
  -> 畫面變化檢查
  -> RapidOCR
  -> confidence filter + fuzzy dedup
  -> chat event

voice/chat events
  -> data/game-meeting-recorder/sessions/<session_id>/events.jsonl
  -> stop_session
  -> timestamp-sorted transcript
  -> 墨雪摘要
```

## 安裝

核心 Bot 不會強制安裝 OCR / Windows loopback 依賴。需要這個工具時執行：

```powershell
python -m pip install -e ".[meeting]"
```

meeting extra 包含：

- `SoundCard`：Windows WASAPI loopback。
- `rapidocr`：聊天室 OCR。
- `onnxruntime`：RapidOCR 預設 CPU 推理後端。
- `Pillow`：Windows ROI screenshot 與影像前處理。

`faster-whisper` 與 NumPy 已由墨雪核心依賴提供。

### Windows 音訊注意

SoundCard 在 Windows/WASAPI 的單聲道錄音有已知問題，所以工具會以裝置原本的多聲道方式擷取，再於程式內 downmix 成 mono；不要為了省流量把 recorder 強制指定成單聲道。

## 設定檔

版本庫中的：

```text
tools/game_meeting_recorder/config.json
```

只是預設模板，不會被執行期間直接修改。

第一次使用後會建立：

```text
data/game-meeting-recorder/config.json
```

之後 `configure_audio` / `configure_chat` / `configure_output` 都只修改 runtime config，因此不會讓 Git working tree 變髒。

## 權限

會議內容、錄音與輸出設定屬於高權限操作，因此 manifest 使用 `requires_dj` 做核心層權限限制。

- `status`：一般成員可查詢工具是否正在運作。
- `list_audio_devices`
- `configure_audio`
- `configure_chat`
- `configure_output`
- `test_chat_capture`
- `start_session`
- `stop_session`
- `get_transcript`

以上除 `status` 外都要求 `is_dj=true`，也就是 DJ 或管理員。非 DJ 的 Gemini 請求不會拿到這些 action 的 function schema；Tool Client 與 Gateway Registry 執行時也會再檢查一次。

不要移除 `requires_dj` 來處理「模型叫不到工具」之類問題。若需要讓一般成員使用某個 action，必須先確認該 action 不會開始錄音、讀取逐字稿、修改設定或跨頻道發布。

## Actions

### `status`

回傳：

- 是否正在錄製
- Session ID / 名稱
- voice/chat event 數量
- 最近錯誤
- audio/chat 是否啟用
- ROI
- summary channel id
- SoundCard / RapidOCR / Whisper 等依賴是否可 import

### `list_audio_devices`

列出 SoundCard 的 speakers 與 loopback microphones。

建議先呼叫一次，若預設輸出自動匹配錯誤，再把正確 loopback `id` 填進 `configure_audio.loopback_device_id`。

### `configure_audio`

可調：

- `loopback_device_id`
- `sample_rate`（預設 48000）
- `speech_threshold_dbfs`
- `silence_seconds`
- `pre_roll_seconds`
- `max_segment_seconds`
- `whisper_model`
- `whisper_language`
- `whisper_beam_size`
- `whisper_initial_prompt`

目前第一層切段使用 RMS threshold，第二層再由 faster-whisper 的 VAD 過濾。因為目標用途是干擾較少、定期出現的語音播報，先用這個方式降低複雜度。

`pre_roll_seconds` 用來保留偵測到語音以前的一小段音訊，避免吃掉句首。

### `configure_chat`

可調固定 ROI：

- `enabled`
- `x`, `y`, `width`, `height`
- `poll_seconds`
- `change_threshold`
- `ocr_min_confidence`
- `scale`

OCR 不會每幀都跑。工具先對低解析度 grayscale signature 做差異比較，只有聊天室畫面改變超過 `change_threshold` 才跑 RapidOCR。

已經看過的文字會進 fuzzy dedup window，降低同一句停留在畫面數秒造成的重複事件。

### `test_chat_capture`

只截圖一次，不需要先開始 Session。

會保存：

```text
data/game-meeting-recorder/calibration/chat_<time>.png
data/game-meeting-recorder/calibration/chat_<time>_processed.png
```

並回傳 OCR 文字與 confidence。這是設定 ROI 時最重要的校正工具。

### `configure_output`

- `summary_channel_id`：預計讓墨雪發布最終重點的 Discord channel ID。
- `retain_audio`：是否保留每段 WAV。

Recorder 本體**不直接持有 Discord Token / Bot 物件**。`summary_channel_id` 會跟逐字稿一起回傳給墨雪；跨頻道發布由 Discord 整合層處理，避免外接工具越權操作 Discord。

### `start_session`

開始新的 Session。

若 audio/chat 已啟用但缺少 optional dependencies，會拒絕開始並回傳缺少的套件，不會假裝成功。

### `stop_session`

1. 停止音訊與 OCR worker。
2. 等待已排隊的 Whisper 辨識完成。
3. 依 timestamp 重排 events。
4. 更新 `session.json`。
5. 回傳最多 350 個事件與 summary-ready transcript。

Gateway 單次呼叫有 timeout，因此 stop 最多等待 50 秒。若仍有辨識尚未完成，會明確回傳 `incomplete=true` 與 errors，墨雪摘要時必須標示。

### `get_transcript`

取得進行中、上一場或磁碟上最近一場的紀錄。

## Session 儲存格式

```text
data/game-meeting-recorder/
  config.json
  whisper-models/
  calibration/
  sessions/
    20260819_231500/
      session.json
      events.jsonl
      audio/
        segment_0001.wav
        segment_0002.wav
```

事件範例：

```json
{
  "timestamp": "2026-08-19T23:15:32.125+08:00",
  "source": "voice",
  "text": "下一階段先處理北部區域。",
  "confidence": 0.88,
  "audio_file": "data/game-meeting-recorder/sessions/.../segment_0001.wav"
}
```

聊天室：

```json
{
  "timestamp": "2026-08-19T23:16:04.412+08:00",
  "source": "chat",
  "text": "等第二隊回來再打",
  "confidence": 0.94
}
```

## Confidence 規則

- RapidOCR 的 `scores` 直接保存為 OCR confidence。
- Whisper 沒有提供一句話的校準式「正確率」。目前工具由 segment `avg_logprob` 與 `no_speech_prob` 算出 0~1 heuristic confidence，只作為摘要風險提示，不能當成統計意義上的機率。
- 墨雪摘要不得因文字讀起來合理，就把低信心內容自行補成正式決議。

## 已知限制

1. RMS 是「有聲音」而非真正 speech detector，爆炸聲 / BGM 仍可能觸發切段；Whisper VAD 會再過濾，但不保證零誤觸。
2. 遊戲若把語音、音效、BGM 全混在同一輸出，ASR 品質仍取決於人聲相對音量。
3. OCR ROI 必須先校正。畫面解析度、UI scale 或遊戲視窗位置改變時需重新設定。
4. OCR dedup 以文字相似度為主；兩個人短時間內真的送出完全相同訊息時可能被視為重複。
5. 第一版沒有聲紋 / speaker diarization；voice source 目前視為 game/system narration。
6. Recorder 本體不直接發布 Discord 指定頻道；它回傳 target channel id 與 transcript，由墨雪核心完成摘要與跨頻道發布。

## Codex / LLM 維護規則

修改此工具時：

1. 不要把 Discord Token、Gemini API key 或其他 secrets 放進 Tool 目錄。
2. 不要直接 import 墨雪 Database、MusicManager 或 Discord Bot instance。
3. 新增/刪除 action 時同步更新 `manifest.json`、本文件與 `CHANGELOG.md`。
4. 改動 config schema 時同步更新 `config.json` default template 與 validation。
5. 修 ASR/OCR 問題時保留原始資料可追溯性，不要用「辨識失敗就自行猜字」當 fallback。
6. 每次功能、介面、依賴、設定、權限或 bug fix 都必須在 `CHANGELOG.md` 留紀錄。
7. 涉及錄製、逐字稿、裝置資訊、設定或發布的 action 應保持 `requires_dj=true`，除非已完成明確安全評估。
