# 墨雪 Slash Command 分類

Discord 的 `/` 指令清單以功能群組整理，避免大量低頻指令全部擠在頂層。

## 保留在頂層的常用指令

```text
/play
/skip
/queue
/pause
/resume
/ask
```

## /meeting — 遊戲會議紀錄

```text
/meeting status
/meeting start [name]
/meeting stop
/meeting transcript [limit]
/meeting devices
/meeting output <channel>

/meeting agents
/meeting pair <code>
/meeting select <agent_id>
/meeting agent
/meeting calibrate [game] [profile_name] [delay_seconds]
/meeting profiles
/meeting profile <profile_id>
/meeting ocr
```

`start/status/devices/output/transcript` 目前仍直接控制 Server 端 `game_meeting_recorder` Tool；`stop` 只有最後整理會議重點時才使用 Gemini。

Capture Agent 類指令支援本機與遠端裝置：

- `agents`：列出已配對 Agent 與等待配對的裝置。
- `pair`：用遠端 Agent 顯示的 6 碼配對碼授權裝置。
- `select`：選擇此 Discord Server 目前要控制的 Agent。
- `agent`：查看目前 Agent 狀態。
- `calibrate`：在目前 Agent 上倒數、擷取凍結遊戲畫面並開啟最上層框選 GUI。
- `profiles/profile`：列出／切換該 Agent 自己的 ROI Profile。
- `ocr`：直接在目前 Agent 上擷取 ROI 並做 OCR 測試。

若已選取的遠端 Agent 在線，以上 Capture 操作會透過 Capture Hub RPC 發送到遠端；否則 fallback 到 Server 本機 Agent。

遠端 Session Audio/Whisper 尚在下一階段，因此目前 `/meeting start|stop|devices|transcript` 仍是 Server 端執行。

## /comms — Discord 語音通訊對話記錄

```text
/comms status
/comms start
/comms stop
/comms tuning [model] [beam_size] [silence_seconds] [max_segment_seconds]
```

這組指令對應原本的 `voice_recognition_*` 功能。

## /voice — 語音回答與朗讀

```text
/voice ask <prompt>
/voice say <text>
/voice read <channel>
/voice stopread
/voice join
/voice leave
```

- `ask`：Gemini 回答，文字回在 Discord，同時使用 TTS 在使用者所在語音頻道朗讀。
- `say`：不經 Gemini，直接 TTS。
- `read`：持續朗讀指定文字頻道之後出現的新訊息，聲音送到使用者所在語音頻道。
- `stopread`：停止持續朗讀。

## /music — 進階播放控制

```text
/music next
/music now
/music previous
/music repeat
/music shuffle
/music autoplay
/music volume
/music stop
/music youtube
```

常用播放操作仍留在 `/play`、`/skip`、`/queue`、`/pause`、`/resume`。

## /library — 本地音樂庫

```text
/library add
/library search
/library list
/library delete
```

## /playlist — 播放清單

```text
/playlist create
/playlist add
/playlist list
/playlist view
/playlist play
```

## /persona — 墨雪人格與主動互動

```text
/persona auto
/persona comfort
/persona dnd
/persona dnd_clear
```

## /memory — 使用者記憶

```text
/memory add
/memory list
/memory delete
/memory clear
/memory passive
```

## 相容性

舊功能的 Python callback 暫時保留在 `AssistantCommands` 內，讓新群組可以重用同一套權限與邏輯；但已搬移的舊頂層 Slash Command 會在 Discord Command Tree 同步前移除，因此不會繼續出現在 `/` 自動完成清單。
