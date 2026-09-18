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

## /memory — Memory V2

### 個人記憶

```text
/memory add
/memory list
/memory delete
/memory clear
/memory passive
```

- `add/list/delete/clear`：管理自己的個人記憶。
- `passive`：開啟或關閉自己的被動 Memory V2 整理。這個設定只控制被動學習；明確要求「記住」的手動路徑仍獨立存在。

### 個人 Memory V2 檢查

```text
/memory search <query>
/memory projects
/memory project <name>
/memory provenance <memory_id>
/memory conflicts
```

- `search`：用目前 Memory V2 的 query-aware ranking 模擬召回結果，顯示 score、kind、project/subject 與內容。
- `projects`：列出自己的 active project scopes。
- `project`：查看指定專案的目前狀態、blocker、next step 與近期歷史摘要。
- `provenance`：查看自己的被動記憶由哪些 Discord message/channel/batch/session 產生；不會顯示另一位使用者的私人來源。
- `conflicts`：查看目前保留、尚未自動覆寫的潛在 project fact 衝突。

以上個人檢查指令皆為 ephemeral，且維持 guild + user scope。

### 頻道 Memory V2 模式（DJ／管理員）

```text
/memory channel <mode>
```

可用 mode：

```text
auto
mixed
social
game
game:<subdomain>
project
off
```

- `auto`：依頻道名稱／分類與訊息明確證據判斷。
- `mixed`：允許同一頻道同時維持多個聊天主題，不固定單一 domain。
- `social`：偏向一般社交／日常內容。
- `game`：遊戲頻道，但不固定某一款遊戲。
- `game:<subdomain>`：提供特定遊戲 prior；訊息中的強明確證據仍可覆蓋 prior。
- `project`：專案型被動記憶；不建立 guild-shared 社交 episode。
- `off`：停用目前頻道的被動個人／shared 記憶形成。

目前已知 game subdomain 由 Memory V2 domain registry 提供，例如 `genshin`、`star_rail`、`zzz`、`lifeafter`、`counter_strike`、`arena_of_valor`、`minecraft` 等。輸入未知 subdomain 時指令會回覆可用清單。

### Guild-shared memory

```text
/memory shared_search <query>
/memory shared_forget <memory_id>
```

- `shared_search`：搜尋目前伺服器的 active 公開 shared memory，例如群體事件、遊戲 episode 或群內梗。
- `shared_forget`：DJ／管理員刪除目前 guild 的一項 shared memory；不能跨 guild 刪除。

Shared memory 與個人 `user_memories` 分開儲存。第三方八卦、敏感個資與不符合公開分享政策的內容不應被自動提升為 shared memory。

## 相容性

已搬移的舊頂層 Slash Command 會在 Discord Command Tree 同步前移除，因此不會繼續出現在 `/` 自動完成清單。

Memory V2 social rebalance 已退休 `AssistantCommands` 內舊的被動記憶 queue/flush；production 被動學習由單一 `MemoryV2PassiveRuntime` 負責。這不影響 `/memory add` 或聊天中明確要求墨雪記住內容的手動記憶路徑。
