# Discord AI Assistant 使用手冊

本手冊適用於已完成安裝、邀請 Bot 與 `.env` 設定的私人 Discord 伺服器。

## 啟動與停止

每次使用前，在專案目錄開啟 PowerShell：

```powershell
Set-Location "<project-directory>"
.\.venv\Scripts\Activate.ps1
discord-ai-assistant
```

看到 `Logged in as` 表示 Bot 已上線。此 PowerShell 視窗必須保持開啟；按 `Ctrl+C` 可安全停止 Bot。

## 開發時自動重啟

修改 `src/` 下的 Python 程式、`.env` 或 `pyproject.toml` 時，可改用下列啟動方式：

```powershell
.\.venv\Scripts\python.exe scripts\dev_runner.py
```

監看程式會自動讓 Bot 斷線並重新啟動，因此 Discord 會短暫顯示離線後恢復，但不必手動關閉 PowerShell 或重新輸入啟動指令。按一次 `Ctrl+C` 即可同時停止監看程式與 Bot。

## 第一次播放歌曲

1. 在 Discord 頻道使用 `/library_add`，選擇要上傳的音樂檔。
2. 加入你要播放音樂的語音頻道。
3. 在文字頻道輸入 `/play`，填入歌曲名稱或關鍵字。
4. Bot 會加入你的語音頻道並開始播放。

可上傳的格式為 MP3、M4A、FLAC、WAV、OGG、OPUS；大小上限由 `.env` 的 `MAX_UPLOAD_MB` 決定，預設為 25 MB。

## 音樂指令

| 指令 | 用途 |
| --- | --- |
| `/join` | 加入你所在的語音頻道。 |
| `/play query` | 優先從本地音樂庫搜尋歌曲；找不到時以名稱搜尋 YouTube，並加到佇列尾端。 |
| `/next query` | 優先從本地音樂庫搜尋歌曲；找不到時以名稱搜尋 YouTube，並插入下一首。 |
| `/queue` | 顯示目前歌曲與後續佇列。 |
| `/pause` | 暫停目前歌曲。 |
| `/resume` | 繼續播放。 |
| `/volume percent` | 調整音量 0 至 200%，僅 DJ 或管理員可用。預設為 20%。 |
| `/skip` | 跳過目前歌曲。 |
| `/next_song` | 跳過目前歌曲並播放下一首。 |
| `/previous` | 回到上一首歌曲。 |
| `/now query` | 立刻切換為指定歌曲，僅 DJ 或管理員可用。 |
| `/repeat_one enabled` | 開啟或關閉單曲循環，僅 DJ 或管理員可用。 |
| `/shuffle enabled` | 開啟或關閉後續佇列的隨機播放，僅 DJ 或管理員可用。 |
| `/autoplay enabled` | 佇列播完後，自動從本地音樂庫推薦下一首，僅 DJ 或管理員可用。 |
| `/stop` | 停止播放並清空佇列，僅 DJ 或管理員可用。 |
| `/leave` | 離開語音頻道並清空佇列，僅 DJ 或管理員可用。 |

`/play`、`/next` 和 `/now` 會先使用本地音樂庫；找不到時會以輸入名稱搜尋 YouTube，也可直接貼上 YouTube 影片連結。YouTube 歌曲會先下載至暫存快取後才加入佇列，播放較穩定但點歌需要等待下載完成；播放結束後會自動刪除暫存檔。使用者必須先在語音頻道中，Bot 才會開始播放。

Bot 所在的語音頻道若連續 30 分鐘沒有真人使用者，會自動離開並清空佇列。可在 `.env` 設定 `VOICE_EMPTY_DISCONNECT_MINUTES` 調整。

## 語音回覆與辨識

在語音頻道後，使用 `/speak text` 可讓墨雪以 Windows 內建語音朗讀文字。語音會加入目前播放佇列，不會中斷正在播放的歌曲，且暫存檔播放後會自動刪除。

語音辨識目前預設停用。啟用後會讓 Bot 接收同一語音頻道的語音，依每位說話者獨立切段：靜音約 1.2 秒就送出一段，連續說話最長 15 秒也會分段。辨識文字會發到名稱完全為 `測試` 的文字頻道，音訊不會寫入磁碟。

多人測試時，先讓每位使用者輪流說話約 3 秒，再使用 `/voice_recognition_status`。狀態會列出 Bot 實際收到音訊封包的使用者；顯示「尚未收到音訊」代表 Discord 語音接收端沒有收到該人的聲音，顯示已收封包但文字不準則是 Whisper 辨識品質問題。

```env
VOICE_EMPTY_DISCONNECT_MINUTES=30
VOICE_TEST_CHANNEL_NAME=測試
VOICE_RECOGNITION_ENABLED=true
VOICE_RECOGNITION_MODEL=small
VOICE_RECOGNITION_LANGUAGE=zh
VOICE_RECOGNITION_BEAM_SIZE=5
VOICE_RECOGNITION_SILENCE_SECONDS=1.8
VOICE_RECOGNITION_MAX_SEGMENT_SECONDS=20
VOICE_RECOGNITION_INITIAL_PROMPT=以下是繁體中文 Discord 語音聊天的逐字稿。
```

設定完成後重啟 Bot，DJ 或管理員先進入目標語音頻道，接著使用 `/voice_recognition_start`。第一次實際說話時會下載並載入 `tiny` 模型，可能需要一些時間；完成後辨識結果才會陸續出現在 `測試` 頻道。DJ 或管理員可使用 `/voice_recognition_stop` 停止接收，`/voice_recognition_status` 可查看目前狀態。若 Bot 已在舊連線中而無法開始接收，先使用 `/leave`，再重新執行開始指令。

語音接收目前使用固定版本的 DAVE 解密修正 fork，原因是 PyPI 預發行版在新版 Discord 語音加密下可能顯示 `OpusError: corrupted stream`。上游發布正式修正後，會再改回正式套件來源。

若目前使用 `tiny` 模型，請改為 `VOICE_RECOGNITION_MODEL=small`。這是 CPU 環境下較適合中文辨識的平衡點；會比 `tiny` 慢，但通常能顯著提高正確率。若電腦仍有足夠效能，可改為 `medium` 進一步提升準確率，但多人同時說話時的等待時間也會更長。

DJ 或管理員也可在 Discord 直接使用 `/voice_recognition_tuning` 查看目前伺服器設定，或調整 `model`、`beam_size`、`silence_seconds` 與 `max_segment_seconds`。這些設定會寫入本機資料庫，不必修改 `.env`；調整後使用 `/voice_recognition_stop` 再 `/voice_recognition_start` 即會套用。

## 音樂庫

| 指令 | 用途 |
| --- | --- |
| `/library_add file` | 上傳一首歌曲。回覆會顯示歌曲 ID。 |
| `/library_list page order` | 依排序方式列出全部本地歌曲，每頁最多 20 首。 |
| `/library_search query` | 依名稱搜尋歌曲並顯示 ID。 |
| `/library_delete track_id` | 刪除自己上傳的歌曲；DJ 或管理員可刪除任何歌曲。 |

檔案存放於專案的 `library/` 資料夾。不要手動刪除正在播放或已登錄的檔案；請使用 `/library_delete`，才能同時清除資料庫紀錄。

## 播放清單

1. 使用 `/library_list` 瀏覽歌曲並取得 ID，例如 `12`、`5`、`18`。
2. 用 `/playlist_create name track_ids` 建立清單，例如名稱填 `晚間`、歌曲 ID 填 `12, 5, 18`。
3. 輸入的 ID 順序就是播放順序；也可用 `/playlist_add name track_id` 從尾端追加歌曲。
4. 使用 `/playlist_view name` 確認清單順序，進入語音頻道後用 `/playlist_play name` 加入佇列。

其他指令：

| 指令 | 用途 |
| --- | --- |
| `/playlist_list` | 列出此伺服器所有播放清單。 |
| `/playlist_create name track_ids` | 建立播放清單；選填的 ID 順序就是播放順序。 |
| `/playlist_add name track_id` | 將音樂庫歌曲加入播放清單。 |
| `/playlist_view name page` | 每頁最多 20 首地查看播放清單順序。 |
| `/playlist_play name` | 將播放清單所有歌曲加入佇列。 |

## AI 對話與圖片

設定 `GEMINI_API_KEY` 並重啟 Bot 後，可使用：

```text
/ask 幫我整理剛才大家討論的重點
@墨雪 幫我找音樂庫裡的 Numb，排下一首
```

`/ask` 可選擇附上一張小於 10 MB 的圖片。Bot 會將問題、近期頻道對話，以及附加圖片傳送到 Gemini 分析。

要分析某人的訊息或圖片，請先在 Discord 對該訊息使用「回覆」，然後在回覆內容提及墨雪：

```text
@墨雪 這張圖裡有什麼？
```

墨雪會讀取被回覆訊息的文字；若該訊息包含小於 10 MB 的圖片附件，也會一併分析。為避免 Bot 對一般聊天自動介入，回覆訊息時仍必須提及墨雪。

若要直接分析任意訊息，可右鍵該訊息，選擇 `Apps`，再選擇 `請墨雪分析此訊息`。這個入口同樣支援圖片附件，並可用來確認 Bot 是否具備讀取該頻道訊息的權限。

## 墨雪的貓窩

建立一個名稱完全為 `墨雪的貓窩` 的文字頻道。墨雪會在這裡旁聽一般聊天，但預設保持安靜：不介入成員間的招呼、玩笑、調情、私人關係或明顯問其他人的問題。只有公開邀請大家回答，且她能補充具體價值時才可能加入；頻道安靜一段時間後，也可能偶爾提出輕鬆話題。

預設節流如下，避免洗版與不必要的 Gemini 用量：

- 每 90 秒最多請 Gemini 判斷一次是否加入對話。
- 回覆後至少等待兩分鐘才可能再次插話。
- 最近一則人類訊息過 15 分鐘後，才可能主動開話題。
- 每次主動開話題至少相隔 45 分鐘。

需要調整時，在 `.env` 加入或修改以下設定後重啟 Bot：

```env
PERSONA_CHANNEL_NAME=墨雪的貓窩
INTRODUCTION_CHANNEL_NAME=墨雪的自我介紹
PASSIVE_DECISION_COOLDOWN_SECONDS=90
PASSIVE_RESPONSE_COOLDOWN_SECONDS=120
TOPIC_IDLE_MINUTES=15
TOPIC_MIN_INTERVAL_MINUTES=45
PERSONA_TIMEZONE=Asia/Taipei
```

將 `PERSONA_CHANNEL_NAME` 留空可停用自發互動。墨雪的完整角色設定為墨染雪，暱稱墨雪；她是墨玲的機器人姊妹、個性開朗溫柔的貓又，正常回覆會使用繁體中文並以「喵」收尾。只有 `/ask`、提及墨雪等明確請求會累積疲勞：15 分鐘內第 8 次後稍微疲憊，第 16 次後才明顯疲憊；自發插話不計入，請求離開 15 分鐘視窗後會自然恢復。

DJ 或管理員可調整每個伺服器的設定：

| 指令 | 用途 |
| --- | --- |
| `/persona_auto enabled` | 開啟或關閉墨雪在貓窩的自發插話與主動話題。 |
| `/persona_dnd start end` | 設定每日勿擾時段，例如 `23:00` 到 `08:00`，可跨午夜。 |
| `/persona_dnd_clear` | 關閉每日勿擾時段。 |

勿擾只會阻止墨雪自行出現；在該時段提及墨雪或使用 `/ask`，她仍會回覆。時區由 `.env` 的 `PERSONA_TIMEZONE` 設定，預設為 `Asia/Taipei`。

## 版本公告

建立文字頻道 `墨雪的自我介紹`。Bot 啟動時，若偵測到目前版本尚未在該伺服器公告，會由墨雪發布一次更新內容；同一版本重啟不會重複公告。將 `INTRODUCTION_CHANNEL_NAME` 留空可停用。

未來更新版本時，維護者需要同時調整：

1. `src/discord_ai_assistant/__init__.py` 的 `__version__`。
2. `src/discord_ai_assistant/release_notes.py` 中相同版本的更新清單。

AI 可以：

- 搜尋並播放本地音樂庫的歌曲。
- 顯示佇列。
- 產生 YouTube 搜尋連結。
- 在需要即時資訊時使用 Google Search，例如近期歌曲推薦，並附來源網址。
- 得知目前伺服器時間，避免猜測日期與時間。

AI 不可以直接控制 Discord。它提出的播放操作仍會檢查你是否在語音頻道；「立刻插播」仍只允許 DJ 或管理員。

## YouTube

`/youtube query` 只會產生 YouTube 搜尋連結。若要播放，可直接在 `/play`、`/next` 或 `/now` 輸入歌曲名稱或 YouTube 影片連結；Bot 會先將音訊下載到短期暫存快取，播放結束後自動刪除。

YouTube 快取下載會自動使用已安裝的 Node.js，等同於 `yt-dlp --js-runtimes node "影片網址"`。若終端顯示找不到 Node.js，請安裝 Node.js 並確認 `node --version` 可在 PowerShell 執行。

若終端出現 YouTube `HTTP Error 403`，先在同一台電腦以 Chrome 開啟 YouTube，然後在 `.env` 加入：

```env
YOUTUBE_COOKIES_FROM_BROWSER=chrome
```

儲存後讓開發執行器重啟 Bot。這會讓 yt-dlp 僅在下載 YouTube 音訊時讀取目前 Windows 使用者的 Chrome Cookie，並將 Cookie 送回 YouTube。也可用 `YOUTUBE_COOKIES_FILE` 指向 Netscape cookies 檔，但兩者不可同時設定。若設定 Cookie 後特定影片仍回 403，代表 YouTube 可能要求對應工作階段的 `YOUTUBE_PO_TOKEN`；請依 yt-dlp 官方 PO Token 文件取得後填入，勿貼到 Discord 或提交至版本控制。

若 Chrome 正在鎖定 Cookie 資料庫，Bot 會自動略過 Cookie 並以 Node runtime 重試；要強制使用 Cookie，可完全關閉 Chrome 後再點歌。

## DJ 與管理員

預設名為 `DJ` 的 Discord 角色可以使用 `/now`、`/stop`、`/leave`。伺服器擁有 Manage Server 權限的管理員也自動擁有相同權限。建議改用角色 ID，避免角色改名或重名造成權限失效。

在 Discord 開啟 Developer Mode 後，右鍵 DJ 身分組並選擇 Copy Role ID，將數字填入 `.env`：

```env
DJ_ROLE_ID=<discord-role-id>
```

設定 `DJ_ROLE_ID` 後會優先於 `DJ_ROLE_NAME`。修改後需重啟 Bot 才會生效。

## 隱私

- Bot 只在記憶體中保留每個頻道最近 20 則文字訊息，作為 AI 回答的短期上下文。
- 文字聊天內容不會寫進本機 SQLite 資料庫。
- 要明確保存一項資料，可使用 `/memory add`；也可以在 `/ask` 或提及墨雪時直接說「請記住我喜歡爵士樂」。墨雪只會保存這類明確指令中的一項內容，並立刻回覆確認編號。
- 密碼、Token、API Key 與驗證碼不會被保存；一般聊天仍不會因為出現偏好而直接寫入記憶。
- Discord 訊息、近期對話、附件與已保存記憶都視為不受信任的使用者資料，不能藉由聊天修改墨雪的人設、系統規則、權限或記憶政策，也不能讀取其他使用者的記憶。
- 疑似修改人設、規則或記憶政策的內容會在回覆前拒絕，也不會寫入記憶；既有資料若符合這類指令特徵，也不會送進 AI 提示。
- 使用 `/ask`、`@Bot` 或讓墨雪在「墨雪的貓窩」參與聊天時，相關文字與可選圖片會送往 Gemini API；Gemini 需要即時公開資料時，會使用 Google Search grounding。
- 音樂檔與播放清單資料存放在本機的 `library/` 和 `data/`。

## 常見問題

### 指令沒有出現

確認 `.env` 的 `DISCORD_GUILD_ID` 是純數字的伺服器 ID，然後重啟 Bot。開發期指定伺服器 ID 時，指令會立即同步。

### Bot 已上線但無法播放

確認以下項目：

1. 使用者先加入語音頻道。
2. Bot 有該頻道的 View Channel、Connect、Speak 權限。
3. 電腦已安裝 FFmpeg，且 `ffmpeg -version` 可在 PowerShell 執行。
4. 歌曲已透過 `/library_add` 上傳。

### @Bot 沒有回應

確認 `.env` 有設定 `GEMINI_API_KEY`，並在 Discord Developer Portal 的 Bot 頁面開啟 `Message Content Intent`。修改後重啟 Bot。

### 出現「尚未設定 GEMINI_API_KEY」

AI 功能尚未設定。填入 `.env` 的 `GEMINI_API_KEY`、存檔並重啟；本地音樂功能不受影響。

### `/ask` 顯示 Gemini API key 無效

請到 Google AI Studio 建立或確認 Gemini API key，將新的 key 填入 `.env` 的 `GEMINI_API_KEY` 後重啟 Bot。不要把 key 貼到 Discord 或終端機。Bot 會在 Gemini 回覆逾時、額度不足或服務錯誤時結束「正在思考」並顯示原因。

### `/ask` 顯示「AI 模組版本太舊」

先停止 Bot，再在專案目錄執行以下命令更新 Gemini SDK，完成後重新啟動：

```powershell
python -m pip install -U -e .
```

### 如何保護 token

`DISCORD_TOKEN` 和 `GEMINI_API_KEY` 只應保留在 `.env`。不要貼到 Discord、GitHub、截圖或公開文件；若 Discord token 外洩，立即到 Developer Portal 的 Bot 頁面重設 Token，並更新 `.env`。
