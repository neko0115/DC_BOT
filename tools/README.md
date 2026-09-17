# 墨雪外接工具規範

`tools/` 是墨雪的外接工具目錄。Tool Gateway 會掃描每個非 `_` 開頭的子資料夾，透過同一個 HTTP Port 提供給墨雪與其他本機程式使用。

## 必要檔案

每個工具資料夾必須包含：

- `manifest.json`：機器可讀的工具名稱、版本、觸發字、權限與 action JSON Schema。
- `tool.py`：預設工具本體；entrypoint 可在 manifest 改名，但必須位於同一資料夾。
- `TOOL.md`：給 Codex、LLM 與維護者看的用途、限制、輸入輸出與維護說明。
- `CHANGELOG.md`：此工具自己的更新紀錄。

缺少任何必要文件、manifest 格式錯誤或 entrypoint 無法載入時，Gateway 會略過該工具並在 `/health` 與 `/v1/tools` 的 `load_errors` 回報原因，不會讓 Discord Bot 整體停止。

## manifest.json

```json
{
  "name": "example_echo",
  "version": "0.1.0",
  "description": "A short description shown to the model.",
  "enabled": true,
  "entrypoint": "tool.py",
  "trigger_keywords": ["工具測試", "echo"],
  "always_available": false,
  "actions": {
    "echo": {
      "description": "Echo text back to the caller.",
      "requires_dj": false,
      "trigger_keywords": ["echo", "回聲測試"],
      "parameters": {
        "type": "object",
        "properties": {
          "text": {"type": "string"}
        },
        "required": ["text"]
      }
    }
  }
}
```

`name` 與 action 名稱只能使用小寫英文字母、數字與底線，且必須以小寫英文字母開頭。`name` 必須和資料夾名稱一致。

工具層 `trigger_keywords` 是相容預設值。每個 action 可另外設定自己的 `trigger_keywords`；只要 action 有設定，就以 action 自己的觸發詞為準。這可避免一句「查狀態」把同一工具的所有 function schema 一次送給 Gemini。觸發比對會忽略空白與標點，並做 Unicode 正規化。

若 `always_available=true`，所有一般 AI 請求都會附帶該工具的可用 action schema，請只對非常通用且低風險的工具使用。

### Action 權限

每個 action 可設定：

```json
"requires_dj": true
```

預設為 `false`。設為 `true` 時：

1. 一般成員的 Gemini request 不會收到該 action 的 function schema。
2. Tool Client 執行前會再次拒絕非 DJ／管理員。
3. Gateway Registry 也會在真正呼叫 entrypoint 前再檢查一次。

因此不要只在 `TOOL.md` 或 prompt 裡寫「僅限管理員」；有權限風險的操作必須使用 `requires_dj`。

## tool.py 介面

entrypoint 必須匯出：

```python
async def invoke(action: str, arguments: dict[str, object], context: dict[str, object]) -> object:
    ...
```

同步 `def invoke(...)` 也可以。回傳值必須可 JSON 序列化。

Gateway 目前提供的 context：

```text
guild_id
user_id
is_dj
voice_channel_id
voice_channel_name
```

工具不得依賴 Discord Bot 內部物件、直接 import 墨雪的 Database/MusicManager，或假設未列在 context 的資料一定存在。需要更多權限時，應先擴充明確的 Gateway contract。

## 摘要後發布到 Discord

外接工具本體不得拿 Discord Token 或 Bot instance。若工具產生的是「需要讓 Gemini 根據工具結果整理後，再發布到指定頻道」的資料，可在結果中同時回傳：

```json
{
  "summary_channel_id": 123456789012345678,
  "summary_instruction": "只根據 transcript/events 整理重點。",
  "transcript": "...",
  "events": []
}
```

ToolRouter 會把這兩個 metadata 轉成核心內部的 `publish_final_reply` effect；effect 不會傳給 Gemini。Gemini 只看到逐字稿與摘要指令，生成最終回覆後，再由墨雪核心檢查目標頻道是否屬於同一個 Guild，並用禁止 mentions 的方式發送。

若 `summary_channel_id` 為 `null`、無效值，或缺少 `summary_instruction`，就不會跨頻道發布，最後回覆留在原本的對話頻道。

## HTTP API

預設：`http://127.0.0.1:8765`

```text
GET  /health
GET  /v1/tools
GET  /v1/tools/{tool_name}
POST /v1/tools/{tool_name}/invoke
POST /v1/reload
```

invoke body：

```json
{
  "action": "echo",
  "arguments": {"text": "hello"},
  "context": {"guild_id": 1, "user_id": 2, "is_dj": false}
}
```

## 更新規則

任何會改變工具行為、action、輸入輸出、依賴、設定、限制、權限或 bug 修正的修改，都必須在同一個變更中更新該工具的 `CHANGELOG.md`。若介面或使用方式改變，也要同步更新 `TOOL.md` 與 `manifest.json`。

建議版本規則：

- Patch：bug fix，不改既有介面。
- Minor：新增相容 action/能力。
- Major：移除或破壞既有介面。

## 新增工具

最簡單的方式是複製 `tools/example_echo/`，重新命名資料夾與 manifest 的 `name`，再實作自己的 actions。Gateway 在下一次工具清單刷新時會重新掃描；預設約 30 秒，不需要修改 Gemini 或 Discord Bot 主程式。

## 安全注意

Gateway 預設只綁 `127.0.0.1`。如果改成 LAN/公開介面，必須設定 `TOOL_GATEWAY_TOKEN`；沒有 token 時 Gateway 會拒絕綁定非 loopback 位址。
