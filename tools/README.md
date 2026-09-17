# 墨雪外接工具規範

`tools/` 是墨雪的外接工具目錄。Tool Gateway 會掃描每個非 `_` 開頭的子資料夾，透過同一個 HTTP Port 提供給墨雪與其他本機程式使用。

## 必要檔案

每個工具資料夾必須包含：

- `manifest.json`：機器可讀的工具名稱、版本、觸發字與 action JSON Schema。
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

`trigger_keywords` 決定 Gemini 何時看得到這個工具。若 `always_available=true`，所有一般 AI 請求都會附帶該工具 schema，請只對非常通用且低風險的工具使用。

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

任何會改變工具行為、action、輸入輸出、依賴、設定、限制或 bug 修正的修改，都必須在同一個變更中更新該工具的 `CHANGELOG.md`。若介面或使用方式改變，也要同步更新 `TOOL.md` 與 `manifest.json`。

建議版本規則：

- Patch：bug fix，不改既有介面。
- Minor：新增相容 action/能力。
- Major：移除或破壞既有介面。

## 新增工具

最簡單的方式是複製 `tools/example_echo/`，重新命名資料夾與 manifest 的 `name`，再實作自己的 actions。Gateway 在下一次工具清單刷新時會重新掃描；預設約 30 秒，不需要修改 Gemini 或 Discord Bot 主程式。

## 安全注意

Gateway 預設只綁 `127.0.0.1`。如果改成 LAN/公開介面，必須設定 `TOOL_GATEWAY_TOKEN`；沒有 token 時 Gateway 會拒絕綁定非 loopback 位址。
