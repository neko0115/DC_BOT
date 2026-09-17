# Example Echo Tool

## 用途

這是墨雪 Tool Gateway 的最小可執行範例，用來驗證：

- Gateway 是否成功掃描 `tools/`。
- manifest 是否成功轉成 Gemini function schema。
- 墨雪是否能透過統一 Port 呼叫外接工具。
- context 是否只包含允許的欄位。

它不是正式使用者功能，後續新增工具時可直接複製此資料夾作為模板。

## Actions

### `echo`

輸入：

- `text`：非空字串。

輸出：原樣回傳文字，並標記 `source=example_echo`。

### `status`

不需要參數。回傳 Gateway 正常訊息，以及目前呼叫的 `guild_id`、`user_id`、`is_dj`。

## LLM 使用規則

只有使用者明確提到「工具測試」、「外接工具測試」或 `echo tool` 時才應曝光給模型。不要在一般聊天中呼叫此工具。

## 維護規則

修改 action、參數、回傳格式、觸發字或行為時，必須同步更新：

1. `manifest.json`
2. 本 `TOOL.md`
3. `CHANGELOG.md`
