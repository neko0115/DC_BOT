# Web Research Tool

提供墨雪一個受限的 Tavily 公開網路研究工具。它不取代 Gemini 原生 Google Search；只有使用者明確要求深入研究、多來源交叉查證，或指定要讀某個網頁時才應暴露給模型。

## Actions

### `web_research`

輸入 `query`，可選 `depth=basic|deep` 與 `freshness=day|week|month|year`。

- `basic`：一次 Tavily basic Search，最多使用 `WEB_RESEARCH_MAX_RESULTS` 個來源。
- `deep`：一次 advanced Search，再針對最多前三個公開來源做 basic Extract。
- 工具回傳受限長度的 evidence 與來源 URL；最終答案仍由 Gemini 根據 evidence 整理。

### `read_webpage`

讀取單一公開 `http`/`https` URL。可傳 `intent` 讓 Tavily 依使用者問題挑選相關內容。

拒絕 localhost、`.local`、`.internal`、私有/loopback/link-local/reserved IP、URL 內嵌帳密與非 HTTP(S) scheme。

## Environment

```text
TAVILY_API_KEY=
WEB_RESEARCH_MAX_RESULTS=5
WEB_RESEARCH_TIMEOUT_SECONDS=20
```

所有呼叫都由 DC_BOT 主機主動使用 outbound HTTPS 443，不需要新增 inbound port。

## Security

搜尋結果與網頁內容一律是不受信任資料。工具不執行頁面中的指令，也不取得 Discord Bot token、Database 或 MusicManager。Provider 錯誤只回抽象化訊息，不回 API key 或 provider response body；工具結果有長度上限，避免大量網頁內容直接灌入 Gemini context。
