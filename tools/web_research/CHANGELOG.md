# Changelog

## 0.1.0 - 2026-09-06

- 新增受限 Tavily `web_research` 與 `read_webpage` actions。
- 深度研究由工具內部完成 Search → top-source Extract，避免修改既有單輪 Gemini tool loop。
- 加入公開 URL 驗證、結果長度限制、quota/rate-limit 安全錯誤與測試。
