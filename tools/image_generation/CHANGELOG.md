# Changelog

## 0.2.0 - 2026-09-06

- 新增墨雪 canonical self-appearance grounding；自畫時固定銀白偏淡紫長髮、紫色眼睛、白色貓耳、黑色蝴蝶結與星空魔法風格。
- 貓又尾巴數量固定為兩條；舊橫幅多出的第三條明確視為 AI 生圖瑕疵。
- 新增 `use_moxue_appearance`，且 Core 會在「畫你自己／墨雪／Nyxie」類請求強制啟用，避免模型漏掉角色設定。

## 0.1.0 - 2026-09-06

- 新增 Cloudflare Workers AI FLUX.1 Schnell `generate_image` action。
- Base64 僅在 Tool 內 decode，生成檔以相對 artifact descriptor 交給 Core，不回灌 Gemini。
- 加入使用者 cooldown、bounded concurrency、steps/timeout 上限、圖片 magic-byte/大小驗證與安全 provider error handling。
