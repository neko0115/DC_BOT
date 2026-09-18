# Image Generation Tool

使用 Cloudflare Workers AI 的 `@cf/black-forest-labs/flux-1-schnell` 產生圖片。只有使用者明確要求生圖、畫圖或產生圖片時才應把 action 暴露給 Gemini；禁止 proactive 自行生圖。

## Action

### `generate_image`

接受 `prompt`，另有可選的 `use_moxue_appearance` boolean。當使用者要求畫墨雪／墨染雪／Nyxie／「你自己」時，Core 會強制此 flag 為 `true`；Tool 也會對 prompt 中明確出現的墨雪名稱自動補 canonical profile。

墨雪的 canonical 外觀由 `discord_ai_assistant.character_profile` 單一來源管理。尾巴固定為 **兩條**；舊橫幅若看起來有第三條，視為 AI 生圖瑕疵，不可用來覆蓋角色設定。自畫 prompt 會加入 `exactly two tails` 約束，並固定銀白偏淡紫長髮、紫色眼睛、白色貓耳、黑色蝴蝶結、星形元素與深紫魔法系造型。

模型名稱、steps、timeout、concurrency 與 cooldown 都由應用環境設定控制，不交給 Gemini 自行提高成本。

Cloudflare 回傳的 Base64 圖片會在 Tool 內立刻 decode，驗證 JPEG/PNG/WebP magic bytes 後寫到：

```text
data/tool-artifacts/image_generation/
```

Tool result 不包含 Base64；只以 `_moxue_artifacts` 回傳受限的相對 artifact descriptor，由墨雪 Core 再驗證並送到 Discord。

## Environment

```text
CLOUDFLARE_ACCOUNT_ID=
CLOUDFLARE_API_TOKEN=
IMAGE_GEN_MODEL=@cf/black-forest-labs/flux-1-schnell
IMAGE_GEN_STEPS=4
IMAGE_GEN_TIMEOUT_SECONDS=50
IMAGE_GEN_USER_COOLDOWN_SECONDS=60
IMAGE_GEN_MAX_CONCURRENCY=1
```

`IMAGE_GEN_STEPS` 強制限制在 1–8；預設 4。生成圖片最大接受 10 MiB。Provider traffic 只需要 DC_BOT 主機 outbound HTTPS 443。

## Security

Tool 不取得 Discord token 或 Bot instance；Base64 不進 Gemini context。Artifact descriptor 不能直接等同 Discord 發送權限，Core 仍須驗證 artifact root containment、symlink、MIME、大小與 cleanup。Provider 錯誤不包含 API token 或完整 response body。
