from __future__ import annotations

import asyncio
import base64
import binascii
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

import aiohttp

from discord_ai_assistant.character_profile import build_moxue_image_prompt, is_moxue_self_reference

DEFAULT_MODEL = "@cf/black-forest-labs/flux-1-schnell"
MAX_PROMPT_CHARACTERS = 2048
MAX_PROVIDER_PROMPT_CHARACTERS = 4096
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MODEL_PATTERN = re.compile(r"^@cf/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
ACCOUNT_PATTERN = re.compile(r"^[A-Za-z0-9_-]{6,128}$")
_LAST_USER_REQUEST: dict[int, float] = {}
_SEMAPHORE: asyncio.Semaphore | None = None
_SEMAPHORE_LIMIT: int | None = None
_SEMAPHORE_LOOP: asyncio.AbstractEventLoop | None = None


class ImageGenerationError(RuntimeError):
    """Provider failure translated into a safe tool result."""


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name, "").strip()
    try:
        value = float(raw) if raw else default
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))


def _semaphore() -> asyncio.Semaphore:
    global _SEMAPHORE, _SEMAPHORE_LIMIT, _SEMAPHORE_LOOP
    loop = asyncio.get_running_loop()
    limit = _env_int("IMAGE_GEN_MAX_CONCURRENCY", 1, 1, 4)
    if _SEMAPHORE is None or _SEMAPHORE_LIMIT != limit or _SEMAPHORE_LOOP is not loop:
        _SEMAPHORE = asyncio.Semaphore(limit)
        _SEMAPHORE_LIMIT = limit
        _SEMAPHORE_LOOP = loop
    return _SEMAPHORE


def _model_name() -> str:
    value = os.getenv("IMAGE_GEN_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    if not MODEL_PATTERN.fullmatch(value):
        raise ImageGenerationError("IMAGE_GEN_MODEL 格式不安全或不受支援。")
    return value


def _prompt(arguments: dict[str, object]) -> str:
    value = arguments.get("prompt")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("prompt is required.")
    text = value.strip()
    if len(text) > MAX_PROMPT_CHARACTERS:
        raise ValueError(f"prompt 最多 {MAX_PROMPT_CHARACTERS} 個字元。")

    use_moxue_appearance = arguments.get("use_moxue_appearance", False)
    if not isinstance(use_moxue_appearance, bool):
        raise ValueError("use_moxue_appearance 必須是 boolean。")
    if use_moxue_appearance or is_moxue_self_reference(text):
        text = build_moxue_image_prompt(text)
    if len(text) > MAX_PROVIDER_PROMPT_CHARACTERS:
        raise ValueError(f"加入角色設定後的 prompt 最多 {MAX_PROVIDER_PROMPT_CHARACTERS} 個字元。")
    return text


def _detect_image_type(data: bytes) -> tuple[str, str]:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg", ".jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png", ".png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp", ".webp"
    raise ImageGenerationError("Cloudflare 回傳的資料不是允許的圖片格式。")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_artifact(data: bytes, *, root: Path | None = None) -> dict[str, object]:
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ImageGenerationError("生成圖片大小異常，已拒絕保存。")
    mime_type, extension = _detect_image_type(data)
    base_root = root if root is not None else _project_root() / "data" / "tool-artifacts"
    directory = base_root / "image_generation"
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex}{extension}"
    final_path = directory / name
    temporary_path = directory / f".{name}.tmp"
    temporary_path.write_bytes(data)
    os.replace(temporary_path, final_path)
    return {
        "relative_path": f"image_generation/{name}",
        "filename": f"moxue-generated-{name[:8]}{extension}",
        "mime_type": mime_type,
        "delete_after_send": True,
    }


def _cooldown_remaining(context: dict[str, object]) -> float:
    user_id = context.get("user_id")
    if not isinstance(user_id, int) or user_id <= 0:
        return 0.0
    cooldown = _env_float("IMAGE_GEN_USER_COOLDOWN_SECONDS", 60.0, 0.0, 3600.0)
    elapsed = time.monotonic() - _LAST_USER_REQUEST.get(user_id, 0.0)
    return max(0.0, cooldown - elapsed)


def _record_request(context: dict[str, object]) -> None:
    user_id = context.get("user_id")
    if isinstance(user_id, int) and user_id > 0:
        _LAST_USER_REQUEST[user_id] = time.monotonic()


async def _request_image(prompt: str) -> tuple[bytes, str]:
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
    api_token = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
    if not account_id or not api_token:
        raise ImageGenerationError("尚未設定 CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN。")
    if not ACCOUNT_PATTERN.fullmatch(account_id):
        raise ImageGenerationError("CLOUDFLARE_ACCOUNT_ID 格式不正確。")
    model = _model_name()
    steps = _env_int("IMAGE_GEN_STEPS", 4, 1, 8)
    timeout_seconds = _env_float("IMAGE_GEN_TIMEOUT_SECONDS", 50.0, 10.0, 120.0)
    endpoint = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
    headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    try:
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.post(endpoint, json={"prompt": prompt, "steps": steps}) as response:
                if response.status == 200:
                    payload: Any = await response.json()
                elif response.status in {401, 403}:
                    raise ImageGenerationError("Cloudflare Workers AI Token 或帳號權限無效。")
                elif response.status == 429:
                    raise ImageGenerationError("Cloudflare Workers AI 目前碰到速率或免費額度限制。")
                elif response.status >= 500:
                    raise ImageGenerationError("Cloudflare Workers AI 服務暫時不可用。")
                else:
                    raise ImageGenerationError(f"Cloudflare 拒絕了這次生圖請求（HTTP {response.status}）。")
    except ImageGenerationError:
        raise
    except (aiohttp.ClientError, TimeoutError) as error:
        raise ImageGenerationError("連線 Cloudflare Workers AI 時逾時或網路失敗。") from error

    if not isinstance(payload, dict) or payload.get("success") is not True:
        raise ImageGenerationError("Cloudflare Workers AI 沒有成功完成生圖。")
    result = payload.get("result")
    encoded = result.get("image") if isinstance(result, dict) else None
    if not isinstance(encoded, str) or not encoded:
        raise ImageGenerationError("Cloudflare Workers AI 沒有回傳圖片資料。")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ImageGenerationError("Cloudflare Workers AI 回傳的圖片編碼無效。") from error
    _detect_image_type(data)
    return data, model


async def _generate_image(arguments: dict[str, object], context: dict[str, object]) -> dict[str, object]:
    prompt = _prompt(arguments)
    remaining = _cooldown_remaining(context)
    if remaining > 0:
        return {"message": f"生圖冷卻中，約 {max(1, int(remaining + 0.999))} 秒後可再次使用。", "error": True}
    _record_request(context)
    async with _semaphore():
        data, model = await _request_image(prompt)
        artifact = _write_artifact(data)
    return {
        "message": "圖片已生成。請用核心附件機制把生成圖片交付給使用者，不要聲稱圖片 URL 存在。",
        "provider": "cloudflare_workers_ai",
        "model": model,
        "_moxue_artifacts": [artifact],
    }


async def invoke(action: str, arguments: dict[str, object], context: dict[str, object]) -> object:
    try:
        if action == "generate_image":
            return await _generate_image(arguments, context)
        raise ValueError(f"Unknown action: {action}")
    except (ValueError, ImageGenerationError) as error:
        return {"message": str(error), "error": True}
