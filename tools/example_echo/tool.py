from __future__ import annotations


async def invoke(
    action: str,
    arguments: dict[str, object],
    context: dict[str, object],
) -> dict[str, object]:
    if action == "echo":
        text = arguments.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        return {"message": text.strip(), "source": "example_echo"}

    if action == "status":
        return {
            "message": "外接工具 Gateway 運作正常。",
            "guild_id": context.get("guild_id"),
            "user_id": context.get("user_id"),
            "is_dj": bool(context.get("is_dj", False)),
        }

    raise ValueError(f"unsupported action: {action}")
