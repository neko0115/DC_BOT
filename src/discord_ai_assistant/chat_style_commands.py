from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import discord
from discord import app_commands

from discord_ai_assistant.app_knowledge import AppKnowledgeStore
from discord_ai_assistant.chat_style_runtime import ChatStyleRuntime
from discord_ai_assistant.memory_inspection_commands import _ephemeral_send, _is_memory_admin
from discord_ai_assistant.knowledge_enrichment import KnowledgeEnrichmentStore


LOGGER = logging.getLogger(__name__)
_COMMAND_NAMES = ("chat_profile", "chat_profile_reset", "chat_learning", "lexicon_search", "lexicon_forget")
_STYLE_LABELS = {
    "formality": "語氣", "length": "篇幅", "teasing": "玩笑", "reply_chain": "接話",
    "emoji": "表情符號", "punctuation": "標點", "code_switching": "語言混用", "directness": "直接程度",
}
_VALUE_LABELS = {
    "casual": "隨性", "balanced": "適中", "formal": "正式", "short": "精簡", "detailed": "詳細",
    "low": "低", "medium": "中", "high": "高", "light": "少", "normal": "一般", "heavy": "多",
}
_SUMMARY_LIMIT = 5


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _display(text: str, limit: int) -> str:
    escaped = discord.utils.escape_mentions(discord.utils.escape_markdown(" ".join(text.split())))
    return escaped if len(escaped) <= limit else escaped[:limit - 1] + "…"


def _profile_text(runtime: ChatStyleRuntime, guild_id: int, user_id: int) -> str:
    store = runtime.store
    profile = store.get_profile(guild_id, user_id)
    # Current counts still apply while opted out; eligibility deliberately returns
    # zero early in that case. Never render the raw samples used for these counts.
    samples = store.list_samples(guild_id, user_id, now=_now())
    watermark = profile.sample_watermark_id if profile is not None else 0
    lines = ["**你的聊天風格（本伺服器）**"]
    if profile is None:
        lines.append("回覆風格：尚未建立。")
    else:
        style = profile.profile.get("style", {})
        summary = [f"{label}：{_VALUE_LABELS[value]}" for key, label in _STYLE_LABELS.items()
                   if isinstance(value := style.get(key), str) and value in _VALUE_LABELS]
        lines.append("回覆風格：" + ("；".join(summary) or "尚無可用摘要"))
    # Personal C belongs to Chat Style, not the public guild lexicon. Only show
    # this owner's derived aids, not profile JSON or guild evidence.
    terms = store.connection.execute(
        "SELECT term, meaning FROM chat_style_terms WHERE guild_id = ? AND user_id = ? "
        "AND status = 'active' ORDER BY term LIMIT ?", (guild_id, user_id, _SUMMARY_LIMIT),
    ).fetchall()
    lines.append("理解輔助（最多 5 項）：" if terms else "理解輔助：尚未建立。")
    lines.extend(f"• {_display(term, 60)}：{_display(meaning, 150)}" for term, meaning in terms)
    last_update = (profile.last_successful_update_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                   if profile is not None else "尚無成功更新")
    lines.extend([
        f"有效樣本：{len(samples)}",
        f"上次成功更新後新增：{sum(sample.id > watermark for sample in samples)}",
        f"上次成功更新：{last_update}",
        f"聊天風格學習：{'啟用' if store.learning_enabled(guild_id, user_id) else '停用'}",
    ])
    return "\n".join(lines)


async def _reset(runtime: ChatStyleRuntime, guild_id: int, user_id: int) -> None:
    key = (guild_id, user_id)
    # Share the observer's lock so an in-flight summary cannot resurrect the
    # profile after reset completes. Keep the lock for already queued observers.
    async with runtime._locks.setdefault(key, asyncio.Lock()):
        connection = runtime.store.connection
        with connection:
            for table in ("chat_style_profiles", "chat_style_terms", "chat_style_samples", "chat_style_profile_updates"):
                connection.execute(f"DELETE FROM {table} WHERE guild_id = ? AND user_id = ?", key)
        runtime._last_attempt.pop(key, None)
    # Learning is an independent, authoritative opt-out; reset must preserve it.


async def _command_error(interaction: discord.Interaction, error: app_commands.AppCommandError) -> None:
    LOGGER.warning("Chat Style control failed (%s)", type(error).__name__)
    await _ephemeral_send(interaction, "操作未完成，請稍後再試。")


def install_chat_style_commands(grouped_commands: Any, runtime: ChatStyleRuntime) -> None:
    """Idempotently extend the existing /memory group before tree registration/sync."""
    group = grouped_commands.memory
    if all(group.get_command(name) is not None for name in _COMMAND_NAMES):
        return
    core = grouped_commands.core
    lexicon = KnowledgeEnrichmentStore(runtime.store.database, AppKnowledgeStore(runtime.store.database))

    async def allowed(interaction: discord.Interaction, *, admin: bool = False) -> bool:
        if interaction.guild is None:
            await _ephemeral_send(interaction, "這個指令只能在伺服器內使用。")
            return False
        if admin and not _is_memory_admin(core, interaction):
            await _ephemeral_send(interaction, "這個指令只有 DJ 或管理員可以使用。")
            return False
        return True

    @app_commands.command(name="chat_profile", description="私下查看你在本伺服器的聊天風格與學習狀態")
    async def chat_profile(interaction: discord.Interaction) -> None:
        if await allowed(interaction):
            await _ephemeral_send(interaction, _profile_text(runtime, interaction.guild.id, interaction.user.id))

    @app_commands.command(name="chat_profile_reset", description="重設你在本伺服器的聊天風格資料，保留學習開關")
    async def chat_profile_reset(interaction: discord.Interaction) -> None:
        if not await allowed(interaction):
            return
        await interaction.response.defer(ephemeral=True, thinking=True)
        await _reset(runtime, interaction.guild.id, interaction.user.id)
        await _ephemeral_send(interaction, "已重設你在本伺服器的聊天風格、理解輔助、樣本與更新計數；學習開關維持原狀。")

    @app_commands.command(name="chat_learning", description="啟用或停用你在本伺服器的聊天風格學習")
    @app_commands.describe(mode="停用會保留既有資料；刪除請使用 chat_profile_reset")
    @app_commands.choices(mode=[app_commands.Choice(name="啟用", value="啟用"), app_commands.Choice(name="停用", value="停用")])
    async def chat_learning(interaction: discord.Interaction, mode: str) -> None:
        if not await allowed(interaction):
            return
        if mode not in {"啟用", "停用"}:
            await _ephemeral_send(interaction, "請選擇「啟用」或「停用」。")
            return
        # Do not wait for the summary lock: the runtime rechecks this state after
        # every model await, so opt-out immediately blocks sampling and writes.
        runtime.store.set_learning_enabled(interaction.guild.id, interaction.user.id, mode == "啟用", updated_at=_now())
        await _ephemeral_send(interaction, f"已{mode}你在本伺服器的聊天風格學習；既有資料保留。")

    @app_commands.command(name="lexicon_search", description="DJ／管理員搜尋本伺服器已啟用的共用詞彙")
    @app_commands.describe(term="要搜尋的詞彙（最多 80 字）")
    async def lexicon_search(interaction: discord.Interaction, term: app_commands.Range[str, 0, 80]) -> None:
        if not await allowed(interaction, admin=True):
            return
        if len(term) > 80:
            await _ephemeral_send(interaction, "搜尋詞彙最多 80 字。")
            return
        entries = lexicon.search_guild_lexicon(interaction.guild.id, term)[:_SUMMARY_LIMIT]
        if not entries:
            await _ephemeral_send(interaction, "沒有找到本伺服器已啟用的共用詞彙。")
            return
        lines = ["**本伺服器共用詞彙（最多 5 筆）**"]
        lines.extend(f"• ID {entry.id}｜{_display(entry.term, 60)}：{_display(entry.meaning, 150)}" for entry in entries)
        await _ephemeral_send(interaction, "\n".join(lines))

    @app_commands.command(name="lexicon_forget", description="DJ／管理員刪除本伺服器指定的共用詞彙")
    @app_commands.describe(entry_id="lexicon_search 顯示的詞彙 ID")
    @app_commands.rename(entry_id="id")
    async def lexicon_forget(interaction: discord.Interaction, entry_id: int) -> None:
        if not await allowed(interaction, admin=True):
            return
        removed = (0 < entry_id < 2**63 and lexicon.forget_guild_lexicon_entry(interaction.guild.id, entry_id))
        await _ephemeral_send(interaction, "已刪除本伺服器的指定共用詞彙。" if removed else "找不到本伺服器的指定共用詞彙。")

    for command in (chat_profile, chat_profile_reset, chat_learning, lexicon_search, lexicon_forget):
        if group.get_command(command.name) is None:
            command.error(_command_error)
            group.add_command(command)
