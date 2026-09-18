from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import discord
from discord import app_commands


@dataclass(frozen=True, slots=True)
class MemoryAddChoice:
    label: str
    storage_category: str


MEMORY_ADD_CHOICES = (
    MemoryAddChoice("偏好 / 喜好", "偏好"),
    MemoryAddChoice("習慣", "習慣"),
    MemoryAddChoice("興趣", "興趣"),
    MemoryAddChoice("個人資訊 / 事實", "個人資訊"),
    MemoryAddChoice("提醒", "提醒"),
    MemoryAddChoice("其他", "其他"),
)
_MEMORY_ADD_BY_LABEL = {choice.label: choice for choice in MEMORY_ADD_CHOICES}


def normalize_memory_add_category(value: str) -> MemoryAddChoice | None:
    """Resolve only the fixed public `/memory add` category values."""

    return _MEMORY_ADD_BY_LABEL.get(" ".join(value.split()))


def store_manual_memory_choice(
    database: Any,
    guild_id: int,
    user_id: int,
    category: str,
    content: str,
):
    """Validate one public category choice and delegate storage safety to Memory V2."""

    choice = normalize_memory_add_category(category)
    if choice is None:
        raise ValueError("不支援這個記憶分類，請使用 /memory add 提供的固定選項。")
    return database.add_user_memory(guild_id, user_id, choice.storage_category, content)


async def _ephemeral_send(interaction: discord.Interaction, text: str) -> None:
    if interaction.response.is_done():
        await interaction.followup.send(text[:2000], ephemeral=True)
        return
    await interaction.response.send_message(text[:2000], ephemeral=True)


def install_memory_add_command(grouped_commands: Any) -> None:
    """Replace the compatibility wrapper with the fixed-choice public Memory V2 UX."""

    group = grouped_commands.memory
    core = grouped_commands.core
    if group.get_command("add") is not None:
        group.remove_command("add")

    @app_commands.command(name="add", description="請墨雪記住你的偏好、習慣或提醒")
    @app_commands.describe(category="記憶類型", content="要記住的自由文字內容")
    @app_commands.choices(
        category=[
            app_commands.Choice(name=choice.label, value=choice.label)
            for choice in MEMORY_ADD_CHOICES
        ]
    )
    async def memory_add(
        interaction: discord.Interaction,
        category: str,
        content: str,
    ) -> None:
        if not interaction.guild:
            await _ephemeral_send(interaction, "記憶只能在伺服器中使用。")
            return
        try:
            memory = store_manual_memory_choice(
                core.database,
                interaction.guild.id,
                interaction.user.id,
                category,
                content,
            )
        except ValueError as error:
            await _ephemeral_send(interaction, str(error))
            return
        await _ephemeral_send(
            interaction,
            f"已記住：`{memory.id}` 【{memory.category}】{memory.content}",
        )

    group.add_command(memory_add)
