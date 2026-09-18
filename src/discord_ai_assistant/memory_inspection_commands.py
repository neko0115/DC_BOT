from __future__ import annotations

from typing import Any

import discord
from discord import app_commands

from discord_ai_assistant.ai.memory_domain_registry import GAME_SUBDOMAINS
from discord_ai_assistant.ai.memory_phase4 import (
    list_open_memory_conflicts,
    memory_provenance_rows,
)
from discord_ai_assistant.ai.memory_project_state import list_user_projects, project_snapshot_text
from discord_ai_assistant.ai.memory_shared import delete_guild_memory, search_guild_memories


_CHANNEL_MEMORY_MODES = {"auto", "mixed", "social", "game", "project", "off"}


def _ephemeral_send(interaction: discord.Interaction, text: str) -> Any:
    if interaction.response.is_done():
        return interaction.followup.send(text[:2000], ephemeral=True)
    return interaction.response.send_message(text[:2000], ephemeral=True)


def normalize_memory_channel_mode(value: str) -> str | None:
    """Validate a per-channel passive-memory policy override."""

    normalized = " ".join(value.split()).casefold()
    if normalized in _CHANNEL_MEMORY_MODES:
        return normalized
    if normalized.startswith("game:"):
        subdomain = normalized.split(":", 1)[1].strip()
        if subdomain in GAME_SUBDOMAINS:
            return f"game:{subdomain}"
    return None


def _is_memory_admin(core: Any, interaction: discord.Interaction) -> bool:
    checker = getattr(core, "_is_dj", None)
    return bool(callable(checker) and checker(interaction))


def install_memory_inspection_commands(grouped_commands: Any) -> None:
    """Attach Memory V2 inspection/admin commands to the existing `/memory` group."""

    group = grouped_commands.memory
    core = grouped_commands.core

    if group.get_command("search") is None:
        @app_commands.command(name="search", description="查看 Memory V2 會為某個問題召回哪些記憶")
        @app_commands.describe(query="要模擬的問題或關鍵字")
        async def memory_search(interaction: discord.Interaction, query: str) -> None:
            if not interaction.guild:
                await _ephemeral_send(interaction, "這個指令只能在伺服器內使用。")
                return
            matches = core.database.search_user_memories(
                interaction.guild.id,
                interaction.user.id,
                query,
                limit=10,
            )
            if not matches:
                await _ephemeral_send(interaction, "沒有找到可用的 active Memory V2 記憶。")
                return
            lines = [f"**Memory V2 搜尋：{query[:80]}**"]
            for index, match in enumerate(matches, start=1):
                project = f" | project={match.project}" if match.project else ""
                subject = f" | subject={match.subject}" if match.subject else ""
                lines.append(
                    f"{index}. `{match.id}` score={match.score:.3f} | {match.kind}{project}{subject}\n"
                    f"   【{match.category}】{match.content}"
                )
            await _ephemeral_send(interaction, "\n".join(lines))

        group.add_command(memory_search)

    if group.get_command("projects") is None:
        @app_commands.command(name="projects", description="列出 Memory V2 目前辨識到的你的專案")
        async def memory_projects(interaction: discord.Interaction) -> None:
            if not interaction.guild:
                await _ephemeral_send(interaction, "這個指令只能在伺服器內使用。")
                return
            projects = list_user_projects(
                core.database,
                interaction.guild.id,
                interaction.user.id,
                limit=25,
            )
            if not projects:
                await _ephemeral_send(interaction, "目前沒有可用的 active 專案記憶。")
                return
            await _ephemeral_send(
                interaction,
                "**Memory V2 專案**\n" + "\n".join(f"- {project}" for project in projects),
            )

        group.add_command(memory_projects)

    if group.get_command("project") is None:
        @app_commands.command(name="project", description="查看 Memory V2 對某個專案保存的目前狀態與近期歷史")
        @app_commands.describe(name="專案名稱，例如 YuuPo 或 DC_BOT")
        async def memory_project(interaction: discord.Interaction, name: str) -> None:
            if not interaction.guild:
                await _ephemeral_send(interaction, "這個指令只能在伺服器內使用。")
                return
            text = project_snapshot_text(
                core.database,
                interaction.guild.id,
                interaction.user.id,
                name,
            )
            await _ephemeral_send(interaction, text)

        group.add_command(memory_project)

    if group.get_command("provenance") is None:
        @app_commands.command(name="provenance", description="查看一項被動記憶由哪些 Discord 訊息批次產生")
        @app_commands.describe(memory_id="使用 /memory list 或 /memory search 顯示的記憶編號")
        async def memory_provenance(interaction: discord.Interaction, memory_id: int) -> None:
            if not interaction.guild:
                await _ephemeral_send(interaction, "這個指令只能在伺服器內使用。")
                return
            rows = memory_provenance_rows(
                core.database,
                interaction.guild.id,
                interaction.user.id,
                memory_id,
                limit=10,
            )
            if not rows:
                await _ephemeral_send(interaction, "這項記憶沒有可用的被動來源紀錄，或它不屬於你。")
                return
            lines = [f"**Memory V2 provenance：`{memory_id}`**"]
            for row in rows:
                observed = row["observed_at"] or row["created_at"]
                session = f" | session=`{row['session_key']}`" if row["session_key"] else ""
                lines.append(
                    f"- <#{row['channel_id']}> message=`{row['message_id']}` | {observed} | batch=`{row['batch_id']}`{session}"
                )
            await _ephemeral_send(interaction, "\n".join(lines))

        group.add_command(memory_provenance)

    if group.get_command("conflicts") is None:
        @app_commands.command(name="conflicts", description="查看 Memory V2 保留、尚未自動覆蓋的潛在衝突事實")
        async def memory_conflicts(interaction: discord.Interaction) -> None:
            if not interaction.guild:
                await _ephemeral_send(interaction, "這個指令只能在伺服器內使用。")
                return
            rows = list_open_memory_conflicts(
                core.database,
                interaction.guild.id,
                interaction.user.id,
                limit=10,
            )
            if not rows:
                await _ephemeral_send(interaction, "目前沒有 open Memory V2 conflict。")
                return
            lines = ["**Memory V2 潛在衝突**"]
            for row in rows:
                project = f" | {row['project']}" if row["project"] else ""
                lines.append(
                    f"- conflict `{row['id']}`{project} similarity={float(row['similarity']):.2f}\n"
                    f"  `{row['left_id']}` {str(row['left_content'])[:180]}\n"
                    f"  `{row['right_id']}` {str(row['right_content'])[:180]}"
                )
            await _ephemeral_send(interaction, "\n".join(lines))

        group.add_command(memory_conflicts)

    if group.get_command("channel") is None:
        @app_commands.command(name="channel", description="設定目前頻道的 Memory V2 被動學習模式（DJ/管理員）")
        @app_commands.describe(mode="auto/mixed/social/game/game:<game>/project/off")
        async def memory_channel(interaction: discord.Interaction, mode: str) -> None:
            if not interaction.guild or interaction.channel_id is None:
                await _ephemeral_send(interaction, "這個指令只能在伺服器頻道內使用。")
                return
            if not _is_memory_admin(core, interaction):
                await _ephemeral_send(interaction, "只有 DJ 或管理員可以調整頻道記憶模式。")
                return
            normalized = normalize_memory_channel_mode(mode)
            if normalized is None:
                examples = ", ".join(sorted(GAME_SUBDOMAINS))
                await _ephemeral_send(
                    interaction,
                    "不支援這個模式。可用：`auto`、`mixed`、`social`、`game`、"
                    f"`game:<subdomain>`、`project`、`off`。已知 game subdomain：{examples}",
                )
                return
            core.database.set_state(
                f"memory_channel_mode:{interaction.guild.id}:{interaction.channel_id}",
                normalized,
            )
            await _ephemeral_send(
                interaction,
                f"目前頻道的 Memory V2 模式已設為 `{normalized}`。",
            )

        group.add_command(memory_channel)

    if group.get_command("shared_search") is None:
        @app_commands.command(name="shared_search", description="搜尋此伺服器已啟用的公開 shared memory")
        @app_commands.describe(query="群體事件、遊戲或梗的關鍵字")
        async def memory_shared_search(interaction: discord.Interaction, query: str) -> None:
            if not interaction.guild:
                await _ephemeral_send(interaction, "這個指令只能在伺服器內使用。")
                return
            matches = search_guild_memories(
                core.database,
                interaction.guild.id,
                query,
                limit=10,
            )
            if not matches:
                await _ephemeral_send(interaction, "沒有找到 active shared memory。")
                return
            lines = [f"**Shared Memory 搜尋：{query[:80]}**"]
            for index, match in enumerate(matches, start=1):
                scope = match.domain + (f"/{match.subdomain}" if match.subdomain else "")
                lines.append(
                    f"{index}. `{match.id}` score={match.score:.3f} | {scope} | {match.kind} | "
                    f"retention={match.retention} reinforce={match.reinforcement_count}\n"
                    f"   {match.content}"
                )
            await _ephemeral_send(interaction, "\n".join(lines))

        group.add_command(memory_shared_search)

    if group.get_command("shared_forget") is None:
        @app_commands.command(name="shared_forget", description="刪除此伺服器的一項 shared memory（DJ/管理員）")
        @app_commands.describe(memory_id="使用 /memory shared_search 顯示的 shared memory 編號")
        async def memory_shared_forget(interaction: discord.Interaction, memory_id: int) -> None:
            if not interaction.guild:
                await _ephemeral_send(interaction, "這個指令只能在伺服器內使用。")
                return
            if not _is_memory_admin(core, interaction):
                await _ephemeral_send(interaction, "只有 DJ 或管理員可以刪除 shared memory。")
                return
            deleted = delete_guild_memory(core.database, interaction.guild.id, memory_id)
            await _ephemeral_send(
                interaction,
                "已刪除這項 shared memory。" if deleted else "找不到這項 shared memory，或它不屬於此伺服器。",
            )

        group.add_command(memory_shared_forget)
