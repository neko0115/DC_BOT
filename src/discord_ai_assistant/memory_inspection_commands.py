from __future__ import annotations

from typing import Any

import discord
from discord import app_commands

from discord_ai_assistant.ai.memory_phase4 import (
    list_open_memory_conflicts,
    memory_provenance_rows,
)
from discord_ai_assistant.ai.memory_project_state import list_user_projects, project_snapshot_text


def _ephemeral_send(interaction: discord.Interaction, text: str) -> Any:
    if interaction.response.is_done():
        return interaction.followup.send(text[:2000], ephemeral=True)
    return interaction.response.send_message(text[:2000], ephemeral=True)


def install_memory_inspection_commands(grouped_commands: Any) -> None:
    """Attach Memory V2 inspection commands to the existing `/memory` group."""

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
                lines.append(
                    f"- <#{row['channel_id']}> message=`{row['message_id']}` | {observed} | batch=`{row['batch_id']}`"
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
