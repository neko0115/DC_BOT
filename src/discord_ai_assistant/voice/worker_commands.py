from __future__ import annotations

import discord
from discord import app_commands
from discord.ext import commands

from discord_ai_assistant.config import Settings
from discord_ai_assistant.storage.database import Database
from discord_ai_assistant.voice.worker_pool import WorkerPoolSpeechSynthesizer


class TTSWorkerCommands(commands.Cog):
    def __init__(
        self,
        settings: Settings,
        database: Database,
        router: WorkerPoolSpeechSynthesizer,
    ) -> None:
        self.settings = settings
        self.database = database
        self.router = router

    @app_commands.command(name="tts_compute_status", description="查看墨雪目前的 TTS 算力節點與 GPU 狀態")
    async def tts_compute_status(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        health = await self.router.status()
        lines = [f"目前模式：`{self.router.mode}`"]
        if not health:
            lines.append("尚未設定遠端 TTS Worker，會直接使用本機/既有 TTS fallback。")
        for item in health:
            gpu = ""
            if item.gpu_utilization is not None:
                gpu += f" | GPU {item.gpu_utilization:.0f}%"
            if item.gpu_memory_free_mb is not None:
                gpu += f" | VRAM free {item.gpu_memory_free_mb} MB"

            queue = ""
            if item.active_requests is not None:
                queue += f" | active {item.active_requests}"
            if item.queued_requests is not None:
                queue += f" | queued {item.queued_requests}"
                if item.max_queue_size is not None:
                    queue += f"/{item.max_queue_size}"

            reason = f" | {item.reason}" if item.reason else ""
            lines.append(f"- `{item.name}`: **{item.status}**{gpu}{queue}{reason}")
        await interaction.followup.send("\n".join(lines)[:2000], ephemeral=True)

    @app_commands.command(name="tts_compute", description="切換墨雪使用的 TTS 算力節點（DJ）")
    @app_commands.describe(mode="auto，或設定在 TTS_REMOTE_WORKERS 中的名稱，例如 desktop / laptop / local")
    async def tts_compute(self, interaction: discord.Interaction, mode: str) -> None:
        member = interaction.user if isinstance(interaction.user, discord.Member) else None
        if not member or not self._is_dj_member(member):
            await interaction.response.send_message("只有 DJ 或管理員可切換墨雪的 TTS 算力節點。", ephemeral=True)
            return
        try:
            self.router.set_mode(mode)
        except ValueError as error:
            await interaction.response.send_message(str(error), ephemeral=True)
            return
        self.database.set_state("tts_worker_mode", self.router.mode)
        await interaction.response.send_message(
            f"TTS 算力模式已切換為 `{self.router.mode}`。可用 `/tts_compute_status` 查看節點狀態。",
            ephemeral=True,
        )

    def _is_dj_member(self, member: discord.Member) -> bool:
        if member.guild_permissions.manage_guild:
            return True
        if self.settings.dj_role_id is not None:
            return any(role.id == self.settings.dj_role_id for role in member.roles)
        return any(role.name == self.settings.dj_role_name for role in member.roles)
