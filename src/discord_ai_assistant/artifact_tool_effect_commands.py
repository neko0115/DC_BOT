from __future__ import annotations

import logging

import discord

from discord_ai_assistant.ai.gemini import AssistantReply
from discord_ai_assistant.artifact_delivery import ResolvedArtifact, resolve_artifact_effect
from discord_ai_assistant.tool_effect_commands import ToolEffectAssistantCommands

LOGGER = logging.getLogger(__name__)


class ArtifactToolEffectAssistantCommands(ToolEffectAssistantCommands, name="AssistantCommands"):
    """Tool-effect commands with core-owned local artifact delivery."""

    async def apply_tool_effects(
        self,
        guild_id: int,
        origin_channel_id: int,
        reply: AssistantReply,
    ) -> str | None:
        publication = await super().apply_tool_effects(guild_id, origin_channel_id, reply)
        artifact_effects = tuple(effect for effect in reply.effects if effect.get("type") == "attach_artifact")
        if not artifact_effects:
            return publication

        artifacts: list[ResolvedArtifact] = []
        rejected = 0
        for effect in artifact_effects:
            artifact = resolve_artifact_effect(
                self.settings.project_root,
                self.settings.max_upload_bytes,
                dict(effect),
            )
            if artifact is None:
                rejected += 1
                continue
            artifacts.append(artifact)

        if not artifacts:
            LOGGER.warning("All %s tool artifact(s) failed core validation", len(artifact_effects))
            return publication or "圖片已生成，但產物沒有通過墨雪核心的附件安全驗證。"

        channel = self.bot.get_channel(origin_channel_id)
        channel_guild = getattr(channel, "guild", None)
        send = getattr(channel, "send", None)
        if channel is None or getattr(channel_guild, "id", None) != guild_id or not callable(send):
            LOGGER.warning(
                "Rejected artifact delivery for invalid origin channel %s in guild %s",
                origin_channel_id,
                guild_id,
            )
            self._cleanup_artifacts(artifacts)
            return publication or "圖片已生成，但墨雪無法安全地把附件送到這個頻道。"

        files = [discord.File(str(item.path), filename=item.filename) for item in artifacts]
        try:
            await send(files=files, allowed_mentions=discord.AllowedMentions.none())
        except (discord.Forbidden, discord.HTTPException):
            LOGGER.exception(
                "Could not send %s tool artifact(s) to channel %s in guild %s",
                len(artifacts),
                origin_channel_id,
                guild_id,
            )
            return publication or "圖片已生成，但 Discord 附件傳送失敗。"
        finally:
            for file in files:
                file.close()
            self._cleanup_artifacts(artifacts)

        if rejected:
            LOGGER.warning(
                "Delivered %s artifact(s) but rejected %s invalid artifact effect(s)",
                len(artifacts),
                rejected,
            )
        return publication

    @staticmethod
    def _cleanup_artifacts(artifacts: list[ResolvedArtifact]) -> None:
        for item in artifacts:
            if not item.delete_after_send:
                continue
            try:
                item.path.unlink(missing_ok=True)
            except OSError:
                LOGGER.warning("Could not clean tool artifact %s", item.path, exc_info=True)
