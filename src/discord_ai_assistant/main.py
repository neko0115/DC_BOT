from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import discord
from discord.ext import commands

from discord_ai_assistant.agent import AgentCoordinator, AgentEvent, AgentEventBus, AgentEventKind
from discord_ai_assistant.agent.bridge import AgentEventBridge
from discord_ai_assistant.ai.evented_tools import EventedToolRouter
from discord_ai_assistant.ai.resilient_gemini import ResilientGeminiAssistant
from discord_ai_assistant.artifact_tool_effect_commands import ArtifactToolEffectAssistantCommands
from discord_ai_assistant.capture_agent.slash import install_capture_agent_commands
from discord_ai_assistant.capture_hub import CaptureHubServer
from discord_ai_assistant.commands import AssistantCommands
from discord_ai_assistant.config import Settings, load_settings
from discord_ai_assistant.knowledge_help_runtime import KnowledgeHelpRuntime
from discord_ai_assistant.lyrics_commands import LyricsCommands
from discord_ai_assistant.meeting_full_article_cog import MeetingReportCommands
from discord_ai_assistant.memory_inspection_commands import install_memory_inspection_commands
from discord_ai_assistant.memory_v2_runtime import MemoryV2PassiveRuntime
from discord_ai_assistant.music.evented_player import EventedEnhancedMusicManager
from discord_ai_assistant.music.library import LibraryService
from discord_ai_assistant.music.player import configure_event_loop
from discord_ai_assistant.slash_groups import GroupedSlashCommands, remove_grouped_legacy_commands
from discord_ai_assistant.storage.agent_database import AgentDatabase
from discord_ai_assistant.tool_gateway.client import ToolGatewayClient
from discord_ai_assistant.tool_gateway.server import ToolGatewayServer
from discord_ai_assistant.voice.playback_watchdog import VoicePlaybackWatchdog
from discord_ai_assistant.voice.resilient_synthesis import ResilientWindowsSpeechSynthesizer
from discord_ai_assistant.voice.worker_commands import TTSWorkerCommands
from discord_ai_assistant.voice.worker_pool import WorkerPoolSpeechSynthesizer, parse_worker_specs

LOGGER = logging.getLogger(__name__)


class AssistantBot(commands.Bot):
    def __init__(self, settings: Settings) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents)
        self.settings = settings

        self.agent_bus = AgentEventBus()
        self.agent = AgentCoordinator(self.agent_bus)
        self.database = AgentDatabase(settings.database_path, self.agent_bus)
        self.library = LibraryService(settings.library_path, self.database, settings.max_upload_bytes)
        self.music = EventedEnhancedMusicManager(
            settings.library_path,
            self.library.random_track,
            event_bus=self.agent_bus,
            youtube_cookies_from_browser=settings.youtube_cookies_from_browser,
            youtube_cookies_file=settings.youtube_cookies_file,
            youtube_po_token=settings.youtube_po_token,
        )
        self.speech_router: WorkerPoolSpeechSynthesizer | None = None

        external_client: ToolGatewayClient | None = None
        self.tool_gateway_server: ToolGatewayServer | None = None
        if settings.tool_gateway_enabled:
            external_client = ToolGatewayClient(
                host=settings.tool_gateway_host,
                port=settings.tool_gateway_port,
                token=settings.tool_gateway_token,
                refresh_seconds=settings.tool_gateway_refresh_seconds,
                request_timeout_seconds=settings.tool_gateway_timeout_seconds + 5,
            )
            if settings.tool_gateway_embedded:
                self.tool_gateway_server = ToolGatewayServer(
                    settings.tools_path,
                    host=settings.tool_gateway_host,
                    port=settings.tool_gateway_port,
                    token=settings.tool_gateway_token,
                    invocation_timeout_seconds=settings.tool_gateway_timeout_seconds,
                )

        capture_hub_enabled = os.getenv("CAPTURE_HUB_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
        self.capture_hub_server: CaptureHubServer | None = None
        if capture_hub_enabled:
            capture_hub_host = os.getenv("CAPTURE_HUB_HOST", "127.0.0.1").strip() or "127.0.0.1"
            capture_hub_port = int(os.getenv("CAPTURE_HUB_PORT", "8878"))
            self.capture_hub_server = CaptureHubServer(
                settings.project_root,
                host=capture_hub_host,
                port=capture_hub_port,
            )

        self.tool_router = EventedToolRouter(
            self.library,
            self.music,
            self.database,
            external_client,
            event_bus=self.agent_bus,
        )
        self.ai = ResilientGeminiAssistant(
            settings.gemini_api_key,
            settings.gemini_model,
            self.tool_router,
            settings.persona_timezone,
        )

    async def setup_hook(self) -> None:
        configure_event_loop(asyncio.get_running_loop())
        await self.agent.start()
        if self.tool_gateway_server is not None:
            await self.tool_gateway_server.start()
            await self.tool_router.refresh_external_tools()
        if self.capture_hub_server is not None:
            await self.capture_hub_server.start()

        core_commands = ArtifactToolEffectAssistantCommands(
            self, self.settings, self.database, self.library, self.music, self.ai
        )
        legacy_speech = ResilientWindowsSpeechSynthesizer(
            self.settings.project_root / "data" / "tts",
            primary_api_key=self.settings.gemini_api_key,
            provider=self.settings.tts_provider,
            gemini_model=self.settings.tts_gemini_model,
            gemini_voice=self.settings.tts_gemini_voice,
            gemini_style=self.settings.tts_gemini_style,
            gemini_timeout_seconds=self.settings.tts_gemini_timeout_seconds,
            kokoro_enabled=self.settings.tts_kokoro_enabled,
            kokoro_voice=self.settings.tts_kokoro_voice,
            kokoro_speed=self.settings.tts_kokoro_speed,
        )
        try:
            worker_specs = parse_worker_specs(self.settings.tts_remote_workers)
        except ValueError as error:
            raise RuntimeError(f"Invalid TTS_REMOTE_WORKERS: {error}") from error
        configured_mode = self.database.get_state("tts_worker_mode") or self.settings.tts_worker_mode
        try:
            self.speech_router = WorkerPoolSpeechSynthesizer(
                self.settings.project_root / "data" / "tts",
                worker_specs,
                token=self.settings.tts_worker_token,
                fallback=legacy_speech,
                default_mode=configured_mode,
                health_timeout_seconds=self.settings.tts_worker_health_timeout_seconds,
                request_timeout_seconds=self.settings.tts_worker_request_timeout_seconds,
                failure_cooldown_seconds=self.settings.tts_worker_failure_cooldown_seconds,
            )
        except ValueError:
            LOGGER.warning("Stored/configured TTS worker mode %r is unavailable; reverting to auto", configured_mode)
            self.speech_router = WorkerPoolSpeechSynthesizer(
                self.settings.project_root / "data" / "tts",
                worker_specs,
                token=self.settings.tts_worker_token,
                fallback=legacy_speech,
                default_mode="auto",
                health_timeout_seconds=self.settings.tts_worker_health_timeout_seconds,
                request_timeout_seconds=self.settings.tts_worker_request_timeout_seconds,
                failure_cooldown_seconds=self.settings.tts_worker_failure_cooldown_seconds,
            )
            self.database.set_state("tts_worker_mode", "auto")
        core_commands.speech = self.speech_router
        core_commands.social.session_state = core_commands.memory_session
        await self.add_cog(core_commands)
        await self.add_cog(
            MemoryV2PassiveRuntime(self, self.database, self.ai, core_commands.memory_session)
        )
        await self.add_cog(KnowledgeHelpRuntime(self, core_commands.social))
        await self.add_cog(AgentEventBridge(self, self.settings, self.agent_bus, self.database))
        await self.add_cog(TTSWorkerCommands(self.settings, self.database, self.speech_router))
        await self.add_cog(VoicePlaybackWatchdog(self, self.music))
        grouped_commands = GroupedSlashCommands(core_commands)
        install_capture_agent_commands(grouped_commands, self.capture_hub_server)
        install_memory_inspection_commands(grouped_commands)
        await self.add_cog(grouped_commands)
        await self.add_cog(LyricsCommands(self, self.music, self.database))
        remove_grouped_legacy_commands(self.tree)
        await self.add_cog(MeetingReportCommands(self, self.settings, self.ai))

        if self.settings.discord_guild_id:
            guild = discord.Object(id=self.settings.discord_guild_id)
            self.tree.copy_global_to(guild=guild)
            synced = await self.tree.sync(guild=guild)
            LOGGER.info("Synced %s slash commands to test guild %s", len(synced), guild.id)
        else:
            synced = await self.tree.sync()
            LOGGER.info("Synced %s global slash commands", len(synced))

    async def on_ready(self) -> None:
        LOGGER.info("Logged in as %s (%s)", self.user, self.user.id if self.user else "unknown")
        await self.agent.publish(
            AgentEvent(
                AgentEventKind.BOT_READY,
                payload={"bot_user_id": self.user.id if self.user else None},
            )
        )
        commands_cog = self.get_cog("AssistantCommands")
        if isinstance(commands_cog, AssistantCommands):
            await commands_cog.announce_version_if_needed()

    async def close(self) -> None:
        await self.agent.stop()
        if self.speech_router is not None:
            await self.speech_router.close()
        if self.capture_hub_server is not None:
            await self.capture_hub_server.close()
        if self.tool_gateway_server is not None:
            await self.tool_gateway_server.close()
        self.database.close()
        await super().close()


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = load_settings(project_root)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("discord.ext.voice_recv.reader").setLevel(logging.WARNING)
    logging.getLogger("discord.ext.voice_recv.gateway").setLevel(logging.WARNING)
    AssistantBot(settings).run(settings.discord_token, log_handler=None)


if __name__ == "__main__":
    main()
