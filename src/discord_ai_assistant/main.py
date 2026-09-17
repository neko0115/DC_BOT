from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import discord
from discord.ext import commands

from discord_ai_assistant.ai.gemini import GeminiAssistant
from discord_ai_assistant.ai.tools import ToolRouter
from discord_ai_assistant.commands import AssistantCommands
from discord_ai_assistant.config import Settings, load_settings
from discord_ai_assistant.music.library import LibraryService
from discord_ai_assistant.music.player import MusicManager, configure_event_loop
from discord_ai_assistant.storage.database import Database
from discord_ai_assistant.tool_gateway.client import ToolGatewayClient
from discord_ai_assistant.tool_gateway.server import ToolGatewayServer

LOGGER = logging.getLogger(__name__)


class AssistantBot(commands.Bot):
    def __init__(self, settings: Settings) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents)
        self.settings = settings
        self.database = Database(settings.database_path)
        self.library = LibraryService(settings.library_path, self.database, settings.max_upload_bytes)
        self.music = MusicManager(
            settings.library_path,
            self.library.random_track,
            youtube_cookies_from_browser=settings.youtube_cookies_from_browser,
            youtube_cookies_file=settings.youtube_cookies_file,
            youtube_po_token=settings.youtube_po_token,
        )

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

        self.tool_router = ToolRouter(self.library, self.music, self.database, external_client)
        self.ai = GeminiAssistant(
            settings.gemini_api_key,
            settings.gemini_model,
            self.tool_router,
            settings.persona_timezone,
        )

    async def setup_hook(self) -> None:
        configure_event_loop(asyncio.get_running_loop())
        if self.tool_gateway_server is not None:
            await self.tool_gateway_server.start()
            await self.tool_router.refresh_external_tools()
        await self.add_cog(
            AssistantCommands(self, self.settings, self.database, self.library, self.music, self.ai)
        )
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
        commands_cog = self.get_cog("AssistantCommands")
        if isinstance(commands_cog, AssistantCommands):
            await commands_cog.announce_version_if_needed()

    async def close(self) -> None:
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
