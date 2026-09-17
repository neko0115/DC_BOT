from __future__ import annotations

import asyncio
import logging
import time

import discord
from discord.ext import commands

from discord_ai_assistant.ai.knowledge_help import KNOWLEDGE_HELP_WAIT_SECONDS, KnowledgeHelpState
from discord_ai_assistant.ai.social import SocialParticipant

LOGGER = logging.getLogger(__name__)


class KnowledgeHelpRuntime(commands.Cog):
    """Fallback delayed human-first knowledge-help runtime.

    Current production uses ``ToolEffectAssistantCommands``, which already owns the
    original delayed knowledge-help state machine. This cog remains as a compatibility
    fallback for command stacks that do not provide that native runtime. It must never
    schedule a second answer when the active AssistantCommands cog already exposes
    ``_schedule_knowledge_help``.
    """

    def __init__(
        self,
        bot: commands.Bot,
        social: SocialParticipant,
        *,
        wait_seconds: float = KNOWLEDGE_HELP_WAIT_SECONDS,
        state: KnowledgeHelpState | None = None,
    ) -> None:
        self.bot = bot
        self.social = social
        self.wait_seconds = max(0.0, float(wait_seconds))
        self.state = state or KnowledgeHelpState()
        self._tasks: set[asyncio.Task[None]] = set()

    def cog_unload(self) -> None:
        for task in tuple(self._tasks):
            task.cancel()
        self._tasks.clear()

    def _native_runtime_present(self) -> bool:
        core = self.bot.get_cog("AssistantCommands")
        return core is not None and callable(getattr(core, "_schedule_knowledge_help", None))

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        # ToolEffectAssistantCommands already processes the same Discord event and owns the
        # authoritative KnowledgeHelpState. Running both paths would duplicate Gemini/Search
        # requests and replies, so this compatibility cog becomes a no-op in that setup.
        if self._native_runtime_present():
            return

        bot_user = self.bot.user
        if bot_user is None or message.author.bot or message.guild is None:
            return

        key = (message.guild.id, message.channel.id)
        mentions_bot = bot_user.id in message.raw_mentions or any(
            member.id == bot_user.id for member in message.mentions
        )
        reference = message.reference
        reply_to_message_id = reference.message_id if reference and reference.message_id else None
        now = time.monotonic()

        cancelled = self.state.observe_human(
            key,
            author_id=message.author.id,
            content=message.clean_content,
            now=now,
            reply_to_message_id=reply_to_message_id,
            mentions_bot=mentions_bot,
        )
        if cancelled:
            LOGGER.info(
                "Cancelled pending knowledge help in guild %s channel %s because a human spoke first",
                message.guild.id,
                message.channel.id,
            )

        # Explicit mentions are handled immediately by AssistantCommands and also count as
        # acknowledgement of an earlier proactive response. Never create a delayed duplicate.
        if mentions_bot:
            return
        if not self.social.can_offer_knowledge_help(message):
            return
        if not self.state.can_schedule(key, now):
            return

        self.state.start_pending(
            key,
            author_id=message.author.id,
            source_message_id=message.id,
        )
        LOGGER.info(
            "Scheduled fallback proactive knowledge help in guild %s channel %s after %.1f seconds",
            message.guild.id,
            message.channel.id,
            self.wait_seconds,
        )
        task = asyncio.create_task(self._deliver_after_delay(message, key))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _deliver_after_delay(self, message: discord.Message, key: tuple[int, int]) -> None:
        try:
            await asyncio.sleep(self.wait_seconds)
            if not self.state.pending_matches(key, message.id):
                return

            if not self.social.can_offer_knowledge_help(message):
                self.state.clear_pending(key, message.id)
                return

            reply = await self.social.knowledge_help(message)
            if not self.state.pending_matches(key, message.id):
                return
            if not reply:
                self.state.clear_pending(key, message.id)
                return

            reference = message.to_reference(fail_if_not_exists=False)
            sent = await message.channel.send(
                reply[:2000],
                reference=reference,
                mention_author=False,
            )
            self.state.record_sent(key, response_message_id=sent.id, now=time.monotonic())
        except asyncio.CancelledError:
            self.state.clear_pending(key, message.id)
            raise
        except Exception:
            self.state.clear_pending(key, message.id)
            LOGGER.exception(
                "Fallback proactive knowledge help runtime failed in guild %s channel %s",
                message.guild.id if message.guild else 0,
                message.channel.id,
            )
