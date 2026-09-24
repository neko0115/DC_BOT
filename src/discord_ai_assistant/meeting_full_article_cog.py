from __future__ import annotations

import asyncio
import difflib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import discord
from discord import app_commands
from discord.ext import commands

import discord_ai_assistant.meeting_feedback_cog as feedback_module
from discord_ai_assistant.ai.gemini import GeminiRequestError
from discord_ai_assistant.ai.persona import BASE_PERSONA_INSTRUCTION
from discord_ai_assistant.meeting_feedback import (
    FeedbackCandidate,
    infer_partial_asr_candidates,
    merge_candidates,
    parse_ai_candidates,
)
from discord_ai_assistant.meeting_feedback_cog import (
    MeetingReportCommands as FeedbackMeetingReportCommands,
    MeetingReportReviewView as LegacyMeetingReportReviewView,
    _candidate_label,
)
from discord_ai_assistant.meeting_upload import (
    build_meeting_report_prompt,
    normalize_report_timecode_particle_order,
)


LOGGER = logging.getLogger(__name__)
SUPPORTED_REVIEW_EXTENSIONS = {".md", ".txt"}
MAX_REVIEW_BYTES = 2 * 1024 * 1024
MAX_REVIEW_CHARACTERS = 50_000
MIN_REVIEW_CHARACTERS = 250
PENDING_REVIEW_SECONDS = 600.0
MEETING_REPORT_TIMEOUT_SECONDS = 240


@dataclass(slots=True)
class PendingFullArticleRevision:
    report_path: Path
    review_path: Path
    title: str
    created_at_monotonic: float

    def expired(self, now: float | None = None) -> bool:
        moment = time.monotonic() if now is None else now
        return moment - self.created_at_monotonic > PENDING_REVIEW_SECONDS


def normalize_full_revision_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(normalized) < MIN_REVIEW_CHARACTERS:
        raise ValueError("完整修正版內容太短，請貼上整篇週會報或上傳完整 .md/.txt。")
    if len(normalized) > MAX_REVIEW_CHARACTERS:
        raise ValueError(f"完整修正版超過 {MAX_REVIEW_CHARACTERS:,} 字元，請縮短後再提交。")
    return normalized


def full_revision_diff(original_report: str, revised_report: str, limit: int = 14_000) -> str:
    lines = difflib.unified_diff(
        original_report.splitlines(),
        revised_report.splitlines(),
        fromfile="原草稿",
        tofile="人工完整修正版",
        lineterm="",
        n=2,
    )
    text = "\n".join(lines)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...（diff 已截斷）"


class FullArticleMeetingReportReviewView(LegacyMeetingReportReviewView):
    """Replace the 4k modal review with a full-article upload flow."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        old_button: discord.ui.Button | None = None
        for child in self.children:
            if isinstance(child, discord.ui.Button) and child.label == "修正並重新整理":
                old_button = child
                break
        if old_button is not None:
            self.remove_item(old_button)

        button = discord.ui.Button(
            label="提交整篇修正版",
            style=discord.ButtonStyle.primary,
            emoji="✏️",
            custom_id="meeting_full_article_revision",
        )
        button.callback = self._begin_full_article_revision
        self.add_item(button)

    async def _begin_full_article_revision(self, interaction: discord.Interaction) -> None:
        if not await self._guard(interaction):
            return
        await self.cog.begin_full_article_revision(
            interaction,
            report_path=self.report_path,
            review_path=self.review_path,
            title=self.title,
        )


class ResilientMeetingFeedbackDecisionView(discord.ui.View):
    """Keep a meeting revision retryable when Gemini fails after human approval."""

    def __init__(
        self,
        cog: "MeetingReportCommands",
        *,
        revision_id: int,
        review_path: Path,
        report_path: Path,
        title: str,
        correction_text: str,
        candidate_ids: list[int],
    ) -> None:
        super().__init__(timeout=1800)
        self.cog = cog
        self.revision_id = revision_id
        self.review_path = review_path
        self.report_path = report_path
        self.title = title
        self.correction_text = correction_text
        self.candidate_ids = candidate_ids[:12]
        self.selected_candidate_ids: set[int] = set()
        self._learning_decision: bool | None = None
        self._lock = asyncio.Lock()
        self._handled = False
        if self.candidate_ids:
            store = getattr(cog, "feedback_store", None)
            candidates = dict(store.candidates(revision_id)) if store is not None else {}
            options = []
            for index, item_id in enumerate(self.candidate_ids, start=1):
                candidate = candidates.get(item_id)
                label = _candidate_label(candidate) if candidate is not None else f"候選 #{item_id}"
                options.append(discord.SelectOption(label=f"{index}. {label}"[:100], value=str(item_id)))
            selector = discord.ui.Select(
                placeholder="選擇允許學習的項目（可複選）",
                min_values=0,
                max_values=len(self.candidate_ids),
                custom_id="meeting_feedback_candidates",
                row=1,
                options=options,
            )
            selector.callback = self._select_candidates
            self.add_item(selector)
        self._restore_retry_state()

    def _disable(self) -> None:
        self._handled = True
        for item in self.children:
            if isinstance(item, (discord.ui.Button, discord.ui.Select)):
                item.disabled = True

    def _restore_retry_state(self) -> None:
        self._handled = False
        for item in self.children:
            if isinstance(item, discord.ui.Select):
                item.disabled = self._learning_decision is not None
                for option in item.options:
                    option.default = int(option.value) in self.selected_candidate_ids
            if not isinstance(item, discord.ui.Button):
                continue
            item.disabled = bool(
                (
                    item.custom_id == "meeting_feedback_apply"
                    and (not self.selected_candidate_ids or self._learning_decision is False)
                )
                or (item.custom_id == "meeting_feedback_regenerate_only" and self._learning_decision is True)
            )

    async def _select_candidates(self, interaction: discord.Interaction) -> None:
        if not await self._guard(interaction):
            return
        async with self._lock:
            if not await self._guard(interaction):
                return
            if self._learning_decision is not None:
                await interaction.response.send_message("學習選擇已保存，重試會沿用原選擇。", ephemeral=True)
                return
            selector = next(item for item in self.children if isinstance(item, discord.ui.Select))
            self.selected_candidate_ids = {
                item_id for item_id in self.candidate_ids if str(item_id) in selector.values
            }
            self._restore_retry_state()
            await interaction.response.edit_message(view=self)

    async def _guard(self, interaction: discord.Interaction) -> bool:
        if not interaction.guild or not isinstance(interaction.user, discord.Member):
            await interaction.response.send_message("只能在原伺服器處理修正。", ephemeral=True)
            return False
        if not self.cog._is_dj_member(interaction.user):
            await interaction.response.send_message("只有 DJ 或管理員可以套用週會修正。", ephemeral=True)
            return False
        if self._handled:
            await interaction.response.send_message("這份修正正在處理中，請稍候。", ephemeral=True)
            return False
        return True

    async def _sync_view(self, interaction: discord.Interaction) -> None:
        if interaction.message is None:
            return
        try:
            await interaction.message.edit(view=self)
        except discord.HTTPException:
            LOGGER.warning("Could not refresh meeting feedback buttons", exc_info=True)

    async def _run_regeneration(self, interaction: discord.Interaction, *, learn: bool) -> None:
        if not await self._guard(interaction):
            return
        async with self._lock:
            if not await self._guard(interaction):
                return
            if learn and not self.selected_candidate_ids:
                await interaction.response.send_message("請先選擇至少一個允許學習的項目。", ephemeral=True)
                return
            if self._learning_decision is not None and learn != self._learning_decision:
                await interaction.response.send_message("重試必須沿用已保存的學習選擇。", ephemeral=True)
                return
            self._learning_decision = learn
            self._disable()
            await interaction.response.defer(thinking=True, ephemeral=True)
            await self._sync_view(interaction)
            try:
                await self.cog.regenerate_from_feedback(
                    interaction,
                    revision_id=self.revision_id,
                    review_path=self.review_path,
                    report_path=self.report_path,
                    title=self.title,
                    correction_text=self.correction_text,
                    learn=learn,
                    selected_candidate_ids=sorted(self.selected_candidate_ids) if learn else [],
                )
            except Exception as error:
                LOGGER.exception(
                    "Meeting revision regeneration failed for revision %s", self.revision_id
                )
                if self.cog.feedback_store is not None:
                    self.cog.feedback_store.update_revision(
                        self.revision_id,
                        status=(
                            "generation_failed_after_learning"
                            if learn
                            else "generation_failed_without_learning"
                        ),
                    )
                self._restore_retry_state()
                await self._sync_view(interaction)
                if isinstance(error, GeminiRequestError):
                    detail = str(error)
                else:
                    detail = "重新整理週會報時發生錯誤，原始資料與人工修正版都仍然保留。"
                await interaction.followup.send(
                    f"{detail}\n\n這份 revision **沒有作廢**，你可以直接再按一次重整；"
                    "如果 Bot 已重新啟動，也可以用 `/meeting_revision_retry` 救回最近一次未完成修訂。",
                    ephemeral=True,
                )
                return
            await self._sync_view(interaction)

    @discord.ui.button(
        label="套用已選項目並重整",
        style=discord.ButtonStyle.success,
        emoji="🧠",
        custom_id="meeting_feedback_apply",
    )
    async def apply_and_regenerate(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        del button
        await self._run_regeneration(interaction, learn=True)

    @discord.ui.button(
        label="只重整，不學習",
        style=discord.ButtonStyle.primary,
        emoji="🔄",
        custom_id="meeting_feedback_regenerate_only",
    )
    async def regenerate_only(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        del button
        await self._run_regeneration(interaction, learn=False)

    @discord.ui.button(
        label="取消",
        style=discord.ButtonStyle.secondary,
        emoji="✖️",
        custom_id="meeting_feedback_cancel",
    )
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        del button
        if not await self._guard(interaction):
            return
        self._disable()
        if self.cog.feedback_store is not None:
            self.cog.feedback_store.update_revision(self.revision_id, status="cancelled")
            unresolved = [
                item_id
                for status in ("pending", "approved")
                for item_id, _ in self.cog.feedback_store.candidates(self.revision_id, status=status)
            ]
            self.cog.feedback_store.set_candidate_status(unresolved, "cancelled")
        await interaction.response.edit_message(
            content="已取消這次重整，原草稿保持不變；若先前已完成學習，已套用項目仍會保留。", view=self,
        )


# The base feedback cog creates MeetingReportReviewView from its own module globals
# both for the first draft and later regenerated drafts. Replace only that view class
# so every review round consistently uses the full-article workflow.
feedback_module.MeetingReportReviewView = FullArticleMeetingReportReviewView


class MeetingReportCommands(FeedbackMeetingReportCommands):
    def __init__(self, bot: commands.Bot, settings: Any, ai: Any) -> None:
        super().__init__(bot, settings, ai)
        self._pending_full_article: dict[tuple[int, int, int], PendingFullArticleRevision] = {}

    @staticmethod
    def _pending_key(guild_id: int, user_id: int, channel_id: int) -> tuple[int, int, int]:
        return guild_id, user_id, channel_id

    async def _generate_report_with_feedback(
        self,
        *,
        title: str,
        transcript: str,
        guild_id: int,
        snapshot: Any | None,
        correction_text: str | None = None,
        previous_report: str | None = None,
    ) -> str:
        """Generate meeting reports with a task-specific 240 second Gemini timeout."""
        prompt = build_meeting_report_prompt(title, transcript)
        if snapshot is not None:
            prompt += (
                "\n\n---\n領域專有名詞資料庫：\n"
                f"{snapshot.explanation_block()}\n"
                "重要：上述詞庫只可協助理解詞義與修正名詞，不可把詞庫內的解釋當成本次會議發生的事、決議或進度。\n"
            )
        profile_key = snapshot.profile.key if snapshot is not None else None
        preferences, examples = self._feedback_context(guild_id, profile_key)
        if preferences:
            prompt += (
                "\n\n---\n已由人工核准的穩定報告偏好：\n"
                + "\n".join(f"- {item}" for item in preferences)
                + "\n這些偏好只控制整理方式，不是會議事實。\n"
            )
        if examples:
            prompt += "\n\n---\n以下為近期人工核准報告的格式範例，只可參考結構與寫法，絕對不可沿用其中的事實：\n"
            for index, example in enumerate(examples, start=1):
                prompt += f"\n[範例 {index}]\n{example[:4000]}\n"
        if correction_text:
            prompt += (
                "\n\n---\n本次人工審核提出的修正要求：\n"
                f"{correction_text.strip()}\n"
                "這些修正要求優先於原草稿；但仍不得新增逐字稿與修正要求都沒有支持的事實。\n"
            )
        if previous_report:
            prompt += (
                "\n\n---\n上一版草稿如下，請只把它當成待修正草稿，不可把草稿中沒有逐字稿支持的內容當成事實：\n"
                f"{previous_report[:12000]}\n"
            )
        persona_instruction = (
            f"{BASE_PERSONA_INSTRUCTION}\n"
            "你現在執行會議紀錄整理，不可呼叫工具、搜尋外部資訊或使用逐字稿、人工修正、明示詞庫以外的事實。"
        )
        long_reply = getattr(self.ai, "social_reply_with_timeout", None)
        if callable(long_reply):
            text = await long_reply(
                prompt,
                persona_instruction=persona_instruction,
                timeout_seconds=MEETING_REPORT_TIMEOUT_SECONDS,
                request_kind="meeting-report",
            )
        else:
            LOGGER.warning(
                "AI backend does not support task-specific meeting timeout; using normal social timeout"
            )
            text = await self.ai.social_reply(
                prompt,
                persona_instruction=persona_instruction,
            )
        result = normalize_report_timecode_particle_order(str(text).strip())
        if not result:
            raise RuntimeError("Gemini 沒有回傳週會報內容。")
        return result

    async def begin_full_article_revision(
        self,
        interaction: discord.Interaction,
        *,
        report_path: Path,
        review_path: Path,
        title: str,
    ) -> None:
        if not interaction.guild or not isinstance(interaction.user, discord.Member) or interaction.channel_id is None:
            await interaction.response.send_message("只能在原伺服器頻道提交完整修正版。", ephemeral=True)
            return
        if not self._is_dj_member(interaction.user):
            await interaction.response.send_message("只有 DJ 或管理員可以提交完整修正版。", ephemeral=True)
            return
        key = self._pending_key(interaction.guild.id, interaction.user.id, interaction.channel_id)
        self._pending_full_article[key] = PendingFullArticleRevision(
            report_path=report_path,
            review_path=review_path,
            title=title,
            created_at_monotonic=time.monotonic(),
        )
        review = self._read_review(review_path)
        review_id = str(review.get("review_id") or review_path.stem)
        await interaction.response.send_message(
            "我會在這個頻道等你 **10 分鐘**。請直接上傳你已經人工修改完成的整篇 `.md` / `.txt` 週會報；"
            "如果全文沒有超過 Discord 訊息限制，也可以直接貼整篇文字。\n\n"
            f"這次審核編號是 `{review_id}`，但你**不用記也不用輸入它**。下一則符合條件的完整修正版會自動綁定這份草稿。\n"
            "請不要 @墨雪；直接貼全文或附檔即可，避免一般聊天流程另外回覆。",
            ephemeral=True,
        )

    async def _extract_full_revision(self, message: discord.Message) -> str | None:
        attachment = next(
            (
                item
                for item in message.attachments
                if Path(item.filename).suffix.lower() in SUPPORTED_REVIEW_EXTENSIONS
            ),
            None,
        )
        if attachment is not None:
            if attachment.size <= 0 or attachment.size > MAX_REVIEW_BYTES:
                raise ValueError("修正版附件必須介於 1 byte 與 2 MiB 之間。")
            payload = await attachment.read()
            try:
                text = payload.decode("utf-8-sig")
            except UnicodeDecodeError as error:
                raise ValueError("修正版附件必須是 UTF-8 編碼的 .md 或 .txt。") from error
            return normalize_full_revision_text(text)

        content = message.content.strip()
        if len(content) >= MIN_REVIEW_CHARACTERS:
            return normalize_full_revision_text(content)
        return None

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot or message.guild is None:
            return
        if not isinstance(message.author, discord.Member):
            return
        key = self._pending_key(message.guild.id, message.author.id, message.channel.id)
        pending = self._pending_full_article.get(key)
        if pending is None:
            return
        if pending.expired():
            self._pending_full_article.pop(key, None)
            await message.reply(
                "剛才的完整週會報修正等待已逾時，請回到草稿再按一次「提交整篇修正版」。",
                mention_author=False,
            )
            return
        if not self._is_dj_member(message.author):
            self._pending_full_article.pop(key, None)
            return

        try:
            revised_article = await self._extract_full_revision(message)
        except ValueError as error:
            await message.reply(str(error), mention_author=False)
            return
        except discord.HTTPException:
            LOGGER.exception("Could not download full meeting revision attachment")
            await message.reply("讀取修正版附件失敗，請重新上傳一次。", mention_author=False)
            return

        if revised_article is None:
            return

        self._pending_full_article.pop(key, None)
        await message.reply(
            "收到整篇人工修正版。我正在比對「原逐字稿 ↔ 原草稿 ↔ 你的完整修正版」，先找出值得長期學習的候選。",
            mention_author=False,
        )
        try:
            await self.handle_full_article_submission(message, pending, revised_article)
        except Exception:
            LOGGER.exception("Full-article meeting review failed")
            await message.reply("完整修正版處理失敗；原草稿與你的附件都沒有被覆蓋。請查看 Bot log。", mention_author=False)

    async def handle_full_article_submission(
        self,
        message: discord.Message,
        pending: PendingFullArticleRevision,
        revised_article: str,
    ) -> None:
        if message.guild is None or not isinstance(message.author, discord.Member):
            return
        if self.feedback_store is None:
            await message.reply("目前無法取得 Feedback database。", mention_author=False)
            return

        review = self._read_review(pending.review_path)
        audio_sha = str(review.get("audio_sha256") or pending.review_path.parent.name)
        profile_key = str(review.get("knowledge_profile")) if review.get("knowledge_profile") else None
        review_id = str(review.get("review_id") or pending.review_path.stem)
        original_report = pending.report_path.read_text(encoding="utf-8")

        revision_id = self.feedback_store.create_revision(
            review_id=review_id,
            guild_id=message.guild.id,
            profile_key=profile_key,
            audio_sha256=audio_sha,
            title=pending.title,
            correction_text=revised_article,
            original_report=original_report,
            created_by=message.author.id,
        )
        human_file = pending.review_path.parent / f"human_revision_{revision_id}.md"
        human_file.write_text(revised_article.rstrip() + "\n", encoding="utf-8")

        candidates = await self._propose_learning_candidates(
            correction_text=revised_article,
            report=original_report,
            transcript_path=pending.review_path.parent / "transcript.txt",
        )
        candidate_ids = self.feedback_store.add_candidates(revision_id, candidates)
        self._update_review(
            pending.review_path,
            status="revision_pending",
            pending_revision_id=revision_id,
            pending_revision_by=message.author.id,
            human_revision_file=human_file.name,
        )

        if candidates:
            lines = [f"{index}. {_candidate_label(item)}" for index, item in enumerate(candidates, start=1)]
            candidate_text = "\n".join(lines)
            content = (
                "我已經從你的**整篇修正版**找出以下可重用學習候選；目前仍然**沒有**寫進 Knowledge DB：\n"
                f"{candidate_text[:1500]}\n\n"
                "請在下方逐項選擇允許學習的候選，再按「套用已選項目並重整」；未選項目不會學習。"
                "如果這次只想照你的全文重新整理，選「只重整，不學習」。"
            )
        else:
            content = (
                "我已保存你的整篇人工修正版，但沒有找到足夠明確、適合長期保存的 ASR／詞義／格式學習候選。\n"
                "你仍可選「只重整，不學習」，讓墨雪以你的完整修正版作為本次最高優先級人工指導重新製作。"
            )

        await message.channel.send(
            content=content,
            file=discord.File(human_file, filename=f"{Path(pending.title).stem or 'meeting'}_人工完整修正版_{revision_id}.md"),
            view=ResilientMeetingFeedbackDecisionView(
                self,
                revision_id=revision_id,
                review_path=pending.review_path,
                report_path=pending.report_path,
                title=pending.title,
                correction_text=revised_article,
                candidate_ids=candidate_ids,
            ),
            allowed_mentions=discord.AllowedMentions.none(),
        )

    async def _propose_learning_candidates(
        self, *, correction_text: str, report: str, transcript_path: Path
    ) -> list[FeedbackCandidate]:
        try:
            transcript = transcript_path.read_text(encoding="utf-8")[:16_000]
        except OSError:
            transcript = ""
        diff = full_revision_diff(report, correction_text)
        prompt = (
            "你正在比較『AI 原週會報』與『人類人工修改完成的整篇週會報』。"
            "目標不是重寫文章，而是找出未來可重用的學習候選。\n"
            "只允許三種 kind：\n"
            "- asr_alias：人工完整版清楚修正了 Whisper 誤辨識，source_text 是錯詞，canonical_term 是正確標準詞。\n"
            "- domain_term：人工完整版清楚補足一個專有名詞的可重用簡短解釋。\n"
            "- report_preference：人工修改反映未來同 Profile 都應遵守的穩定格式／寫法偏好。\n"
            "禁止把本次會議的一次性事件、日期、數字、負責人、Boss 安排、戰術決議當成長期學習。\n"
            "只有從『原草稿 ↔ 人工完整修正版』差異能明確支持的項目才能提出；不確定就省略。"
            "若同一錯詞在原稿出現多次，只要人工修正版至少明確修正其中一次，就仍應提出 asr_alias 候選；"
            "不要因為其他相同出現位置尚未全部修改而省略，是否學習由後續人工勾選決定。"
            "不得使用外部知識、記憶或自行猜測。\n"
            "只回傳 JSON，不要 Markdown：\n"
            '{"candidates":[{"kind":"asr_alias|domain_term|report_preference","source_text":"",'
            '"canonical_term":"","explanation":"","aliases":[],"confidence":0.0}]}\n\n'
            f"差異 diff：\n{diff}\n\n"
            f"AI 原草稿：\n{report[:9000]}\n\n"
            f"人類完整修正版：\n{correction_text[:14000]}\n\n"
            f"原始逐字稿節錄：\n{transcript}\n"
        )
        try:
            response = await self.ai.social_reply(
                prompt,
                persona_instruction=(
                    "只做人工修正版差異分類，不可呼叫工具、搜尋、使用記憶或補充外部事實。"
                    "輸出必須是單一 JSON object。"
                ),
            )
        except Exception:
            LOGGER.warning("Could not propose full-article meeting learning candidates", exc_info=True)
            return []
        inferred = infer_partial_asr_candidates(report, correction_text, transcript)
        return merge_candidates(inferred, parse_ai_candidates(str(response)))

    def _latest_recoverable_revision(
        self, guild_id: int, user_id: int
    ) -> tuple[dict[str, object], bool] | None:
        if self.feedback_store is None:
            return None
        rows = self.feedback_store.connection.execute(
            """SELECT * FROM meeting_feedback_revisions
               WHERE guild_id = ? AND created_by = ? AND revised_report = ''
               ORDER BY id DESC LIMIT 20""",
            (guild_id, user_id),
        ).fetchall()
        for row in rows:
            payload = {key: row[key] for key in row.keys()}
            revision_id = int(payload["id"])
            status = str(payload.get("status") or "")
            if status == "cancelled":
                continue
            if status == "generation_failed_after_learning":
                return payload, True
            if status == "generation_failed_without_learning":
                return payload, False
            applied = self.feedback_store.candidates(revision_id, status="applied")
            approved = self.feedback_store.candidates(revision_id, status="approved")
            skipped = self.feedback_store.candidates(revision_id, status="skipped")
            pending = self.feedback_store.candidates(revision_id, status="pending")
            if applied or approved:
                return payload, True
            if skipped and not pending:
                return payload, False
        return None

    def _review_path_for_revision(self, audio_sha: str, review_id: str) -> Path | None:
        job_root = self.import_root / audio_sha
        if not job_root.exists():
            return None
        for path in sorted(job_root.glob("review_*.json"), reverse=True):
            review = self._read_review(path)
            if str(review.get("review_id") or path.stem) == review_id:
                return path
        return None

    @app_commands.command(
        name="meeting_revision_retry",
        description="重試最近一次因 AI 逾時或失敗而未完成的週會人工修訂",
    )
    async def meeting_revision_retry(self, interaction: discord.Interaction) -> None:
        if not interaction.guild or not isinstance(interaction.user, discord.Member):
            await interaction.response.send_message("只能在伺服器內重試週會修訂。", ephemeral=True)
            return
        if not self._is_dj_member(interaction.user) or self.feedback_store is None:
            await interaction.response.send_message("只有 DJ／管理員可以重試週會修訂。", ephemeral=True)
            return

        recoverable = self._latest_recoverable_revision(interaction.guild.id, interaction.user.id)
        if recoverable is None:
            await interaction.response.send_message(
                "找不到你最近可安全重試的未完成週會修訂。",
                ephemeral=True,
            )
            return
        revision, learn = recoverable
        revision_id = int(revision["id"])
        audio_sha = str(revision["audio_sha256"])
        review_id = str(revision["review_id"])
        review_path = self._review_path_for_revision(audio_sha, review_id)
        if review_path is None:
            await interaction.response.send_message(
                "找到 Feedback revision，但找不到對應 review JSON；沒有重新執行任何動作。",
                ephemeral=True,
            )
            return
        review = self._read_review(review_path)
        report_name = str(review.get("report_file") or "")
        report_path = review_path.parent / report_name if report_name else Path()
        if not report_name or not report_path.exists():
            report_path = review_path.parent / f"recovery_original_report_{revision_id}.md"
            report_path.write_text(str(revision["original_report"]), encoding="utf-8")

        await interaction.response.defer(thinking=True, ephemeral=True)
        try:
            await self.regenerate_from_feedback(
                interaction,
                revision_id=revision_id,
                review_path=review_path,
                report_path=report_path,
                title=str(revision["title"]),
                correction_text=str(revision["correction_text"]),
                learn=learn,
            )
        except Exception as error:
            LOGGER.exception("Meeting revision recovery failed for revision %s", revision_id)
            self.feedback_store.update_revision(
                revision_id,
                status=(
                    "generation_failed_after_learning"
                    if learn
                    else "generation_failed_without_learning"
                ),
            )
            detail = str(error) if isinstance(error, GeminiRequestError) else "週會修訂重試仍然失敗，資料沒有被刪除。"
            await interaction.followup.send(
                f"{detail}\n你可以稍後再次執行 `/meeting_revision_retry`。",
                ephemeral=True,
            )
