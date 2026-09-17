from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import aiohttp
import discord
from discord import app_commands

if TYPE_CHECKING:
    from discord_ai_assistant.capture_hub import CaptureHubServer

DEFAULT_AGENT_URL = "http://127.0.0.1:8877"


async def _local_request(
    method: str,
    path: str,
    payload: dict[str, object] | None = None,
    *,
    timeout_seconds: float = 30.0,
) -> dict[str, object]:
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.request(method, f"{DEFAULT_AGENT_URL}{path}", json=payload) as response:
            text = await response.text()
            try:
                data = json.loads(text)
            except ValueError:
                data = {"message": text or f"Capture Agent HTTP {response.status}"}
            if response.status >= 400:
                raise RuntimeError(str(data.get("message") or f"Capture Agent HTTP {response.status}"))
            return data if isinstance(data, dict) else {"message": "Capture Agent 回傳格式不正確。"}


def _profile_text(profile: object) -> str:
    if not isinstance(profile, dict):
        return "未設定"
    roi = profile.get("pixel_roi") if isinstance(profile.get("pixel_roi"), dict) else {}
    return (
        f"{profile.get('name', '-')} | game={profile.get('game', '-')} | "
        f"ROI=({roi.get('x', '?')},{roi.get('y', '?')}) {roi.get('width', '?')}x{roi.get('height', '?')}"
    )


def install_capture_agent_commands(grouped: object, hub: "CaptureHubServer | None" = None) -> None:
    """Attach local/remote Capture Agent controls to the existing `/meeting` group."""
    meeting = getattr(grouped, "meeting")
    for name in ("agent", "agents", "pair", "select", "calibrate", "profiles", "profile", "ocr"):
        if meeting.get_command(name) is not None:
            meeting.remove_command(name)

    def _authorized(interaction: discord.Interaction) -> bool:
        checker = getattr(grouped.core, "_is_dj", None)
        return bool(checker and checker(interaction))

    async def _deny(interaction: discord.Interaction) -> None:
        if interaction.response.is_done():
            await interaction.followup.send("這個操作只有 DJ 或管理員可以使用。", ephemeral=True)
        else:
            await interaction.response.send_message("這個操作只有 DJ 或管理員可以使用。", ephemeral=True)

    async def _capture_request(
        interaction: discord.Interaction,
        action: str,
        payload: dict[str, object] | None = None,
        *,
        timeout_seconds: float = 60.0,
    ) -> tuple[dict[str, object], str]:
        guild_id = int(interaction.guild_id or 0)
        if hub is not None and guild_id and hub.selected_connection(guild_id) is not None:
            return await hub.request(guild_id, action, payload or {}, timeout_seconds=timeout_seconds), "remote"
        local_routes = {
            "health": ("GET", "/health"),
            "profiles": ("GET", "/profiles"),
            "select_profile": ("POST", "/profiles/select"),
            "calibrate": ("POST", "/calibrate"),
            "ocr_test": ("POST", "/ocr-test"),
        }
        method, path = local_routes[action]
        return await _local_request(method, path, payload, timeout_seconds=timeout_seconds), "local"

    async def agent_status(interaction: discord.Interaction) -> None:
        await interaction.response.defer(thinking=True, ephemeral=True)
        try:
            data, mode = await _capture_request(interaction, "health")
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as error:
            await interaction.edit_original_response(
                content=f"Capture Agent 尚未連線：{error}\n請確認本機 Agent 或已選取的遠端 Agent 在線。"
            )
            return
        screen = data.get("screen") if isinstance(data.get("screen"), dict) else {}
        await interaction.edit_original_response(
            content=(
                f"**Moxue Capture Agent ({'遠端' if mode == 'remote' else '本機'})**\n"
                f"裝置：{data.get('name', '-')}\n"
                f"Agent ID：`{data.get('agent_id', '-')}`\n"
                f"螢幕：{screen.get('width', '?')}x{screen.get('height', '?')}\n"
                f"Profiles：{data.get('profile_count', 0)}\n"
                f"目前：{_profile_text(data.get('active_profile'))}"
            )[:2000]
        )

    async def agents(interaction: discord.Interaction) -> None:
        await interaction.response.defer(thinking=True, ephemeral=True)
        if hub is None:
            await interaction.edit_original_response(content="Capture Hub 未啟用。")
            return
        data = hub.list_agents(int(interaction.guild_id or 0))
        lines = ["**Capture Agents**"]
        for item in data.get("agents", []):
            if not isinstance(item, dict):
                continue
            marker = "▶" if item.get("selected") else "•"
            screen = item.get("screen") if isinstance(item.get("screen"), dict) else {}
            lines.append(
                f"{marker} 🟢 `{item.get('agent_id')}` — {item.get('name')} "
                f"({screen.get('width', '?')}x{screen.get('height', '?')})"
            )
        pending = data.get("pending") if isinstance(data.get("pending"), list) else []
        for item in pending:
            if isinstance(item, dict):
                lines.append(f"• 🟡 {item.get('name')} — 待配對，請在該裝置查看 6 碼配對碼")
        if len(lines) == 1:
            lines.append("目前沒有任何 Capture Agent 連線。")
        await interaction.edit_original_response(content="\n".join(lines)[:2000])

    async def pair(interaction: discord.Interaction, code: str) -> None:
        if not _authorized(interaction):
            await _deny(interaction)
            return
        await interaction.response.defer(thinking=True, ephemeral=True)
        if hub is None:
            await interaction.edit_original_response(content="Capture Hub 未啟用。")
            return
        try:
            result = await hub.pair(code, int(interaction.guild_id or 0), int(interaction.user.id))
        except RuntimeError as error:
            await interaction.edit_original_response(content=f"配對失敗：{error}")
            return
        await interaction.edit_original_response(
            content=f"已配對並選取 **{result.get('name')}**\nAgent ID：`{result.get('agent_id')}`"
        )

    async def select_agent(interaction: discord.Interaction, agent_id: str) -> None:
        if not _authorized(interaction):
            await _deny(interaction)
            return
        await interaction.response.defer(thinking=True, ephemeral=True)
        if hub is None:
            await interaction.edit_original_response(content="Capture Hub 未啟用。")
            return
        try:
            result = await hub.select(int(interaction.guild_id or 0), agent_id.strip())
        except RuntimeError as error:
            await interaction.edit_original_response(content=f"切換 Agent 失敗：{error}")
            return
        await interaction.edit_original_response(
            content=f"目前 Capture Agent 已切換為 **{result.get('name')}** (`{result.get('agent_id')}`)."
        )

    async def calibrate(
        interaction: discord.Interaction,
        game: str = "game",
        profile_name: str | None = None,
        delay_seconds: app_commands.Range[int, 0, 15] = 5,
    ) -> None:
        if not _authorized(interaction):
            await _deny(interaction)
            return
        await interaction.response.defer(thinking=True, ephemeral=True)
        selected = hub.selected_connection(int(interaction.guild_id or 0)) if hub is not None else None
        target_name = selected.name if selected is not None else "本機"
        await interaction.edit_original_response(
            content=(
                f"目標裝置：**{target_name}**\n"
                f"{delay_seconds} 秒後會在該裝置擷取螢幕。請切回遊戲並讓聊天室顯示出來。\n"
                "擷取後會在遊戲裝置跳出凍結畫面；拖曳框選聊天室，Enter 儲存，Esc 取消。"
            )
        )
        try:
            data, mode = await _capture_request(
                interaction,
                "calibrate",
                {"game": game, "profile_name": profile_name, "delay_seconds": int(delay_seconds)},
                timeout_seconds=max(150.0, float(delay_seconds) + 120.0),
            )
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as error:
            await interaction.edit_original_response(content=f"校準失敗：{error}")
            return
        if data.get("cancelled"):
            await interaction.edit_original_response(content="已取消聊天室框選。")
            return
        await interaction.edit_original_response(
            content=(
                f"{data.get('message', '校準完成。')}\n"
                f"裝置：{target_name} ({'遠端' if mode == 'remote' else '本機'})\n"
                f"擷取方式：{data.get('backend', '-')}\n"
                f"Profile：{_profile_text(data.get('profile'))}\n"
                "接著可用 `/meeting ocr` 驗證。"
            )[:2000]
        )

    async def profiles(interaction: discord.Interaction) -> None:
        await interaction.response.defer(thinking=True, ephemeral=True)
        try:
            data, mode = await _capture_request(interaction, "profiles")
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as error:
            await interaction.edit_original_response(content=f"無法讀取 Capture Profiles：{error}")
            return
        active_id = data.get("active_profile_id")
        lines = [f"**Capture Profiles ({'遠端' if mode == 'remote' else '本機'}) — {data.get('name', '-')}**"]
        for item in data.get("profiles", []) if isinstance(data.get("profiles"), list) else []:
            if not isinstance(item, dict):
                continue
            marker = "▶" if item.get("id") == active_id else "•"
            lines.append(f"{marker} `{item.get('id', '-')}` — {_profile_text(item)}")
        if len(lines) == 1:
            lines.append("尚未建立 Profile；請使用 `/meeting calibrate`。")
        await interaction.edit_original_response(content="\n".join(lines)[:2000])

    async def select_profile(interaction: discord.Interaction, profile_id: str) -> None:
        if not _authorized(interaction):
            await _deny(interaction)
            return
        await interaction.response.defer(thinking=True, ephemeral=True)
        try:
            data, mode = await _capture_request(interaction, "select_profile", {"profile_id": profile_id})
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as error:
            await interaction.edit_original_response(content=f"切換 Profile 失敗：{error}")
            return
        await interaction.edit_original_response(
            content=f"{data.get('message', '已切換。')} ({'遠端' if mode == 'remote' else '本機'})\n{_profile_text(data.get('profile'))}"[:2000]
        )

    async def ocr(interaction: discord.Interaction) -> None:
        if not _authorized(interaction):
            await _deny(interaction)
            return
        await interaction.response.defer(thinking=True, ephemeral=True)
        try:
            data, mode = await _capture_request(interaction, "ocr_test", timeout_seconds=90.0)
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as error:
            await interaction.edit_original_response(content=f"OCR 測試失敗：{error}")
            return
        lines = data.get("lines") if isinstance(data.get("lines"), list) else []
        preview = "\n".join(str(item) for item in lines[:15]) or "沒有辨識到文字。"
        await interaction.edit_original_response(
            content=(
                f"{data.get('message', 'OCR 測試完成。')} ({'遠端' if mode == 'remote' else '本機'})\n"
                f"Profile：{_profile_text(data.get('profile'))}\n\n{preview}"
            )[:2000]
        )

    calibrate = app_commands.describe(
        game="遊戲名稱，例如 LifeAfter",
        profile_name="選填，例如 筆電 / 2K-UI100",
        delay_seconds="送出指令後幾秒擷取畫面，方便切回全螢幕遊戲",
    )(calibrate)
    pair = app_commands.describe(code="筆電 MoxueCapture 顯示的 6 碼配對碼")(pair)
    select_agent = app_commands.describe(agent_id="使用 /meeting agents 顯示的 Agent ID")(select_agent)
    select_profile = app_commands.describe(profile_id="使用 /meeting profiles 顯示的 Profile ID")(select_profile)

    meeting.add_command(app_commands.Command(name="agent", description="查看目前使用的 Capture Agent", callback=agent_status))
    meeting.add_command(app_commands.Command(name="agents", description="列出已配對與等待配對的 Capture Agents", callback=agents))
    meeting.add_command(app_commands.Command(name="pair", description="用 6 碼配對碼授權一台遠端 Capture Agent", callback=pair))
    meeting.add_command(app_commands.Command(name="select", description="選擇本伺服器要控制的遠端 Capture Agent", callback=select_agent))
    meeting.add_command(app_commands.Command(name="calibrate", description="在目前 Agent 凍結遊戲畫面並框選聊天室 ROI", callback=calibrate))
    meeting.add_command(app_commands.Command(name="profiles", description="列出目前 Agent 保存的 Capture Profiles", callback=profiles))
    meeting.add_command(app_commands.Command(name="profile", description="切換目前 Agent 的 Capture Profile", callback=select_profile))
    meeting.add_command(app_commands.Command(name="ocr", description="在目前 Agent 執行一次聊天室 OCR 測試", callback=ocr))
