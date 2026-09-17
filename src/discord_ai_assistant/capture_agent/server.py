from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from aiohttp import web
from PIL import Image, ImageOps

from .calibration import calibrate_profile
from .capture import capture_screen, crop_profile, virtual_screen_bounds
from .profiles import DATA_ROOT, CaptureProfile, ProfileStore

LOGGER = logging.getLogger(__name__)
CALIBRATION_ROOT = DATA_ROOT / "calibration"


class CaptureAgentServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8877) -> None:
        self.host = host
        self.port = port
        self.store = ProfileStore()
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._sync_task: asyncio.Task[None] | None = None
        self._calibration_lock = asyncio.Lock()
        self._ocr_engine: Any | None = None

    async def start(self) -> None:
        app = web.Application()
        app.add_routes(
            [
                web.get("/health", self._health),
                web.get("/profiles", self._profiles),
                web.post("/profiles/select", self._select_profile),
                web.post("/calibrate", self._calibrate),
                web.post("/ocr-test", self._ocr_test),
            ]
        )
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        self._sync_task = asyncio.create_task(self._profile_sync_loop(), name="capture-agent-profile-sync")
        LOGGER.info("Moxue Capture Agent listening on http://%s:%s", self.host, self.port)

    async def close(self) -> None:
        if self._sync_task is not None:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            self._sync_task = None
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
            self._site = None

    async def _profile_sync_loop(self) -> None:
        while True:
            try:
                self.store.sync_meeting_config(virtual_screen_bounds())
            except Exception:
                LOGGER.debug("Could not sync active capture profile", exc_info=True)
            await asyncio.sleep(5)

    async def _health(self, _request: web.Request) -> web.Response:
        screen = virtual_screen_bounds()
        active = self.store.sync_meeting_config(screen)
        return web.json_response(
            {
                "ok": True,
                "agent_id": self.store.agent_id,
                "name": self.store.agent_name,
                "screen": {"x": screen.x, "y": screen.y, "width": screen.width, "height": screen.height},
                "active_profile": self._profile_payload(active, screen) if active else None,
                "profile_count": len(self.store.list_profiles()),
                "remote_enabled": False,
            },
            dumps=lambda value: json.dumps(value, ensure_ascii=False),
        )

    async def _profiles(self, _request: web.Request) -> web.Response:
        screen = virtual_screen_bounds()
        active = self.store.active_profile()
        profiles = [self._profile_payload(item, screen) for item in self.store.list_profiles()]
        return web.json_response(
            {
                "agent_id": self.store.agent_id,
                "name": self.store.agent_name,
                "active_profile_id": active.id if active else None,
                "profiles": profiles,
            },
            dumps=lambda value: json.dumps(value, ensure_ascii=False),
        )

    async def _select_profile(self, request: web.Request) -> web.Response:
        payload = await request.json()
        profile_id = str(payload.get("profile_id") or "").strip()
        profile = self.store.select(profile_id)
        if profile is None:
            return web.json_response({"ok": False, "message": "找不到這個 Capture Profile。"}, status=404)
        screen = virtual_screen_bounds()
        self.store.sync_meeting_config(screen)
        return web.json_response(
            {"ok": True, "message": "已切換 Capture Profile。", "profile": self._profile_payload(profile, screen)},
            dumps=lambda value: json.dumps(value, ensure_ascii=False),
        )

    async def _calibrate(self, request: web.Request) -> web.Response:
        payload = await request.json()
        game = str(payload.get("game") or "game").strip() or "game"
        profile_name_raw = payload.get("profile_name")
        profile_name = str(profile_name_raw).strip() if isinstance(profile_name_raw, str) and profile_name_raw.strip() else None
        delay_raw = payload.get("delay_seconds", 5)
        try:
            delay_seconds = float(delay_raw)
        except (TypeError, ValueError):
            delay_seconds = 5.0
        if self._calibration_lock.locked():
            return web.json_response({"ok": False, "message": "目前已有校準視窗進行中。"}, status=409)
        async with self._calibration_lock:
            profile, backend = await asyncio.to_thread(
                calibrate_profile,
                self.store,
                game=game,
                profile_name=profile_name,
                delay_seconds=delay_seconds,
            )
        if profile is None:
            return web.json_response({"ok": False, "cancelled": True, "message": "已取消聊天室框選。"})
        screen = virtual_screen_bounds()
        return web.json_response(
            {
                "ok": True,
                "message": "聊天室 ROI 已完成框選並保存為裝置 Profile。",
                "backend": backend,
                "profile": self._profile_payload(profile, screen),
            },
            dumps=lambda value: json.dumps(value, ensure_ascii=False),
        )

    async def _ocr_test(self, request: web.Request) -> web.Response:
        payload = await request.json() if request.can_read_body else {}
        profile_id = str(payload.get("profile_id") or "").strip() if isinstance(payload, dict) else ""
        profile = self.store.select(profile_id) if profile_id else self.store.active_profile()
        if profile is None:
            return web.json_response({"ok": False, "message": "尚未設定聊天室 Capture Profile。"}, status=400)
        captured = await asyncio.to_thread(capture_screen, prefer_dxcam=True)
        roi = profile.pixel_roi(captured.bounds)
        cropped = crop_profile(captured.image, roi, captured.bounds)
        processed = self._preprocess(cropped)
        CALIBRATION_ROOT.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        original_path = CALIBRATION_ROOT / f"roi_{stamp}.png"
        processed_path = CALIBRATION_ROOT / f"roi_{stamp}_processed.png"
        await asyncio.to_thread(cropped.save, original_path)
        await asyncio.to_thread(processed.save, processed_path)
        lines = await asyncio.to_thread(self._ocr_lines, processed)
        self.store.sync_meeting_config(captured.bounds)
        return web.json_response(
            {
                "ok": True,
                "message": "Capture Agent OCR 測試完成。",
                "backend": captured.backend,
                "profile": self._profile_payload(profile, captured.bounds),
                "lines": lines,
                "original_image": str(original_path),
                "processed_image": str(processed_path),
            },
            dumps=lambda value: json.dumps(value, ensure_ascii=False),
        )

    @staticmethod
    def _preprocess(image: Image.Image) -> Image.Image:
        gray = ImageOps.grayscale(image)
        gray = ImageOps.autocontrast(gray)
        return gray.resize((gray.width * 2, gray.height * 2))

    def _ocr_lines(self, image: Image.Image) -> list[dict[str, object]]:
        try:
            import numpy as np
            from rapidocr import RapidOCR
        except ImportError as error:
            raise RuntimeError('Capture Agent OCR 依賴尚未安裝；請執行 python -m pip install -e ".[meeting]"') from error
        if self._ocr_engine is None:
            self._ocr_engine = RapidOCR()
        output = self._ocr_engine(np.asarray(image))
        texts = getattr(output, "txts", None)
        scores = getattr(output, "scores", None)
        if texts is not None:
            return [
                {"text": str(text), "confidence": round(float(scores[index]), 4) if scores is not None and index < len(scores) else None}
                for index, text in enumerate(texts)
                if str(text).strip()
            ]
        if isinstance(output, tuple) and output:
            rows = output[0] or []
        elif isinstance(output, list):
            rows = output
        else:
            rows = []
        result: list[dict[str, object]] = []
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            value = row[1]
            if isinstance(value, (list, tuple)) and value:
                text = str(value[0]).strip()
                confidence = float(value[1]) if len(value) > 1 else None
            else:
                text = str(value).strip()
                confidence = None
            if text:
                result.append({"text": text, "confidence": round(confidence, 4) if confidence is not None else None})
        return result

    @staticmethod
    def _profile_payload(profile: CaptureProfile | None, screen: Any) -> dict[str, object] | None:
        if profile is None:
            return None
        return {
            "id": profile.id,
            "name": profile.name,
            "game": profile.game,
            "calibrated_screen": {
                "x": profile.screen.x,
                "y": profile.screen.y,
                "width": profile.screen.width,
                "height": profile.screen.height,
            },
            "normalized_roi": {
                "x": profile.roi_x,
                "y": profile.roi_y,
                "width": profile.roi_width,
                "height": profile.roi_height,
            },
            "pixel_roi": profile.pixel_roi(screen),
        }
