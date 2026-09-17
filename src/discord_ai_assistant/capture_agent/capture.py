from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageGrab

from .profiles import ScreenBounds


@dataclass(frozen=True, slots=True)
class CapturedScreen:
    image: Image.Image
    bounds: ScreenBounds
    backend: str


def virtual_screen_bounds() -> ScreenBounds:
    if os.name != "nt":
        image = ImageGrab.grab(all_screens=True)
        return ScreenBounds(0, 0, image.width, image.height)
    user32 = ctypes.windll.user32
    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass
    x = int(user32.GetSystemMetrics(76))
    y = int(user32.GetSystemMetrics(77))
    width = int(user32.GetSystemMetrics(78))
    height = int(user32.GetSystemMetrics(79))
    if width <= 0 or height <= 0:
        width = int(user32.GetSystemMetrics(0))
        height = int(user32.GetSystemMetrics(1))
        x = 0
        y = 0
    return ScreenBounds(x, y, width, height)


def _capture_dxcam() -> CapturedScreen | None:
    if os.name != "nt":
        return None
    try:
        import dxcam
    except (ImportError, OSError):
        return None
    camera: Any | None = None
    try:
        camera = dxcam.create(output_idx=0, output_color="RGB")
        frame: Any = camera.grab()
        if frame is None:
            return None
        image = Image.fromarray(frame).copy()
        bounds = ScreenBounds(0, 0, image.width, image.height)
        return CapturedScreen(image=image, bounds=bounds, backend="dxcam")
    except Exception:
        return None
    finally:
        release = getattr(camera, "release", None)
        if callable(release):
            try:
                release()
            except Exception:
                pass


def capture_screen(*, prefer_dxcam: bool = True) -> CapturedScreen:
    if prefer_dxcam:
        captured = _capture_dxcam()
        if captured is not None:
            return captured
    bounds = virtual_screen_bounds()
    image = ImageGrab.grab(all_screens=True)
    if image.width != bounds.width or image.height != bounds.height:
        bounds = ScreenBounds(bounds.x, bounds.y, image.width, image.height)
    return CapturedScreen(image=image.convert("RGB"), bounds=bounds, backend="imagegrab")


def crop_profile(image: Image.Image, profile_roi: dict[str, int], bounds: ScreenBounds) -> Image.Image:
    left = max(0, int(profile_roi["x"]) - bounds.x)
    top = max(0, int(profile_roi["y"]) - bounds.y)
    right = min(image.width, left + int(profile_roi["width"]))
    bottom = min(image.height, top + int(profile_roi["height"]))
    if right <= left or bottom <= top:
        raise ValueError("ROI 超出目前螢幕範圍。")
    return image.crop((left, top, right, bottom))
