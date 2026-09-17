from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from PIL import ImageTk

from .capture import CapturedScreen, capture_screen
from .profiles import CaptureProfile, ProfileStore, ScreenBounds


@dataclass(slots=True)
class SelectionState:
    start_x: int | None = None
    start_y: int | None = None
    end_x: int | None = None
    end_y: int | None = None

    def rectangle(self) -> tuple[int, int, int, int] | None:
        if None in {self.start_x, self.start_y, self.end_x, self.end_y}:
            return None
        assert self.start_x is not None and self.start_y is not None
        assert self.end_x is not None and self.end_y is not None
        left, right = sorted((self.start_x, self.end_x))
        top, bottom = sorted((self.start_y, self.end_y))
        if right - left < 20 or bottom - top < 20:
            return None
        return left, top, right, bottom


def _show_selector(captured: CapturedScreen) -> tuple[int, int, int, int] | None:
    import tkinter as tk

    root = tk.Tk()
    root.title("Moxue Capture Calibration")
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    geometry = f"{captured.bounds.width}x{captured.bounds.height}{captured.bounds.x:+d}{captured.bounds.y:+d}"
    root.geometry(geometry)
    root.configure(cursor="crosshair")

    canvas = tk.Canvas(root, width=captured.bounds.width, height=captured.bounds.height, highlightthickness=0)
    canvas.pack(fill="both", expand=True)
    photo = ImageTk.PhotoImage(captured.image)
    canvas.create_image(0, 0, image=photo, anchor="nw")
    canvas.image = photo

    state = SelectionState()
    rectangle_id: int | None = None
    text_id = canvas.create_text(
        24,
        24,
        anchor="nw",
        text="拖曳框選遊戲聊天室。Enter 儲存，Esc 取消。",
        fill="white",
        font=("Segoe UI", 18, "bold"),
    )
    background_id = canvas.create_rectangle(12, 12, 560, 58, fill="black", stipple="gray50", outline="")
    canvas.tag_raise(text_id, background_id)

    result: dict[str, tuple[int, int, int, int] | None] = {"selection": None}

    def on_press(event: tk.Event) -> None:
        nonlocal rectangle_id
        state.start_x = int(event.x)
        state.start_y = int(event.y)
        state.end_x = int(event.x)
        state.end_y = int(event.y)
        if rectangle_id is not None:
            canvas.delete(rectangle_id)
        rectangle_id = canvas.create_rectangle(
            event.x,
            event.y,
            event.x,
            event.y,
            outline="#00ff88",
            width=3,
        )

    def on_drag(event: tk.Event) -> None:
        state.end_x = int(event.x)
        state.end_y = int(event.y)
        if rectangle_id is not None and state.start_x is not None and state.start_y is not None:
            canvas.coords(rectangle_id, state.start_x, state.start_y, event.x, event.y)

    def on_release(event: tk.Event) -> None:
        state.end_x = int(event.x)
        state.end_y = int(event.y)

    def confirm(_event: tk.Event | None = None) -> None:
        selection = state.rectangle()
        if selection is None:
            return
        result["selection"] = selection
        root.destroy()

    def cancel(_event: tk.Event | None = None) -> None:
        result["selection"] = None
        root.destroy()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    root.bind("<Return>", confirm)
    root.bind("<Escape>", cancel)
    root.focus_force()
    root.mainloop()
    return result["selection"]


def calibrate_profile(
    store: ProfileStore,
    *,
    game: str,
    profile_name: str | None = None,
    delay_seconds: float = 5.0,
) -> tuple[CaptureProfile | None, str]:
    delay = max(0.0, min(15.0, float(delay_seconds)))
    if delay:
        time.sleep(delay)
    captured = capture_screen(prefer_dxcam=True)
    selection = _show_selector(captured)
    if selection is None:
        return None, captured.backend

    left, top, right, bottom = selection
    bounds = captured.bounds
    profile = CaptureProfile(
        id=str(uuid.uuid4()),
        name=(profile_name or f"{game}-{bounds.width}x{bounds.height}").strip(),
        game=(game or "unknown").strip(),
        screen=ScreenBounds(bounds.x, bounds.y, bounds.width, bounds.height),
        roi_x=left / bounds.width,
        roi_y=top / bounds.height,
        roi_width=(right - left) / bounds.width,
        roi_height=(bottom - top) / bounds.height,
    )
    store.upsert(profile, activate=True)
    store.sync_meeting_config(bounds)
    return profile, captured.backend
