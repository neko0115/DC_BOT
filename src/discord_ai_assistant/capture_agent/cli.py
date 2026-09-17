from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from .profiles import DATA_ROOT
from .remote_client import CaptureAgentRemoteClient
from .remote_config import load_remote_state, normalize_hub_url, save_hub_url
from .server import CaptureAgentServer

REMOTE_STATE_PATH = DATA_ROOT / "remote.json"


def _prompt_hub_url() -> str | None:
    try:
        import tkinter as tk
        from tkinter import messagebox, simpledialog
    except ImportError:
        return None
    root = tk.Tk()
    root.withdraw()
    try:
        while True:
            value = simpledialog.askstring(
                "Moxue Capture",
                "輸入墨雪 Server 的 IP / hostname。\n\n"
                "例如：192.168.1.20\n"
                "也可以輸入完整位址：ws://192.168.1.20:8878/capture/ws",
                parent=root,
            )
            if value is None:
                return None
            try:
                normalized = normalize_hub_url(value)
            except ValueError as error:
                messagebox.showerror("Moxue Capture", str(error), parent=root)
                continue
            if normalized:
                return normalized
    finally:
        root.destroy()


def _resolve_hub_url(cli_value: str | None, *, local_only: bool) -> str | None:
    if local_only:
        return None
    if cli_value:
        normalized = normalize_hub_url(cli_value)
        save_hub_url(REMOTE_STATE_PATH, normalized)
        return normalized
    state = load_remote_state(REMOTE_STATE_PATH)
    stored = str(state.get("hub_url") or "").strip()
    if stored:
        try:
            return normalize_hub_url(stored)
        except ValueError:
            pass
    if getattr(sys, "frozen", False):
        prompted = _prompt_hub_url()
        if prompted:
            save_hub_url(REMOTE_STATE_PATH, prompted)
        return prompted
    return None


async def _run(host: str, port: int, hub_url: str | None) -> None:
    server = CaptureAgentServer(host=host, port=port)
    await server.start()
    remote: CaptureAgentRemoteClient | None = None
    remote_task: asyncio.Task[None] | None = None
    if hub_url:
        remote = CaptureAgentRemoteClient(hub_url, local_url=f"http://127.0.0.1:{port}")
        remote_task = asyncio.create_task(remote.run_forever(), name="capture-agent-remote-client")
    try:
        await asyncio.Event().wait()
    finally:
        if remote is not None:
            await remote.close()
        if remote_task is not None:
            remote_task.cancel()
            try:
                await remote_task
            except asyncio.CancelledError:
                pass
        await server.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Moxue Capture Agent.")
    parser.add_argument("--host", default="127.0.0.1", help="Local calibration API bind host.")
    parser.add_argument("--port", type=int, default=8877, help="Local calibration API port.")
    parser.add_argument(
        "--hub-url",
        default=os.getenv("MOXUE_CAPTURE_HUB_URL", "").strip() or None,
        help="Outbound Hub WebSocket URL or Server IP, e.g. 192.168.1.20 or ws://192.168.1.20:8878/capture/ws",
    )
    parser.add_argument("--local-only", action="store_true", help="Do not connect to a remote Moxue Capture Hub.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        hub_url = _resolve_hub_url(args.hub_url, local_only=args.local_only)
    except ValueError as error:
        parser.error(str(error))
    try:
        asyncio.run(_run(args.host, args.port, hub_url))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
