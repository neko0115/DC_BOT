"""Restart the Discord bot and local Capture Agent automatically during development."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
EXTRA_FILES = (PROJECT_ROOT / ".env", PROJECT_ROOT / "pyproject.toml")
CAPTURE_AGENT_HOST = "127.0.0.1"
CAPTURE_AGENT_PORT = 8877


@dataclass(slots=True)
class DevelopmentProcesses:
    bot: subprocess.Popen[bytes]
    capture_agent: subprocess.Popen[bytes] | None


def snapshot() -> dict[Path, int]:
    files = {path for path in SOURCE_ROOT.rglob("*.py") if "__pycache__" not in path.parts}
    files.update(path for path in EXTRA_FILES if path.is_file())
    return {path: path.stat().st_mtime_ns for path in files}


def _start_module(module: str, *args: str) -> subprocess.Popen[bytes]:
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    return subprocess.Popen(
        [sys.executable, "-m", module, *args],
        cwd=PROJECT_ROOT,
        creationflags=flags,
    )


def _port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.4):
            return True
    except OSError:
        return False


def _existing_capture_agent(host: str = CAPTURE_AGENT_HOST, port: int = CAPTURE_AGENT_PORT) -> dict[str, object] | None:
    """Return Moxue Capture Agent health data when the occupied port belongs to one."""
    if not _port_is_open(host, port):
        return None
    try:
        with urlopen(f"http://{host}:{port}/health", timeout=1.5) as response:
            if response.status != 200:
                return None
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        return None
    if not payload.get("agent_id") or not payload.get("name"):
        return None
    return payload


def _start_capture_agent() -> subprocess.Popen[bytes] | None:
    existing = _existing_capture_agent()
    if existing is not None:
        print(
            "Existing Moxue Capture Agent detected on "
            f"http://{CAPTURE_AGENT_HOST}:{CAPTURE_AGENT_PORT} "
            f"({existing.get('name')}, {existing.get('agent_id')}); reusing it."
        )
        return None
    if _port_is_open(CAPTURE_AGENT_HOST, CAPTURE_AGENT_PORT):
        raise RuntimeError(
            f"Port {CAPTURE_AGENT_PORT} is already in use by another program. "
            "Close that program or free the port before starting the development runner."
        )
    return _start_module(
        "discord_ai_assistant.capture_agent.cli",
        "--host",
        CAPTURE_AGENT_HOST,
        "--port",
        str(CAPTURE_AGENT_PORT),
    )


def start_processes() -> DevelopmentProcesses:
    capture_agent = _start_capture_agent()
    try:
        bot = _start_module("discord_ai_assistant.main")
    except Exception:
        if capture_agent is not None:
            stop_process(capture_agent)
        raise
    return DevelopmentProcesses(bot=bot, capture_agent=capture_agent)


def stop_process(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.terminate()
        process.wait(timeout=15)
    except (subprocess.TimeoutExpired, OSError):
        process.kill()
        process.wait()


def stop_processes(processes: DevelopmentProcesses) -> None:
    stop_process(processes.bot)
    # Only stop the Capture Agent if this runner started it. An already-running
    # standalone Agent is intentionally reused and remains owned by its creator.
    stop_process(processes.capture_agent)


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch source files and restart the Discord bot plus Capture Agent.")
    parser.add_argument("--interval", type=float, default=1.0, help="Polling interval in seconds (default: 1.0).")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be greater than zero")

    previous = snapshot()
    try:
        processes = start_processes()
    except RuntimeError as error:
        parser.error(str(error))
    print("Development runner started. Press Ctrl+C to stop the runner, bot, and owned Capture Agent.")
    try:
        while True:
            time.sleep(args.interval)
            current = snapshot()
            if current == previous:
                continue
            print("Source or configuration changed; restarting bot and owned Capture Agent.")
            stop_processes(processes)
            previous = current
            try:
                processes = start_processes()
            except RuntimeError as error:
                print(f"Development restart failed: {error}", file=sys.stderr)
                raise
    except KeyboardInterrupt:
        print("Stopping development runner.")
    finally:
        stop_processes(processes)


if __name__ == "__main__":
    main()
