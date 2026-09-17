"""Restart the Discord bot automatically when source or configuration files change."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
EXTRA_FILES = (PROJECT_ROOT / ".env", PROJECT_ROOT / "pyproject.toml")


def snapshot() -> dict[Path, int]:
    files = {path for path in SOURCE_ROOT.rglob("*.py") if "__pycache__" not in path.parts}
    files.update(path for path in EXTRA_FILES if path.is_file())
    return {path: path.stat().st_mtime_ns for path in files}


def start_bot() -> subprocess.Popen[bytes]:
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    return subprocess.Popen(
        [sys.executable, "-m", "discord_ai_assistant.main"],
        cwd=PROJECT_ROOT,
        creationflags=flags,
    )


def stop_bot(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch source files and restart the Discord bot on changes.")
    parser.add_argument("--interval", type=float, default=1.0, help="Polling interval in seconds (default: 1.0).")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be greater than zero")

    previous = snapshot()
    process = start_bot()
    print("Development runner started. Press Ctrl+C to stop both the runner and the bot.")
    try:
        while True:
            time.sleep(args.interval)
            current = snapshot()
            if current == previous:
                continue
            print("Source or configuration changed; restarting the bot.")
            stop_bot(process)
            previous = current
            process = start_bot()
    except KeyboardInterrupt:
        print("Stopping development runner.")
    finally:
        stop_bot(process)


if __name__ == "__main__":
    main()
