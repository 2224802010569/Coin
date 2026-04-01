from __future__ import annotations

import sys


class TrainingLogger:
    def __init__(self, width: int = 28):
        self.width = max(10, int(width))
        self._fallback_mode = False

    def progress(self, prefix: str, current: int, total: int, suffix: str = "") -> None:
        total = max(total, 1)
        current = min(max(current, 0), total)
        ratio = current / total
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        if self._fallback_mode:
            message = f"{prefix} [{bar}] {current}/{total}"
        else:
            message = f"\r{prefix} [{bar}] {current}/{total}"
        if suffix:
            message += f" {suffix}"
        try:
            sys.stdout.write(message)
            sys.stdout.flush()
            if current >= total:
                sys.stdout.write("\n")
                sys.stdout.flush()
        except OSError:
            self._fallback_mode = True
            print(message.replace("\r", ""))

    def info(self, message: str) -> None:
        print(message)
