"""Collect hardware and tool diagnostics without hiding missing tools."""

import platform
import sys

from .common import capture


def hardware_report() -> dict:
    commands = (
        ["lscpu"],
        ["free", "-h"],
        ["nvidia-smi"],
        ["nvcc", "--version"],
        ["cmake", "--version"],
        ["git", "--version"],
    )
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "checks": [capture(command) for command in commands],
    }
