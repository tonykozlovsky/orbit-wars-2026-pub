"""Append-only debug tapes: vector primitives per frame, newline-delimited JSON on disk."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_TAPE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def _color_to_rgba(color: tuple[int, int, int, int] | int) -> tuple[int, int, int, int]:
    if isinstance(color, int):
        c = int(color) & 0xFFFFFF
        r = (c >> 16) & 0xFF
        g = (c >> 8) & 0xFF
        b = c & 0xFF
        return (r, g, b, 255)
    assert len(color) == 4, "color tuple must be RGBA with 4 ints"
    r, g, b, a = (int(color[i]) for i in range(4))
    assert 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255 and 0 <= a <= 255
    return (r, g, b, a)


class DebugViewer:
    """Buffers primitives for the current frame; ``finish_frame`` appends one JSON line to ``frames.jsonl``."""

    FRAMES_FILENAME = "frames.jsonl"
    VERSION = 2

    def __init__(self, tape_root: Path | str, tape_name: str) -> None:
        tape_name = str(tape_name)
        assert _TAPE_NAME_PATTERN.fullmatch(tape_name), f"invalid tape name: {tape_name!r}"
        self._tape_dir = Path(tape_root).expanduser().resolve() / tape_name
        self._tape_dir.mkdir(parents=True, exist_ok=True)
        self._frames_path = self._tape_dir / self.FRAMES_FILENAME
        self._lines: list[dict[str, Any]] = []
        self._points: list[dict[str, Any]] = []
        self._texts: list[dict[str, Any]] = []

    @property
    def tape_dir(self) -> Path:
        return self._tape_dir

    @property
    def frames_path(self) -> Path:
        return self._frames_path

    def draw_line(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        *,
        width_m: float = 0.08,
        color_rgba: tuple[int, int, int, int] | int = 0x1D4ED8,
        layer: str = "default",
    ) -> None:
        assert width_m > 0.0
        rgba = _color_to_rgba(color_rgba)
        self._lines.append(
            {
                "x0": float(x0),
                "y0": float(y0),
                "x1": float(x1),
                "y1": float(y1),
                "w_m": float(width_m),
                "color": [rgba[0], rgba[1], rgba[2], rgba[3]],
                "layer": str(layer),
            }
        )

    def draw_point(
        self,
        x: float,
        y: float,
        *,
        radius_m: float = 0.12,
        color_rgba: tuple[int, int, int, int] | int = 0xEF4444,
        layer: str = "default",
    ) -> None:
        assert radius_m > 0.0
        rgba = _color_to_rgba(color_rgba)
        self._points.append(
            {
                "x": float(x),
                "y": float(y),
                "r_m": float(radius_m),
                "color": [rgba[0], rgba[1], rgba[2], rgba[3]],
                "layer": str(layer),
            }
        )

    def draw_text(
        self,
        x: float,
        y: float,
        text: str,
        *,
        size_px: float = 12.0,
        color_rgba: tuple[int, int, int, int] | int = 0xF8FAFC,
        layer: str = "default",
    ) -> None:
        assert size_px > 0.0
        rgba = _color_to_rgba(color_rgba)
        self._texts.append(
            {
                "x": float(x),
                "y": float(y),
                "text": str(text),
                "size_px": float(size_px),
                "color": [rgba[0], rgba[1], rgba[2], rgba[3]],
                "layer": str(layer),
            }
        )

    def finish_frame(self) -> None:
        payload = {
            "version": int(self.VERSION),
            "lines": list(self._lines),
            "points": list(self._points),
            "texts": list(self._texts),
        }
        line = json.dumps(payload, separators=(",", ":"), sort_keys=False)
        with self._frames_path.open("ab") as out:
            out.write((line + "\n").encode("utf-8"))
        self._lines.clear()
        self._points.clear()
        self._texts.clear()


def append_frames_to_tape(
    tape_root: Path | str,
    tape_name: str,
    frames: list[dict[str, Any]],
) -> Path:
    """
    Append frame lines to ``tape_root/tape_name/frames.jsonl`` (mode ``a``).
    Same JSON line format as ``DebugViewer.finish_frame``.
    """
    viewer = DebugViewer(tape_root, tape_name)
    with viewer.frames_path.open("ab") as out:
        for fr in frames:
            line = json.dumps(fr, separators=(",", ":"), sort_keys=False)
            out.write((line + "\n").encode("utf-8"))
    return viewer.frames_path
