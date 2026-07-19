#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import mmap
import os
import re
import threading
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

INT_PATTERN = re.compile(r"^-?\d+$")
TAPE_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
_IMPALA_PROJECT_ROOT_RAW = os.environ.get(
    "IMPALA_PROJECT_ROOT",
    str(Path(__file__).resolve().parents[2]),
).strip()
assert _IMPALA_PROJECT_ROOT_RAW, "IMPALA_PROJECT_ROOT must be non-empty"
_IMPALA_PROJECT_ROOT = Path(_IMPALA_PROJECT_ROOT_RAW).expanduser().resolve()
_DEFAULT_TAPES_OUTPUT_PARENT = _IMPALA_PROJECT_ROOT / "outputs"
DEFAULT_TAPES_ROOTS = (
    _DEFAULT_TAPES_OUTPUT_PARENT / "analyze",
    _DEFAULT_TAPES_OUTPUT_PARENT / "vis",
)
DEFAULT_TAPE_ID = ""

_tape_frame_ticket_lock = threading.Lock()
_tape_frame_issue_counter = 0
_tape_frame_work_lock = threading.Lock()


def _read_jsonl_line_offsets(path: Path, start_offset: int = 0) -> tuple[list[tuple[int, int]], int]:
    assert start_offset >= 0
    offsets: list[tuple[int, int]] = []
    offset = start_offset
    with path.open("rb") as handle:
        handle.seek(start_offset)
        for line in handle:
            next_offset = offset + len(line)
            if line == b"\n" or not line.endswith(b"\n"):
                return offsets, offset
            line_end = next_offset - 1
            offsets.append((offset, line_end))
            offset = next_offset
    return offsets, offset


def _iter_tape_frames_paths_under_root(tapes_root: Path) -> list[tuple[str, Path]]:
    if not tapes_root.exists():
        return []
    assert tapes_root.is_dir(), f"Tapes root is not a directory: {tapes_root}"
    out: list[tuple[str, Path]] = []
    for child in sorted(tapes_root.iterdir()):
        if not child.is_dir():
            continue
        nested = child / "tapes"
        if nested.is_dir():
            for tape_dir in sorted(nested.iterdir()):
                if not tape_dir.is_dir():
                    continue
                frames_path = tape_dir / "frames.jsonl"
                if not frames_path.is_file():
                    continue
                tape_id = f"{child.name}__{tape_dir.name}"
                if not TAPE_ID_PATTERN.fullmatch(tape_id):
                    continue
                out.append((tape_id, frames_path))
        else:
            frames_path = child / "frames.jsonl"
            if frames_path.is_file():
                tape_id = child.name
                if not TAPE_ID_PATTERN.fullmatch(tape_id):
                    continue
                out.append((tape_id, frames_path))
            else:
                for sub in sorted(child.iterdir()):
                    if not sub.is_dir():
                        continue
                    frames_path = sub / "frames.jsonl"
                    if not frames_path.is_file():
                        continue
                    tape_id = f"{child.name}__{sub.name}"
                    if not TAPE_ID_PATTERN.fullmatch(tape_id):
                        continue
                    out.append((tape_id, frames_path))
    return out


def _resolve_frames_path_under_root(tapes_root: Path, tape_id: str) -> Path | None:
    if "__" in tape_id:
        run_name, tape_name = tape_id.split("__", 1)
        nested = tapes_root / run_name / "tapes" / tape_name / "frames.jsonl"
        if nested.is_file():
            return nested
        grouped = tapes_root / run_name / tape_name / "frames.jsonl"
        if grouped.is_file():
            return grouped
        flat = tapes_root / tape_id / "frames.jsonl"
        if flat.is_file():
            return flat
        return None
    candidate = tapes_root / tape_id / "frames.jsonl"
    if candidate.is_file():
        return candidate
    return None


class TapeCatalog:
    """Tape discovery under one or more ``tapes_roots`` (merged list, unique ``id``).

    Under each root:

    * **Impala runs** (MonoBeast): ``<root>/<run_id>/tapes/<tape_name>/frames.jsonl``
    * **Flat**: ``<root>/<tape_id>/frames.jsonl``
    * **Grouped** (e.g. analyze output): ``<root>/<group>/<tape_name>/frames.jsonl``
      (API id ``<group>__<tape_name>``, same pattern as nested).
    Nested Impala tapes use API id ``<run_id>__<tape_name>`` (double underscore).
    """

    def __init__(self, tapes_roots: Sequence[Path], *, preferred_default_tape_id: str = ""):
        roots = tuple(Path(p).expanduser().resolve() for p in tapes_roots)
        assert len(roots) >= 1, "tapes_roots must contain at least one path"
        self._tapes_roots = roots
        self._preferred_default_tape_id = str(preferred_default_tape_id).strip()
        self._cached_tape_id = ""
        self._cached_frames_path: Path | None = None
        self._cached_mtime_ns = -1
        self._cached_size = -1
        self._cached_line_offsets: list[tuple[int, int]] = []
        self._cached_file = None
        self._cached_mmap: mmap.mmap | None = None

    @property
    def tapes_roots(self) -> tuple[Path, ...]:
        return self._tapes_roots

    def _iter_tape_frames_paths(self) -> list[tuple[str, Path]]:
        by_id: dict[str, Path] = {}
        order: list[str] = []
        for root in self._tapes_roots:
            for tape_id, frames_path in _iter_tape_frames_paths_under_root(root):
                if tape_id in by_id:
                    prev = by_id[tape_id]
                    assert prev == frames_path, (
                        f"duplicate tape id {tape_id!r} under different paths: {prev} vs {frames_path}"
                    )
                    continue
                by_id[tape_id] = frames_path
                order.append(tape_id)
        return [(i, by_id[i]) for i in order]

    def list_tapes(self) -> list[dict[str, object]]:
        tapes: list[dict[str, object]] = []
        for tape_id, frames_path in self._iter_tape_frames_paths():
            frame_offsets, _ = _read_jsonl_line_offsets(frames_path)
            tapes.append(
                {
                    "id": tape_id,
                    "frame_count": int(len(frame_offsets)),
                }
            )
        return tapes

    @property
    def default_tape_id(self) -> str:
        pairs = self._iter_tape_frames_paths()
        if len(pairs) == 0:
            return ""
        pref = self._preferred_default_tape_id
        if pref and TAPE_ID_PATTERN.fullmatch(pref):
            for tape_id, _ in pairs:
                if tape_id == pref:
                    return tape_id
        return pairs[-1][0]

    def _resolve_frames_path(self, tape_id: str) -> Path | None:
        for root in self._tapes_roots:
            p = _resolve_frames_path_under_root(root, tape_id)
            if p is not None:
                return p
        return None

    def has_tape(self, tape_id: str) -> bool:
        if not TAPE_ID_PATTERN.fullmatch(tape_id):
            return False
        p = self._resolve_frames_path(tape_id)
        return p is not None

    def _cache_matches_tape(self, tape_id: str, frames_path: Path) -> bool:
        return (
            self._cached_tape_id == tape_id
            and self._cached_frames_path == frames_path
        )

    def _close_cached_tape(self) -> None:
        if self._cached_mmap is not None:
            self._cached_mmap.close()
        if self._cached_file is not None:
            self._cached_file.close()
        self._cached_tape_id = ""
        self._cached_frames_path = None
        self._cached_mtime_ns = -1
        self._cached_size = -1
        self._cached_line_offsets = []
        self._cached_file = None
        self._cached_mmap = None

    def _load_cached_tape(self, tape_id: str, frames_path: Path) -> None:
        stat = frames_path.stat()
        size = int(stat.st_size)
        mtime_ns = int(stat.st_mtime_ns)
        if self._cache_matches_tape(tape_id, frames_path):
            if size == self._cached_size:
                self._cached_mtime_ns = mtime_ns
                return
            if size > self._cached_size:
                new_line_offsets, indexed_size = _read_jsonl_line_offsets(
                    frames_path,
                    self._cached_size,
                )
                line_offsets = self._cached_line_offsets + new_line_offsets
            else:
                line_offsets, indexed_size = _read_jsonl_line_offsets(frames_path)
            self._close_cached_tape()
        else:
            self._close_cached_tape()
            line_offsets, indexed_size = _read_jsonl_line_offsets(frames_path)

        cached_file = None
        cached_mmap = None
        if indexed_size > 0:
            cached_file = frames_path.open("rb")
            cached_mmap = mmap.mmap(cached_file.fileno(), indexed_size, access=mmap.ACCESS_READ)
        self._cached_tape_id = tape_id
        self._cached_frames_path = frames_path
        self._cached_mtime_ns = mtime_ns
        self._cached_size = indexed_size
        self._cached_line_offsets = line_offsets
        self._cached_file = cached_file
        self._cached_mmap = cached_mmap

    def get_frame_json_bytes(self, tape_id: str, index: int) -> bytes:
        assert TAPE_ID_PATTERN.fullmatch(tape_id), f"Invalid tape id: {tape_id}"
        frames_path = self._resolve_frames_path(tape_id)
        assert frames_path is not None and frames_path.is_file(), (
            f"Tape not found or missing frames.jsonl: {tape_id}"
        )
        assert index >= 0
        self._load_cached_tape(tape_id, frames_path)
        if index >= len(self._cached_line_offsets):
            raise IndexError(f"tape frame index out of range: {index}")
        cached_mmap = self._cached_mmap
        assert cached_mmap is not None
        line_start, line_end = self._cached_line_offsets[index]
        return bytes(cached_mmap[line_start:line_end])


class OrbitTapeApiHandler(BaseHTTPRequestHandler):
    tape_catalog: TapeCatalog | None = None

    def do_OPTIONS(self):
        self._write_headers(status_code=204, content_type="text/plain; charset=utf-8")
        self.wfile.write(b"")

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._write_json({"ok": True})
            return
        if parsed.path == "/api/tapes":
            assert self.tape_catalog is not None
            self._write_json(
                {
                    "default_tape_id": self.tape_catalog.default_tape_id,
                    "tapes": self.tape_catalog.list_tapes(),
                }
            )
            return
        if parsed.path == "/api/tapes/frame":
            self._handle_tape_frame_request(parsed.query)
            return
        self._write_json({"error": "Not Found"}, status_code=404)

    def _handle_tape_frame_request(self, query_string: str) -> None:
        assert self.tape_catalog is not None
        query = parse_qs(query_string, keep_blank_values=False)
        tape_values = query.get("tape")
        index_values = query.get("index")
        if tape_values is None or len(tape_values) != 1:
            self._write_json({"error": "tape query parameter is required"}, status_code=400)
            return
        if index_values is None or len(index_values) != 1:
            self._write_json({"error": "index query parameter is required"}, status_code=400)
            return
        tape_id = tape_values[0]
        if not TAPE_ID_PATTERN.fullmatch(tape_id):
            self._write_json({"error": f"invalid tape id: {tape_id}"}, status_code=400)
            return
        index_raw = index_values[0]
        if not INT_PATTERN.fullmatch(index_raw):
            self._write_json({"error": "index must be an integer value"}, status_code=400)
            return
        frame_index = int(index_raw)
        if frame_index < 0:
            self._write_json({"error": "index must be >= 0"}, status_code=400)
            return
        if not self.tape_catalog.has_tape(tape_id):
            self._write_json({"error": f"tape not found: {tape_id}"}, status_code=404)
            return
        global _tape_frame_issue_counter
        with _tape_frame_ticket_lock:
            _tape_frame_issue_counter += 1
            my_ticket = _tape_frame_issue_counter
        with _tape_frame_work_lock:
            if my_ticket != _tape_frame_issue_counter:
                self.close_connection = True
                return
            try:
                frame_payload = self.tape_catalog.get_frame_json_bytes(tape_id, frame_index)
            except IndexError:
                if my_ticket != _tape_frame_issue_counter:
                    self.close_connection = True
                    return
                self._write_json({"error": f"index out of range: {frame_index}"}, status_code=400)
                return
            if my_ticket != _tape_frame_issue_counter:
                self.close_connection = True
                return
            self._write_raw_json(frame_payload)

    def _write_headers(self, status_code: int, content_type: str) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _write_json(self, payload: object, status_code: int = 200) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self._write_raw_json(raw, status_code=status_code)

    def _write_raw_json(self, raw: bytes, status_code: int = 200) -> None:
        self._write_headers(status_code=status_code, content_type="application/json; charset=utf-8")
        try:
            self.wfile.write(raw)
        except BrokenPipeError:
            pass
        except ConnectionResetError:
            pass


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve Orbit Wars debug tapes.")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Bind host (default: {DEFAULT_HOST})")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Bind port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--tapes-root",
        action="append",
        default=None,
        metavar="DIR",
        help=(
            "Directory scanned for tapes (repeat for multiple). Each DIR uses the same layout as before: "
            "<DIR>/<run>/tapes/<name>/frames.jsonl, flat <DIR>/<id>/frames.jsonl, or grouped "
            "<DIR>/<group>/<name>/frames.jsonl. "
            f"Default: {DEFAULT_TAPES_ROOTS[0]} and {DEFAULT_TAPES_ROOTS[1]}."
        ),
    )
    parser.add_argument(
        "--default-tape-id",
        default=DEFAULT_TAPE_ID,
        help=(
            "Preferred tape id for /api/tapes when listed (nested form RUN__tape_name). "
            "Empty string falls back to last sorted tape."
        ),
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    raw_tape_roots = args.tapes_root
    if raw_tape_roots is None:
        tapes_roots = [p.expanduser().resolve() for p in DEFAULT_TAPES_ROOTS]
    else:
        tapes_roots = [Path(p).expanduser().resolve() for p in raw_tape_roots]
    preferred_tape = str(args.default_tape_id).strip()
    assert args.port > 0
    tape_catalog = TapeCatalog(
        tapes_roots=tapes_roots,
        preferred_default_tape_id=preferred_tape,
    )
    OrbitTapeApiHandler.tape_catalog = tape_catalog

    server = ThreadingHTTPServer((args.host, int(args.port)), OrbitTapeApiHandler)
    print(f"tapes_roots={tapes_roots}")
    print(f"default_tape_id={tape_catalog.default_tape_id}")
    print(f"serving=http://{args.host}:{int(args.port)}")
    server.serve_forever()


if __name__ == "__main__":
    main()
