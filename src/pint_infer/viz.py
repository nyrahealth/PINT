"""Readable renderings of PINT outputs, with the standard library only.

Discrete ids are shown as a *token strip*: one colored band per run of identical ids
along a time axis, so parallel utterances can be compared by eye. The color of an id
is a deterministic function of the id, so the same token looks the same across
utterances, models and figures. Continuous frames are shown as a heatmap or as a
frame-to-frame cosine self-similarity matrix, both rasterized into a PNG that is
embedded in the SVG (no matplotlib, no PIL).

Every function returns an SVG string; in a notebook, ``IPython.display.SVG(...)``
shows it, and ``Path("x.svg").write_text(...)`` saves it.
"""

import base64
import colorsys
import struct
import zlib
from collections.abc import Mapping, Sequence

import numpy as np

_GOLDEN_ANGLE = 0.618033988749895
_FONT = "ui-monospace, Menlo, monospace"


def runs(ids: Sequence[int]) -> list[tuple[int, int]]:
    """Collapse ``ids`` into ``[(id, run_length), ...]`` preserving order."""
    out: list[tuple[int, int]] = []
    for token in ids:
        if out and out[-1][0] == token:
            out[-1] = (token, out[-1][1] + 1)
        else:
            out.append((token, 1))
    return out


def format_runs(ids: Sequence[int], sep: str = " ") -> str:
    """Compact text form, e.g. ``"17×3 42 88×5"`` (a bare id is a run of one)."""
    return sep.join(f"{t}×{n}" if n > 1 else str(t) for t, n in runs(ids))


def token_color(token: int, saturation: float = 0.62, lightness: float = 0.58) -> str:
    """Deterministic, well-spread color for a token id (golden-angle hue walk)."""
    hue = (token * _GOLDEN_ANGLE) % 1.0
    lightness = lightness + (0.12 if token % 2 else -0.06)
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def token_strip_svg(
    sequences: Sequence[int] | Mapping[str, Sequence[int]],
    frame_ms: int = 20,
    px_per_second: float = 160.0,
    row_height: int = 28,
    label_min_frames: int = 3,
    title: str | None = None,
    dedup: bool = False,
) -> str:
    """One colored band per run of identical ids, rows stacked, shared time axis.

    Pass a single id list, or ``{"row label": ids, ...}`` to compare utterances
    (e.g. parallel speakers) against the same time axis. Runs of at least
    ``label_min_frames`` frames get the id printed on the band. With ``dedup``
    every run gets the same width and the axis counts runs instead of seconds, which
    makes sequences of different speaking rate directly comparable.
    """
    rows: Mapping[str, Sequence[int]] = (
        sequences if isinstance(sequences, Mapping) else {"": sequences}
    )
    row_runs = {name: runs(ids) for name, ids in rows.items()}
    audio_frame_ms = frame_ms  # the hover text keeps real durations in dedup mode
    if dedup:
        frame_ms, px_per_second = 1000, 22.0  # one "second" per run
    label_w = 8 + 7 * max((len(name) for name in rows), default=0)
    length = max((len(r) if dedup else len(rows[name]) for name, r in row_runs.items()), default=0)
    seconds = length * frame_ms / 1000.0
    width = int(label_w + seconds * px_per_second + 12)
    top = 22 if title else 4
    height = top + len(rows) * (row_height + 6) + 22
    px_per_frame = px_per_second * frame_ms / 1000.0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="{_FONT}" font-size="11">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
    ]
    if title:
        parts.append(f'<text x="{label_w}" y="15" font-size="12" fill="#222">{_esc(title)}</text>')
    for row, (name, id_runs) in enumerate(row_runs.items()):
        y = top + row * (row_height + 6)
        if name:
            parts.append(
                f'<text x="4" y="{y + row_height / 2 + 4}" fill="#333">{_esc(name)}</text>'
            )
        x = float(label_w)
        for token, n in id_runs:
            w = (1 if dedup else n) * px_per_frame
            parts.append(
                f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{row_height}" '
                f'fill="{token_color(token)}" stroke="white" stroke-width="0.5">'
                f"<title>id {token} × {n} frames ({n * audio_frame_ms} ms)</title></rect>"
            )
            if (dedup or n >= label_min_frames) and w >= 14:
                parts.append(
                    f'<text x="{x + w / 2:.1f}" y="{y + row_height / 2 + 4}" text-anchor="middle" '
                    f'fill="#111" font-size="{min(11, int(w / 1.6))}">{token}</text>'
                )
            x += w
    axis_y = top + len(rows) * (row_height + 6) + 4
    parts.append(
        f'<line x1="{label_w}" y1="{axis_y}" x2="{width - 12}" y2="{axis_y}" stroke="#888"/>'
    )
    tick = 0.0
    step = (5.0 if seconds > 30 else 1.0) if dedup else (0.5 if seconds <= 8 else 1.0)
    unit = " runs" if dedup else "s"
    while tick <= seconds + 1e-9:
        x = label_w + tick * px_per_second
        parts.append(
            f'<line x1="{x:.1f}" y1="{axis_y}" x2="{x:.1f}" y2="{axis_y + 4}" stroke="#888"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{axis_y + 15}" text-anchor="middle" fill="#555">'
            f"{tick:g}{unit if tick == 0 or not dedup else ''}</text>"
        )
        tick += step
    parts.append("</svg>")
    return "\n".join(parts)


def png_bytes(rgb: np.ndarray) -> bytes:
    """Encode an ``(H, W, 3)`` uint8 array as a PNG (stdlib zlib only)."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"expected (H, W, 3) uint8, got {rgb.shape} {rgb.dtype}")
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def colormap(values: np.ndarray) -> np.ndarray:
    """Map values in [0, 1] to an ``(..., 3)`` uint8 blue→yellow ramp (viridis-like)."""
    v = np.clip(np.nan_to_num(values, nan=0.0), 0.0, 1.0)
    stops = np.array(
        [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]],
        dtype=np.float64,
    )
    pos = v * (len(stops) - 1)
    low = np.floor(pos).astype(int)
    high = np.minimum(low + 1, len(stops) - 1)
    frac = (pos - low)[..., None]
    return np.asarray((stops[low] * (1 - frac) + stops[high] * frac).round(), dtype=np.uint8)


def _image_svg(
    rgb: np.ndarray,
    seconds: float,
    px_per_second: float,
    height_px: int,
    title: str | None,
    y_label: str,
) -> str:
    label_w = 40
    top = 22 if title else 4
    img_w = max(int(seconds * px_per_second), 1)
    width = label_w + img_w + 12
    height = top + height_px + 26
    data = base64.b64encode(png_bytes(rgb)).decode("ascii")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="{_FONT}" font-size="11">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
    ]
    if title:
        parts.append(f'<text x="{label_w}" y="15" font-size="12" fill="#222">{_esc(title)}</text>')
    parts.append(
        f'<image x="{label_w}" y="{top}" width="{img_w}" height="{height_px}" '
        f'preserveAspectRatio="none" style="image-rendering: pixelated" '
        f'href="data:image/png;base64,{data}"/>'
    )
    parts.append(
        f'<text x="12" y="{top + height_px / 2}" transform="rotate(-90 12 {top + height_px / 2})" '
        f'text-anchor="middle" fill="#555">{_esc(y_label)}</text>'
    )
    axis_y = top + height_px + 4
    tick, step = 0.0, (0.5 if seconds <= 8 else 1.0)
    while tick <= seconds + 1e-9:
        x = label_w + tick * px_per_second
        parts.append(
            f'<line x1="{x:.1f}" y1="{axis_y}" x2="{x:.1f}" y2="{axis_y + 4}" stroke="#888"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{axis_y + 15}" text-anchor="middle" fill="#555">{tick:g}s</text>'
        )
        tick += step
    parts.append("</svg>")
    return "\n".join(parts)


def embedding_heatmap_svg(
    frames: np.ndarray,
    frame_ms: int = 20,
    px_per_second: float = 160.0,
    height_px: int = 192,
    title: str | None = None,
) -> str:
    """``(T, D)`` continuous frames as a D×T heatmap (dims sorted by variance, per-dim
    min–max scaled), so the temporal structure of the embedding is visible."""
    if frames.ndim != 2:
        raise ValueError(f"expected (T, D) frames, got {frames.shape}")
    order = np.argsort(-frames.var(axis=0))
    x = frames[:, order].T.astype(np.float64)  # (D, T)
    lo, hi = x.min(axis=1, keepdims=True), x.max(axis=1, keepdims=True)
    scaled = (x - lo) / np.where(hi - lo > 0, hi - lo, 1.0)
    seconds = frames.shape[0] * frame_ms / 1000.0
    return _image_svg(colormap(scaled), seconds, px_per_second, height_px, title, "768 dims")


def self_similarity_svg(
    frames: np.ndarray,
    frame_ms: int = 20,
    px_per_second: float = 160.0,
    title: str | None = None,
) -> str:
    """Frame-to-frame cosine similarity ``(T, T)`` of continuous frames. Blocks along the
    diagonal are stretches the encoder considers the same unit; PINT makes them crisp."""
    if frames.ndim != 2:
        raise ValueError(f"expected (T, D) frames, got {frames.shape}")
    unit = frames / np.maximum(np.linalg.norm(frames, axis=1, keepdims=True), 1e-8)
    sim = (unit @ unit.T + 1.0) / 2.0
    seconds = frames.shape[0] * frame_ms / 1000.0
    side = max(int(seconds * px_per_second), 1)
    return _image_svg(colormap(sim), seconds, px_per_second, side, title, "time")


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
