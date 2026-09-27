"""Rendering helpers, WAV reading, the CLI and the README examples (model-free parts run
everywhere; model parts follow tests/test_parity.py's artifact policy)."""

import json
import runpy
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch
from pint_infer import __main__ as cli
from pint_infer.tokenizer import PINTTokenizer, read_wav
from pint_infer.viz import (
    colormap,
    embedding_heatmap_svg,
    format_runs,
    png_bytes,
    runs,
    self_similarity_svg,
    token_color,
    token_strip_svg,
)

from tests.test_parity import FIXTURES, tokenizer_for

EXAMPLES = Path(__file__).parents[1] / "examples"
# two channels of values every tested sample format represents exactly
SIGNAL = np.array([[0.0, 0.5, -0.5, -1.0], [0.25, -0.25, 0.75, 0.0]])
# the 14 bytes of the WAVE_FORMAT_EXTENSIBLE sub-format GUID after its 2-byte format tag
GUID_TAIL = bytes.fromhex("000000001000800000aa00389b71")


def test_runs_and_format() -> None:
    assert runs([]) == []
    assert runs([3, 3, 7, 3]) == [(3, 2), (7, 1), (3, 1)]
    assert format_runs([3, 3, 3, 7, 7, 3, 1]) == "3×3 7×2 3 1"


def test_token_color_is_deterministic_and_distinct() -> None:
    assert token_color(5) == token_color(5)
    assert len({token_color(i) for i in range(200)}) == 200
    assert all(len(token_color(i)) == 7 for i in range(1024))


def test_token_strip_svg_shape() -> None:
    svg = token_strip_svg({"a": [1, 1, 2], "b": [2]}, title="x & y")
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert svg.count("<rect") == 1 + 3  # background + 3 runs
    assert "x &amp; y" in svg and ">a<" in svg and ">b<" in svg
    assert "<title>id 1 × 2 frames (40 ms)</title>" in svg
    # a single list works too, and an empty one still renders an axis
    assert token_strip_svg([4, 4, 4, 4]).count("<rect") == 2
    assert token_strip_svg([]).count("<svg") == 1
    dedup = token_strip_svg({"a": [1, 1, 2, 2, 2], "b": [1, 2]}, dedup=True)
    # both rows collapse to two equal-width bands; the axis counts runs
    assert dedup.count('width="22.0"') == 4 and "0 runs" in dedup
    # the hover text keeps each run's real length
    assert "<title>id 2 × 3 frames (60 ms)</title>" in dedup


def test_png_bytes_roundtrip() -> None:
    rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    data = png_bytes(rgb)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    idat = data.index(b"IDAT") + 4
    length = int.from_bytes(data[idat - 8 : idat - 4], "big")
    raw = zlib.decompress(data[idat : idat + length])
    assert raw == b"\x00" + rgb[0].tobytes() + b"\x00" + rgb[1].tobytes()
    with pytest.raises(ValueError, match="uint8"):
        png_bytes(rgb.astype(np.float32))


def test_colormap_endpoints_and_nan() -> None:
    out = colormap(np.array([0.0, 1.0, np.nan, 2.0]))
    assert out.tolist() == [[68, 1, 84], [253, 231, 37], [68, 1, 84], [253, 231, 37]]


def test_continuous_renderers() -> None:
    frames = np.random.default_rng(0).standard_normal((50, 768)).astype(np.float32)
    heat = embedding_heatmap_svg(frames, title="h")
    sim = self_similarity_svg(frames)
    assert "data:image/png;base64," in heat and "768 dims" in heat and ">h<" in heat
    assert "data:image/png;base64," in sim
    with pytest.raises(ValueError, match="expected"):
        embedding_heatmap_svg(frames[0])
    with pytest.raises(ValueError, match="expected"):
        self_similarity_svg(frames.ravel())


def test_embed_matches_encode(tmp_path: Path) -> None:
    from pint_infer.kmeans import nearest_centroid

    tok = tokenizer_for("pint")
    wav = FIXTURES / "speech_vctk_p230_001.wav"
    frames = tok.embed(wav)
    assert frames.ndim == 2 and frames.shape[1] == 768 and frames.dtype == np.float32
    ids = tok.encode(wav, method="kmeans_200")
    assert len(ids) == frames.shape[0]
    # the same frames, discretized by hand, give the same ids
    assert nearest_centroid(frames, tok._centroids(200)).tolist() == ids
    assert tok.frame_stride_ms == 20
    batch = tok.embed_batch([wav, FIXTURES / "synth_clicks.wav"])
    assert np.allclose(batch[0], frames, atol=1e-4)


def test_inputs_shorter_than_one_frame_are_rejected_clearly() -> None:
    import torch

    tok = tokenizer_for("pint")
    for n in (0, 1, 80, 399):
        with pytest.raises(ValueError, match="one frame needs 400"):
            tok.encode(torch.zeros(n), sample_rate=16000)
    assert len(tok.encode(torch.zeros(400), sample_rate=16000)) == 1
    # a 5 ms clip at 48 kHz is still too short after resampling
    with pytest.raises(ValueError, match="nothing to encode"):
        tok.embed(torch.zeros(240), sample_rate=48000)


def test_cli_formats(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from tests.test_parity import MODEL_SOURCES

    model = MODEL_SOURCES["pint"]
    wav = str(FIXTURES / "speech_vctk_p230_001.wav")
    svg = tmp_path / "strip.svg"
    assert cli.main([wav, "--model", model, "--format", "json", "--svg", str(svg)]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["method"] == "kmeans_200" and record["ids"]
    assert svg.read_text().startswith("<svg")

    assert cli.main([wav, "--model", model, "--method", "kmeans_100", "--format", "ids"]) == 0
    assert all(0 <= int(t) < 100 for t in capsys.readouterr().out.split())

    assert cli.main([wav, "--model", model]) == 0
    assert f"{wav} [kmeans_200, {len(record['ids'])} frames]: " in capsys.readouterr().out

    assert cli.main([wav, "--model", model, "--dedup", "--svg", str(svg)]) == 0
    runs_count = len(runs(record["ids"]))
    assert f"{wav} [kmeans_200, {runs_count} runs]: " in capsys.readouterr().out
    assert svg.read_text() == token_strip_svg({wav: record["ids"]}, title="kmeans_200", dedup=True)

    # two recordings with the same file name get one row each
    for folder, fixture in (("x", "speech_vctk_p230_001.wav"), ("y", "synth_sweep.wav")):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "same.wav").write_bytes((FIXTURES / fixture).read_bytes())
    same = [str(tmp_path / "x" / "same.wav"), str(tmp_path / "y" / "same.wav")]
    assert cli.main([*same, "--model", model, "--svg", str(svg)]) == 0
    capsys.readouterr()
    assert all(f">{name}</text>" in svg.read_text() for name in same)

    assert cli.main([wav, "--model", model, "--rle", "8"]) == 0
    rle = json.loads(capsys.readouterr().out)
    assert max(rle["run_lengths"]) <= 8 and len(rle["tokens"]) == len(rle["run_lengths"])


def wav_bytes(tag: int, bits: int, samples: bytes, extensible: bool = False) -> bytes:
    """A two-channel 16 kHz RIFF/WAVE file, with an odd-sized chunk before ``data``."""
    channels, rate, block = 2, 16000, 2 * bits // 8
    fmt = struct.pack(
        "<HHIIHH", 0xFFFE if extensible else tag, channels, rate, rate * block, block, bits
    )
    if extensible:
        fmt += struct.pack("<HHIH", 22, bits, 0b11, tag) + GUID_TAIL
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"LIST" + struct.pack("<I", 3) + b"abc\x00"
    body += b"data" + struct.pack("<I", len(samples)) + samples
    return b"RIFF" + struct.pack("<I", len(body)) + body


def interleaved(scale: float, dtype: str, offset: float = 0.0) -> bytes:
    return (SIGNAL.T * scale + offset).astype(dtype).tobytes()


PCM24 = b"".join(int(v).to_bytes(3, "little", signed=True) for v in (SIGNAL.T * 2**23).ravel())


@pytest.mark.parametrize(
    ("tag", "bits", "samples", "extensible"),
    [
        (1, 8, interleaved(128, "u1", 128), False),
        (1, 16, interleaved(2**15, "<i2"), False),
        (1, 24, PCM24, False),
        (1, 24, PCM24, True),
        (1, 32, interleaved(2**31, "<i4"), False),
        (3, 32, interleaved(1, "<f4"), False),
        (3, 32, interleaved(1, "<f4"), True),
        (3, 64, interleaved(1, "<f8"), False),
    ],
)
def test_read_wav_decodes_uncompressed_formats(
    tmp_path: Path, tag: int, bits: int, samples: bytes, extensible: bool
) -> None:
    path = tmp_path / "x.wav"
    path.write_bytes(wav_bytes(tag, bits, samples, extensible))
    decoded = read_wav(path)
    assert decoded is not None
    wav, rate = decoded
    assert rate == 16000 and wav.dtype == torch.float32
    assert wav.numpy().tolist() == SIGNAL.tolist()


def test_read_wav_leaves_other_files_to_torchaudio(tmp_path: Path) -> None:
    mulaw = tmp_path / "mulaw.wav"
    mulaw.write_bytes(wav_bytes(7, 8, bytes(8)))
    flac = tmp_path / "x.flac"
    flac.write_bytes(b"fLaC" + bytes(40))
    assert read_wav(mulaw) is None and read_wav(flac) is None


@pytest.mark.parametrize("tag", [0x0002, 0x0011])
def test_adpcm_wav_is_rejected(tmp_path: Path, tag: int) -> None:
    """torchaudio returns only part of an ADPCM file, so loading it must fail loudly."""
    path = tmp_path / "adpcm.wav"
    path.write_bytes(wav_bytes(tag, 4, bytes(64)))
    with pytest.raises(ValueError, match="ADPCM WAV.*pcm_s16le"):
        PINTTokenizer._load_audio_file(path)


@pytest.mark.parametrize("error", [ImportError("no torchcodec"), RuntimeError("no FFmpeg")])
def test_wav_needs_no_decoder_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    """Without torchcodec, or with torchcodec but no FFmpeg, .wav still loads and anything
    else fails with an error that names the missing packages."""

    def broken_load(path: str) -> None:
        raise error

    monkeypatch.setattr("torchaudio.load", broken_load)
    wav, rate = PINTTokenizer._load_audio_file(FIXTURES / "speech_vctk_p230_001.wav")
    assert rate == 16000 and wav.shape[0] == 1
    flac = tmp_path / "x.flac"
    flac.write_bytes(b"fLaC" + bytes(40))
    with pytest.raises(ValueError, match="pip install torchcodec"):
        PINTTokenizer._load_audio_file(flac)


def test_readme_strips_are_drawn_from_tokens_json() -> None:
    figures = runpy.run_path(str(EXAMPLES / "make_figures.py"))
    drawn = 0
    for folder, dedup in figures["FIGURES"].items():
        tokens = json.loads((EXAMPLES / folder / "tokens.json").read_text())
        for name, svg in figures["strips"](tokens, dedup).items():
            assert (EXAMPLES / folder / name).read_text() == svg, name
            drawn += 1
    assert drawn == len(list(EXAMPLES.glob("*/tokens_*.svg"))) == 6


def test_example_clips_reproduce_tokens_json() -> None:
    tok = tokenizer_for("pint")
    token_files = sorted(EXAMPLES.glob("*/tokens.json"))
    assert len(token_files) == 2
    for token_file in token_files:
        for clip in json.loads(token_file.read_text()).values():
            wav = token_file.parent / clip["audio"]
            for method in tok.available_methods:
                assert tok.encode(wav, method=method) == clip[f"pint/{method}"], (wav, method)
