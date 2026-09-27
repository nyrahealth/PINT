"""PINT tokenizer: audio in, frame-level features or token ids out.

Per-utterance feature extraction → HuBERT encoder → one 768-d frame per 20 ms →
nearest k-means centroid (``"kmeans_{k}"``).
"""

import json
import re
import struct
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torchaudio
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import HubertModel, Wav2Vec2FeatureExtractor

from pint_infer.kmeans import nearest_centroid

SAMPLE_RATE = 16000
MIN_SAMPLES = 400  # one 20 ms frame: the conv stack's receptive field; shorter inputs have no frame
KMEANS_PATTERN = re.compile(r"^kmeans_(\d+)$")

type Wav = str | Path | np.ndarray | torch.Tensor

# RIFF format tags; WAVE_FORMAT_EXTENSIBLE carries PCM or float as its sub-format
WAV_PCM, WAV_FLOAT, WAV_EXTENSIBLE = 1, 3, 0xFFFE
# Microsoft and IMA ADPCM: torchcodec 0.16 decodes only the first ~30 % of such a file
WAV_ADPCM = {0x0002: "Microsoft ADPCM", 0x0011: "IMA ADPCM"}
# (format tag, bits per sample) -> (numpy dtype, full scale); 24-bit is widened to int32 first
WAV_SAMPLES = {
    (WAV_PCM, 8): ("u1", 128.0),
    (WAV_PCM, 16): ("<i2", 32768.0),
    (WAV_PCM, 24): ("<i4", 2147483648.0),
    (WAV_PCM, 32): ("<i4", 2147483648.0),
    (WAV_FLOAT, 32): ("<f4", 1.0),
    (WAV_FLOAT, 64): ("<f8", 1.0),
}


def read_wav(path: Path) -> tuple[torch.Tensor, int] | None:
    """Decode an uncompressed RIFF/WAVE file to ``(channels, samples)`` float32 in [-1, 1].

    Returns None for any other container or codec, and raises ValueError for ADPCM, which
    torchaudio would silently truncate. This path needs no decoder backend:
    torchaudio >= 2.9 decodes through torchcodec and FFmpeg, and the stdlib ``wave`` module
    reads no float samples.
    """
    fmt, raw = b"", None
    with path.open("rb") as f:
        header = f.read(12)
        if len(header) < 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
            return None
        while raw is None and len(chunk := f.read(8)) == 8:
            name, size = chunk[:4], struct.unpack("<I", chunk[4:])[0]
            if name == b"data":
                raw = f.read(size)
            else:
                body = f.read(size + size % 2)  # chunks are padded to an even size
                if name == b"fmt ":
                    fmt = body[:size]
    if raw is None or len(fmt) < 16:
        return None
    tag, channels, sample_rate, _, _, bits = struct.unpack("<HHIIHH", fmt[:16])
    if tag == WAV_EXTENSIBLE and len(fmt) >= 26:
        tag = struct.unpack("<H", fmt[24:26])[0]
    if tag in WAV_ADPCM:
        raise ValueError(
            f"{path} is {WAV_ADPCM[tag]} WAV, which torchaudio decodes only in part; convert it "
            "to PCM first, e.g. `ffmpeg -i in.wav -c:a pcm_s16le out.wav`"
        )
    if (tag, bits) not in WAV_SAMPLES or channels < 1:
        return None
    dtype, scale = WAV_SAMPLES[(tag, bits)]
    frame_bytes = bits // 8 * channels
    raw = raw[: len(raw) // frame_bytes * frame_bytes]
    if bits == 24:  # the three bytes become the top of an int32, so the sign carries over
        wide = np.zeros((len(raw) // 3, 4), dtype=np.uint8)
        wide[:, 1:] = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        raw = wide.tobytes()
    data = np.frombuffer(raw, dtype=np.dtype(dtype)).astype(np.float32)
    if bits == 8:
        data = data - 128.0
    per_channel = (data / scale).reshape(-1, channels).T
    return torch.from_numpy(np.ascontiguousarray(per_channel)), sample_rate


class PINTTokenizer:
    """Turns 16 kHz audio into PINT token ids (one id per 20 ms frame)."""

    def __init__(
        self,
        hubert: HubertModel,
        feature_extractor: Wav2Vec2FeatureExtractor,
        pint_config: dict[str, Any],
        model_dir: Path,
        device: str = "auto",
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # transformers wraps ``PreTrainedModel.to``, which loses its signature for mypy.
        self.hubert: HubertModel = hubert.to(device).eval()  # type: ignore[arg-type]
        self.feature_extractor = feature_extractor
        self.config = pint_config
        self.model_dir = model_dir
        self.device = device
        self._kmeans_cache: dict[int, np.ndarray] = {}

    @classmethod
    def from_pretrained(
        cls,
        repo_id_or_path: str | Path,
        device: str = "auto",
        revision: str | None = None,
        token: str | None = None,
    ) -> "PINTTokenizer":
        """Load from a Hugging Face repo id (e.g. ``nyralabs/PINT``) or a local
        folder holding the same files.

        ``device`` is ``"auto"`` (CUDA if available), ``"cpu"`` or a torch device string.
        """
        path = Path(repo_id_or_path).expanduser()
        if path.is_dir():
            model_dir = path
        else:
            model_dir = Path(
                snapshot_download(str(repo_id_or_path), revision=revision, token=token)
            )

        pint_config = json.loads((model_dir / "pint_config.json").read_text())
        hubert = HubertModel.from_pretrained(model_dir)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
        return cls(hubert, feature_extractor, pint_config, model_dir, device)

    @property
    def available_methods(self) -> list[str]:
        return [f"kmeans_{k}" for k in sorted(int(k) for k in self.config.get("kmeans", {}))]

    @property
    def frame_stride_ms(self) -> int:
        return int(self.config.get("frame_stride_ms", 20))

    def _centroids(self, k: int) -> np.ndarray:
        """Load the ``(k, 768)`` cluster centers for ``kmeans_{k}``."""
        if k not in self._kmeans_cache:
            files = self.config.get("kmeans", {})
            if str(k) not in files:
                raise ValueError(
                    f"No kmeans_{k} model in this repo; available: {self.available_methods}"
                )
            # pint_config.json comes from the model repo: keep it from escaping model_dir.
            # Check the name, not the resolved path: the Hub cache stores snapshot files
            # as symlinks into its blobs/ directory, which lies outside model_dir.
            name = files[str(k)]
            if Path(name).is_absolute() or Path(name).drive or ".." in Path(name).parts:
                raise ValueError(f"kmeans path '{name}' escapes the model directory")
            centers = load_file(self.model_dir / name)["cluster_centers"].numpy()
            if centers.shape != (k, self.config["hidden_size"]):
                raise ValueError(f"{name}: expected ({k}, {self.config['hidden_size']}) centers")
            self._kmeans_cache[k] = centers
        return self._kmeans_cache[k]

    @staticmethod
    def _load_audio_file(path: str | Path) -> tuple[torch.Tensor, int]:
        decoded = read_wav(Path(path))
        if decoded is not None:
            return decoded
        try:
            wav, sample_rate = torchaudio.load(str(path))
        except (ImportError, RuntimeError) as err:
            # ImportError: torchcodec missing; RuntimeError: torchcodec without FFmpeg, or a
            # file FFmpeg cannot read (the chained error says which)
            raise ValueError(
                f"torchaudio could not decode {path}. Uncompressed .wav needs no extra "
                "packages; other formats need torchcodec (`pip install torchcodec`) and FFmpeg."
            ) from err
        return wav, int(sample_rate)

    def _load_wav(
        self, wav: Wav, sample_rate: int | None, length: int | None = None
    ) -> torch.Tensor:
        """Return mono float32 (T,) at 16 kHz, trimmed to ``length`` samples if given."""
        if isinstance(wav, str | Path):
            data, sr = self._load_audio_file(wav)
            if length is not None:
                raise ValueError("lengths apply to raw arrays/tensors, not file paths")
        else:
            data = torch.as_tensor(wav).detach().cpu().to(torch.float32)
            if sample_rate is None:
                raise ValueError("sample_rate is required when passing a raw array/tensor")
            sr = sample_rate
            if length is not None:
                if not 0 < length <= data.shape[-1]:
                    raise ValueError(
                        f"length {length} out of range for a {data.shape[-1]}-sample input"
                    )
                data = data[..., :length]
        if data.dim() == 1:
            data = data.unsqueeze(0)
        if data.shape[0] > 1:
            data = data.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            data = torchaudio.functional.resample(data, sr, SAMPLE_RATE)
        if data.shape[-1] < MIN_SAMPLES:
            raise ValueError(
                f"input has {data.shape[-1]} samples at 16 kHz but one frame needs "
                f"{MIN_SAMPLES} (25 ms); nothing to encode"
            )
        return data.squeeze(0).to(torch.float32)

    @torch.no_grad()
    def _encode_frames(self, wavs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched encoder forward. Returns (frames (B, T_enc, 768), lengths (B,)).

        The feature extractor normalizes each utterance on its own unpadded signal
        (padding must not leak into the normalization); the padded batch then goes
        through one HubertModel forward with the attention mask.
        """
        audios = [w.detach().cpu().numpy() for w in wavs]
        features = self.feature_extractor(
            audios,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=True,
            padding=True,
        )
        unpadded_lengths = torch.tensor([len(a) for a in audios], dtype=torch.long)
        lengths = torch.as_tensor(
            self.hubert._get_feat_extract_output_lengths(cast("torch.LongTensor", unpadded_lengths))
        )
        output = self.hubert(
            input_values=features.input_values.to(self.device),
            attention_mask=features.attention_mask.to(self.device),
        )
        return output.last_hidden_state, lengths

    def _extract(self, frames: torch.Tensor, method: str) -> torch.Tensor:
        match = KMEANS_PATTERN.match(method)
        if not match:
            raise ValueError(f"Unknown method '{method}'; available: {self.available_methods}")
        centers = self._centroids(int(match.group(1)))
        batch, seq_len, hidden = frames.shape
        flat = frames.reshape(-1, hidden).detach().cpu().numpy().astype(np.float32)
        ids = nearest_centroid(flat, centers)
        return torch.from_numpy(ids).reshape(batch, seq_len)

    def embed_batch(
        self,
        wavs: list[Wav],
        sample_rate: int | None = None,
        lengths: list[int] | None = None,
    ) -> list[np.ndarray]:
        """Continuous encoder output, one ``(T, 768)`` float32 array per utterance.

        These are the frames every ``kmeans_k`` method assigns to its nearest centroid.
        """
        if lengths is not None and len(lengths) != len(wavs):
            raise ValueError(f"got {len(lengths)} lengths for {len(wavs)} inputs")
        loaded = [
            self._load_wav(w, sample_rate, None if lengths is None else lengths[i])
            for i, w in enumerate(wavs)
        ]
        frames, frame_lengths = self._encode_frames(loaded)
        return [
            frames[i, : int(frame_lengths[i].item())].float().cpu().numpy()
            for i in range(len(loaded))
        ]

    def embed(
        self, wav: Wav, sample_rate: int | None = None, length: int | None = None
    ) -> np.ndarray:
        """Continuous ``(T, 768)`` frames for one utterance (see :meth:`embed_batch`)."""
        return self.embed_batch([wav], sample_rate, None if length is None else [length])[0]

    def encode_batch(
        self,
        wavs: list[Wav],
        method: str = "kmeans_200",
        sample_rate: int | None = None,
        lengths: list[int] | None = None,
    ) -> list[list[int]]:
        """Encode several utterances in one padded forward pass.

        Args:
            lengths: valid sample count per item, for callers holding an
                already-padded batch. Padding must be stripped before feature
                extraction, which normalizes per utterance — passing padded audio
                without ``lengths`` changes the tokens.
        """
        if lengths is not None and len(lengths) != len(wavs):
            raise ValueError(f"got {len(lengths)} lengths for {len(wavs)} inputs")
        loaded = [
            self._load_wav(w, sample_rate, None if lengths is None else lengths[i])
            for i, w in enumerate(wavs)
        ]
        frames, frame_lengths = self._encode_frames(loaded)
        token_ids = self._extract(frames, method)
        return [
            token_ids[i, : int(frame_lengths[i].item())].cpu().tolist() for i in range(len(loaded))
        ]

    def encode(
        self,
        wav: Wav,
        method: str = "kmeans_200",
        dedup: bool = False,
        rle: int | None = None,
        sample_rate: int | None = None,
        length: int | None = None,
    ) -> list[int] | tuple[list[int], list[int]]:
        """Encode one utterance.

        Args:
            wav: path to an audio file, or a raw mono waveform (with ``sample_rate``).
            method: ``"kmeans_{k}"``, one of :attr:`available_methods`.
            dedup: collapse consecutive identical ids (mutually exclusive with ``rle``).
            rle: run-length encode with this ``max_run``; returns ``(tokens, run_lengths)``.
            sample_rate: required when ``wav`` is a raw array/tensor.
            length: valid sample count, if ``wav`` is a padded array (see
                :meth:`encode_batch`).

        Returns:
            Token ids (one per 20 ms frame), or ``(tokens, run_lengths)`` when ``rle``.
        """
        if dedup and rle is not None:
            raise ValueError("dedup and rle are mutually exclusive")
        ids = self.encode_batch([wav], method, sample_rate, None if length is None else [length])[0]
        if dedup:
            from pint_infer.postprocess import deduplicate_ids

            return deduplicate_ids(ids)
        if rle is not None:
            from pint_infer.postprocess import run_length_encode

            return run_length_encode(ids, max_run=rle)
        return ids
