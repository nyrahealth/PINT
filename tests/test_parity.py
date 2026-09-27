"""Exact-match parity against reference token ids (``golden_tokens.json``) computed on CPU
by an independent implementation of the encoder.

The model defaults to the Hugging Face repo; point PINT_MODEL at a local folder to
test that instead. A model that cannot be loaded is a FAILURE, not a
skip, so the suite never passes vacuously. Set PINT_ALLOW_MISSING_MODELS=1 to skip instead.
"""

import json
import os
from pathlib import Path

import pytest
from pint_infer import PINTTokenizer, deduplicate_ids, run_length_encode

FIXTURES = Path(__file__).parent / "fixtures"
GOLDENS = json.loads((Path(__file__).parent / "golden_tokens.json").read_text())

MODEL_SOURCES = {
    "pint": os.environ.get("PINT_MODEL", "nyralabs/PINT"),
}
ALLOW_MISSING = os.environ.get("PINT_ALLOW_MISSING_MODELS") == "1"
# reference method name -> pint_infer method name
METHOD_MAP = {"kmeans_200": "kmeans_200", "kmeans_100": "kmeans_100"}
# the parity matrix this suite must cover, independent of golden_tokens.json — so
# deleting a case from the goldens fails the gate instead of shrinking it
REQUIRED_CASES = {
    ("pint", "kmeans_100"),
    ("pint", "kmeans_200"),
}
EXPECTED_FIXTURE_COUNT = 6
SPEECH = FIXTURES / "speech_vctk_p230_001.wav"

_tokenizers: dict[str, PINTTokenizer] = {}


def tokenizer_for(model: str) -> PINTTokenizer:
    source = MODEL_SOURCES[model]
    if source not in _tokenizers:
        try:
            _tokenizers[source] = PINTTokenizer.from_pretrained(source)
        except Exception as err:
            if ALLOW_MISSING:
                pytest.skip(f"cannot load {source}: {err}")
            raise AssertionError(
                f"cannot load {model} from '{source}': {err}. Parity cannot be proven "
                f"without the artifact; set PINT_ALLOW_MISSING_MODELS=1 to skip instead."
            ) from err
    return _tokenizers[source]


def model_method_cases() -> list[tuple[str, str]]:
    return [(model, method) for model, methods in GOLDENS["models"].items() for method in methods]


def test_golden_file_covers_the_required_matrix() -> None:
    """Guards the gate itself: goldens must not silently lose cases or fixtures."""
    assert set(model_method_cases()) == REQUIRED_CASES
    assert len(sorted(FIXTURES.glob("*.wav"))) == EXPECTED_FIXTURE_COUNT
    assert GOLDENS["device"] == "cpu"
    for model, methods in GOLDENS["models"].items():
        for method, sections in methods.items():
            assert set(sections) == {"vocab_size", "single", "batched"}, f"{model}/{method}"
            vocab = sections["vocab_size"]
            for section in ("single", "batched"):
                cases = sections[section]
                assert len(cases) == EXPECTED_FIXTURE_COUNT, f"{model}/{method}/{section}"
                for name, ids in cases.items():
                    assert (FIXTURES / name).is_file(), name
                    assert ids, f"{model}/{method}/{section}/{name} is empty"
                    assert all(isinstance(i, int) and 0 <= i < vocab for i in ids), (
                        f"{model}/{method}/{section}/{name} has ids outside [0, {vocab})"
                    )


@pytest.mark.parametrize("model", sorted(MODEL_SOURCES))
def test_artifact_config(model: str) -> None:
    tok = tokenizer_for(model)
    assert tok.config["base_model"] == "facebook/hubert-base-ls960"
    assert tok.config["sample_rate"] == 16000
    assert tok.config["hidden_size"] == 768


@pytest.mark.parametrize(("model", "golden_method"), model_method_cases())
def test_single_utterance_exact_parity(model: str, golden_method: str) -> None:
    tok = tokenizer_for(model)
    golden = GOLDENS["models"][model][golden_method]["single"]
    for wav_name, expected in golden.items():
        ids = tok.encode(FIXTURES / wav_name, method=METHOD_MAP[golden_method])
        assert ids == expected, f"{model}/{golden_method}/{wav_name}"


@pytest.mark.parametrize(("model", "golden_method"), model_method_cases())
def test_padded_batch_exact_parity(model: str, golden_method: str) -> None:
    tok = tokenizer_for(model)
    golden = GOLDENS["models"][model][golden_method]["batched"]
    names = list(golden)
    batch_ids = tok.encode_batch([FIXTURES / n for n in names], method=METHOD_MAP[golden_method])
    for name, ids in zip(names, batch_ids, strict=True):
        assert ids == golden[name], f"{model}/{golden_method}/{name} (batched)"


def test_dedup_and_rle_match_postprocessed_goldens() -> None:
    tok = tokenizer_for("pint")
    golden = GOLDENS["models"]["pint"]["kmeans_200"]["single"]
    for wav_name, expected in golden.items():
        assert tok.encode(FIXTURES / wav_name, dedup=True) == deduplicate_ids(expected)
        assert tok.encode(FIXTURES / wav_name, rle=64) == run_length_encode(expected, 64)


def test_bad_arguments_rejected() -> None:
    tok = tokenizer_for("pint")
    with pytest.raises(ValueError, match="mutually exclusive"):
        tok.encode(FIXTURES / "synth_clicks.wav", dedup=True, rle=64)
    with pytest.raises(ValueError, match="Unknown method"):
        tok.encode(FIXTURES / "synth_clicks.wav", method="quantum")
    with pytest.raises(ValueError, match="No kmeans_7 model"):
        tok.encode(FIXTURES / "synth_clicks.wav", method="kmeans_7")
    with pytest.raises(ValueError, match="sample_rate is required"):
        tok.encode([0.0] * 16000)


def test_padded_input_with_lengths_matches_unpadded() -> None:
    """Padding must be stripped before feature extraction, which normalizes per utterance."""
    import torch

    tok = tokenizer_for("pint")
    wav_path = SPEECH
    data, sr = PINTTokenizer._load_audio_file(wav_path)
    mono = data.squeeze(0)
    expected = tok.encode(mono, method="kmeans_200", sample_rate=sr)

    padded = torch.cat([mono, torch.zeros(mono.numel())])
    assert tok.encode(padded, method="kmeans_200", sample_rate=sr, length=mono.numel()) == expected
    # and the trap this closes: padding without lengths does change the tokens
    assert tok.encode(padded, method="kmeans_200", sample_rate=sr) != expected

    batch = tok.encode_batch(
        [padded, padded], method="kmeans_200", sample_rate=sr, lengths=[mono.numel()] * 2
    )
    assert batch == [expected, expected]

    with pytest.raises(ValueError, match="lengths for"):
        tok.encode_batch([padded], method="kmeans_200", sample_rate=sr, lengths=[1, 2])
    with pytest.raises(ValueError, match="out of range"):
        tok.encode(mono, method="kmeans_200", sample_rate=sr, length=mono.numel() + 1)
    with pytest.raises(ValueError, match="not file paths"):
        tok.encode(wav_path, method="kmeans_200", length=100)


def test_kmeans_artifacts_are_pickle_free() -> None:
    """Inference must never unpickle a downloaded file."""
    for model in sorted(MODEL_SOURCES):
        tok = tokenizer_for(model)
        assert tok.config["kmeans"], model
        for name in tok.config["kmeans"].values():
            assert name.endswith(".safetensors"), name
        assert not sorted(tok.model_dir.glob("*.joblib")), f"{model} still ships pickles"
        k = min(int(key) for key in tok.config["kmeans"])
        assert tok._centroids(k).shape == (k, 768)


def test_kmeans_path_traversal_rejected() -> None:
    tok = tokenizer_for("pint")
    tok._kmeans_cache.clear()
    tok.config["kmeans"]["99"] = "../../../etc/passwd"
    try:
        with pytest.raises(ValueError, match="escapes the model directory"):
            tok._centroids(99)
    finally:
        del tok.config["kmeans"]["99"]


def test_kmeans_loads_through_symlinked_snapshot(tmp_path: Path) -> None:
    """The Hub cache stores snapshot files as symlinks into ../../blobs/; the traversal
    guard must accept them whether or not the suite runs on a local folder."""
    tok = tokenizer_for("pint")
    blobs, snapshot = tmp_path / "blobs", tmp_path / "snapshots" / "rev"
    blobs.mkdir()
    snapshot.mkdir(parents=True)
    for name in tok.config["kmeans"].values():
        (blobs / name).write_bytes((tok.model_dir / name).read_bytes())
        try:
            (snapshot / name).symlink_to(Path("../..") / "blobs" / name)
        except OSError as err:  # e.g. Windows without symlink privilege
            pytest.skip(f"cannot create symlinks: {err}")
    # same device as the cached tokenizer: the constructor moves the shared encoder in place
    linked = PINTTokenizer(
        tok.hubert, tok.feature_extractor, tok.config, snapshot, device=tok.device
    )
    for k in (int(key) for key in tok.config["kmeans"]):
        assert (linked._centroids(k) == tok._centroids(k)).all()


def test_resampling_and_raw_input() -> None:
    import torch
    import torchaudio

    tok = tokenizer_for("pint")
    wav_path = SPEECH
    expected = tok.encode(wav_path, method="kmeans_200")

    data, sr = PINTTokenizer._load_audio_file(wav_path)
    assert tok.encode(data.squeeze(0), method="kmeans_200", sample_rate=sr) == expected

    upsampled = torchaudio.functional.resample(data, sr, 24000)
    ids_24k = tok.encode(upsampled.squeeze(0), method="kmeans_200", sample_rate=24000)
    # resampling round-trip is lossy; sequences must agree closely but not exactly
    matches = sum(a == b for a, b in zip(ids_24k, expected, strict=False))
    assert matches / max(len(expected), 1) > 0.8

    stereo = torch.stack([data.squeeze(0), data.squeeze(0)])
    assert tok.encode(stereo, method="kmeans_200", sample_rate=sr) == expected


RELEASED_CODEBOOKS = {50, 100, 200, 250, 500, 1000}


def test_released_codebook_set() -> None:
    """The released k set, and every codebook loads with the right shape."""
    tok = tokenizer_for("pint")
    assert {int(k) for k in tok.config["kmeans"]} == RELEASED_CODEBOOKS
    for k in RELEASED_CODEBOOKS:
        assert tok._centroids(k).shape == (k, 768)
