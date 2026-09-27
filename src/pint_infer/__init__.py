"""PINT inference: wav in, token ids (or continuous frames) out.

Usage:
    from pint_infer import PINTTokenizer

    tok = PINTTokenizer.from_pretrained("nyralabs/PINT")  # or a local folder
    ids = tok.encode("speech.wav")                                # kmeans_200, one id per 20 ms
    ids = tok.encode("speech.wav", method="kmeans_500", dedup=True)
    frames = tok.embed("speech.wav")                              # (T, 768) float32
"""

from pint_infer.postprocess import deduplicate_ids, run_length_encode
from pint_infer.tokenizer import PINTTokenizer
from pint_infer.viz import (
    embedding_heatmap_svg,
    format_runs,
    self_similarity_svg,
    token_strip_svg,
)

__version__ = "0.1.0"
__all__ = [
    "PINTTokenizer",
    "__version__",
    "deduplicate_ids",
    "embedding_heatmap_svg",
    "format_runs",
    "run_length_encode",
    "self_similarity_svg",
    "token_strip_svg",
]
