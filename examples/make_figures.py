"""Re-render the README's token strips from the clips in this folder.

Encodes every clip with the released model and rewrites the six ``tokens_*.svg``. The PINT rows
use the fresh ids; the HuBERT and WavLM rows use the ids stored in ``tokens.json``, because their
codebooks are not distributed. Run it from the repository root with
``python examples/make_figures.py``; when the model reproduces the stored ids, ``git status``
shows no change. ``PINT_MODEL`` overrides the Hub id.
"""

import json
import os
from pathlib import Path
from typing import Any

from pint_infer import PINTTokenizer, token_strip_svg

EXAMPLES = Path(__file__).parent
FIGURES = {"01_parallel_speakers": True, "02_augmentations": False}  # folder -> deduplicated
METHOD = "kmeans_200"
# tokens.json key -> (figure title, codebook label)
MODELS = {
    "pint": ("PINT", METHOD),
    "hubert": ("HuBERT-base L9", "k-means 200"),
    "wavlm": ("WavLM-base+ L12", "k-means 200"),
}


def strips(tokens: dict[str, dict[str, Any]], dedup: bool) -> dict[str, str]:
    """``{file name: svg}`` of one example folder's token strips, drawn from its ``tokens.json``."""
    what = "deduplicated (one band per run)" if dedup else "one band per run of identical ids"
    suffix = "_dedup" if dedup else ""
    return {
        f"tokens_{key}_{METHOD}{suffix}.svg": token_strip_svg(
            {label: clip[f"{key}/{METHOD}"] for label, clip in tokens.items()},
            title=f"{title} · {codebook} · {what}",
            dedup=dedup,
        )
        for key, (title, codebook) in MODELS.items()
    }


def main() -> None:
    tok = PINTTokenizer.from_pretrained(os.environ.get("PINT_MODEL", "nyralabs/PINT"))
    for folder, dedup in FIGURES.items():
        tokens = json.loads((EXAMPLES / folder / "tokens.json").read_text())
        for clip in tokens.values():
            ids = tok.encode(EXAMPLES / folder / clip["audio"], method=METHOD)
            verdict = "match" if ids == clip[f"pint/{METHOD}"] else "DIFFER from"
            print(f"{folder}/{clip['audio']}: ids {verdict} tokens.json")
            clip[f"pint/{METHOD}"] = ids
        for name, svg in strips(tokens, dedup).items():
            (EXAMPLES / folder / name).write_text(svg)


if __name__ == "__main__":
    main()
