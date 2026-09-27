"""``pint-infer AUDIO [AUDIO ...]`` — print PINT tokens for audio files."""

import argparse
import json
import sys
from pathlib import Path

from pint_infer.postprocess import deduplicate_ids, run_length_encode
from pint_infer.tokenizer import PINTTokenizer
from pint_infer.viz import format_runs, token_strip_svg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pint-infer", description=__doc__)
    parser.add_argument("audio", nargs="+", type=Path, help="audio files (any rate / channels)")
    parser.add_argument("--model", default="nyralabs/PINT", help="HF repo id or local folder")
    parser.add_argument(
        "--method", default=None, help="kmeans_<k> (default: kmeans_200 if the model has it)"
    )
    parser.add_argument("--device", default="auto")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dedup", action="store_true", help="collapse repeated ids")
    group.add_argument("--rle", type=int, metavar="MAX_RUN", help="run-length encode")
    parser.add_argument("--format", choices=["ids", "runs", "json"], default="runs")
    parser.add_argument("--svg", type=Path, help="also write a token strip of all inputs here")
    args = parser.parse_args(argv)

    tok = PINTTokenizer.from_pretrained(args.model, device=args.device)
    method = args.method or (
        "kmeans_200" if "kmeans_200" in tok.available_methods else tok.available_methods[0]
    )
    # frame-level ids by the path as given, so files that share a name keep their own row
    strips: dict[str, list[int]] = {}
    for path in args.audio:
        ids = tok.encode_batch([path], method=method)[0]
        strips[str(path)] = ids
        if args.rle is not None:
            tokens, run_lengths = run_length_encode(ids, max_run=args.rle)
            print(json.dumps({"file": str(path), "tokens": tokens, "run_lengths": run_lengths}))
            continue
        if args.dedup:
            ids = deduplicate_ids(ids)
        if args.format == "json":
            print(json.dumps({"file": str(path), "method": method, "ids": ids}))
        elif args.format == "ids":
            print(" ".join(map(str, ids)))
        else:
            unit = "runs" if args.dedup else "frames"
            print(f"{path} [{method}, {len(ids)} {unit}]: {format_runs(ids)}")
    if args.svg and strips:
        args.svg.write_text(
            token_strip_svg(strips, frame_ms=tok.frame_stride_ms, title=method, dedup=args.dedup)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
