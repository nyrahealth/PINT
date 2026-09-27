"""Token post-processing: deduplication and run-length encoding."""


def deduplicate_ids(ids: list[int]) -> list[int]:
    """Collapse consecutive identical token IDs."""
    if not ids:
        return []
    result = [ids[0]]
    for token_id in ids[1:]:
        if token_id != result[-1]:
            result.append(token_id)
    return result


def run_length_encode(ids: list[int], max_run: int = 64) -> tuple[list[int], list[int]]:
    """Collapse consecutive duplicates and record run lengths.

    Runs exceeding *max_run* are split into multiple entries of the same
    token so that every run-length value fits in ``[1, max_run]``.

    Args:
        ids: Raw token-ID sequence.
        max_run: Maximum run length kept per entry.  Longer runs are split.

    Returns:
        ``(rle_tokens, run_lengths)`` of equal length.
    """
    if max_run < 1:
        # would loop forever in the `while count > max_run` splits below
        raise ValueError(f"max_run must be >= 1, got {max_run}")
    if not ids:
        return [], []

    rle_tokens: list[int] = []
    run_lengths: list[int] = []
    current = ids[0]
    count = 1

    for token_id in ids[1:]:
        if token_id == current:
            count += 1
        else:
            while count > max_run:
                rle_tokens.append(current)
                run_lengths.append(max_run)
                count -= max_run
            rle_tokens.append(current)
            run_lengths.append(count)
            current = token_id
            count = 1

    while count > max_run:
        rle_tokens.append(current)
        run_lengths.append(max_run)
        count -= max_run
    rle_tokens.append(current)
    run_lengths.append(count)

    return rle_tokens, run_lengths
