import numpy as np
import pytest
from pint_infer.kmeans import nearest_centroid
from pint_infer.postprocess import deduplicate_ids, run_length_encode


def test_dedup_empty() -> None:
    assert deduplicate_ids([]) == []


def test_dedup_collapses_runs() -> None:
    assert deduplicate_ids([3, 3, 3, 7, 7, 3, 1]) == [3, 7, 3, 1]


def test_rle_empty() -> None:
    assert run_length_encode([]) == ([], [])


def test_rle_basic() -> None:
    tokens, runs = run_length_encode([5, 5, 5, 2, 9, 9])
    assert tokens == [5, 2, 9]
    assert runs == [3, 1, 2]


def test_rle_splits_long_runs() -> None:
    tokens, runs = run_length_encode([4] * 150, max_run=64)
    assert tokens == [4, 4, 4]
    assert runs == [64, 64, 22]
    assert sum(runs) == 150


def test_rle_reconstructs() -> None:
    ids = [1, 1, 2, 2, 2, 3, 1, 1, 1, 1]
    tokens, runs = run_length_encode(ids, max_run=3)
    rebuilt = [t for t, r in zip(tokens, runs, strict=True) for _ in range(r)]
    assert rebuilt == ids


@pytest.mark.parametrize("max_run", [0, -1])
def test_rle_rejects_non_positive_max_run(max_run: int) -> None:
    """These used to spin forever in the run-splitting loop."""
    with pytest.raises(ValueError, match="max_run must be >= 1"):
        run_length_encode([1, 1, 1], max_run=max_run)


def test_nearest_centroid_assigns_and_breaks_ties_low() -> None:
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    x = np.array([[0.1, 0.0], [0.9, 0.1], [0.0, 5.0]], dtype=np.float32)
    assert nearest_centroid(x, centers).tolist() == [0, 1, 2]
    # exactly equidistant from centers 1 and 2 -> lowest index wins, as in sklearn
    assert nearest_centroid(np.array([[1.0, 1.0]], dtype=np.float32), centers).tolist() == [1]


def test_nearest_centroid_matches_bruteforce_on_random_data() -> None:
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((37, 16)).astype(np.float32)
    x = rng.standard_normal((256, 16)).astype(np.float32)
    brute = np.argmin(((x[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
    assert nearest_centroid(x, centers).tolist() == brute.tolist()


def test_nearest_centroid_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        nearest_centroid(np.zeros((4, 8), dtype=np.float32), np.zeros((3, 7), dtype=np.float32))
    with pytest.raises(ValueError, match="shape mismatch"):
        nearest_centroid(np.zeros(8, dtype=np.float32), np.zeros((3, 8), dtype=np.float32))
