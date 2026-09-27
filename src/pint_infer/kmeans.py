"""Nearest-centroid k-means assignment.

Codebooks ship as safetensors cluster centers. Assignment is plain nearest centroid — the
computation ``sklearn.cluster.KMeans.predict`` performs — so inference never unpickles a
downloaded file and does not depend on a scikit-learn version.
"""

import numpy as np


def nearest_centroid(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Assign each row of ``x`` (n, d) to its closest row of ``centers`` (k, d).

    Uses the same squared-norm expansion as scikit-learn
    (``|x|² - 2·x·cᵀ + |c|²``) in float32, so ids match ``KMeans.predict``
    bit for bit. Ties resolve to the lowest centroid index, as in sklearn.
    """
    if x.ndim != 2 or centers.ndim != 2 or x.shape[1] != centers.shape[1]:
        raise ValueError(f"shape mismatch: x {x.shape} vs centers {centers.shape}")
    x32 = np.ascontiguousarray(x, dtype=np.float32)
    c32 = np.ascontiguousarray(centers, dtype=np.float32)
    distances = (
        (x32 * x32).sum(axis=1, keepdims=True) - 2.0 * (x32 @ c32.T) + (c32 * c32).sum(axis=1)
    )
    return np.asarray(np.argmin(distances, axis=1), dtype=np.int64)
