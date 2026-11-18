"""Exact filtered top-k by brute force.

Distances are computed in float64 and ranked by (distance, id), so equal distances are
broken by the smaller id and the result never depends on chunking or thread order.
"""

from dataclasses import dataclass

import numpy as np

from ..clients.api import MetricType

QUERY_BATCH = 256
TRAIN_CHUNK = 16_384


def prepare_vectors(vectors: np.ndarray, metric_type: MetricType) -> np.ndarray:
    """float64 copy in the space the metric ranks in (unit vectors for COSINE)."""
    out = np.asarray(vectors, dtype=np.float64)
    if metric_type == MetricType.COSINE:
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        out = out / np.where(norms == 0, 1.0, norms)
    return out


def pairwise_distance(queries: np.ndarray, train: np.ndarray, metric_type: MetricType) -> np.ndarray:
    """Smaller is closer: squared L2, or negated inner product for IP / COSINE."""
    dots = queries @ train.T
    if metric_type == MetricType.L2:
        sq = (queries * queries).sum(axis=1)[:, None] - 2.0 * dots + (train * train).sum(axis=1)[None, :]
        return np.maximum(sq, 0.0)
    return -dots


@dataclass(frozen=True)
class TopK:
    ids: list[np.ndarray]  # per query, ascending by (distance, id); shorter than k if fewer rows match
    distances: list[np.ndarray]
    boundary_ties: int  # queries whose k-th and (k+1)-th distances are equal, so another valid answer exists


def exact_topk(
    queries: np.ndarray,
    train: np.ndarray,
    mask: np.ndarray,
    k: int,
    metric_type: MetricType,
) -> TopK:
    """Top-k of `train` rows where `mask` is True, for each query. Inputs come from prepare_vectors."""
    candidate_ids = np.flatnonzero(mask)
    num_queries = len(queries)
    if candidate_ids.size == 0:
        empty = [np.empty(0, dtype=np.int64) for _ in range(num_queries)]
        return TopK(ids=empty, distances=[np.empty(0) for _ in range(num_queries)], boundary_ties=0)

    keep = k + 1  # one extra slot to tell whether the k-th place is tied
    all_ids: list[np.ndarray] = []
    all_dists: list[np.ndarray] = []
    boundary_ties = 0
    for q_start in range(0, num_queries, QUERY_BATCH):
        q = queries[q_start : q_start + QUERY_BATCH]
        best_d = np.full((len(q), 0), np.inf)
        best_i = np.empty((len(q), 0), dtype=np.int64)
        # Chunks go in ascending id order, so every id kept so far is smaller than every id in the
        # next chunk; a stable sort on distance alone then ranks ties by id.
        for c_start in range(0, candidate_ids.size, TRAIN_CHUNK):
            chunk_ids = candidate_ids[c_start : c_start + TRAIN_CHUNK]
            d = pairwise_distance(q, train[chunk_ids], metric_type)
            merged_d = np.concatenate([best_d, d], axis=1)
            merged_i = np.concatenate([best_i, np.broadcast_to(chunk_ids, d.shape)], axis=1)
            order = np.argsort(merged_d, axis=1, kind="stable")[:, :keep]
            best_d = np.take_along_axis(merged_d, order, axis=1)
            best_i = np.take_along_axis(merged_i, order, axis=1)
        width = min(k, best_d.shape[1])
        if best_d.shape[1] > k:
            boundary_ties += int(np.count_nonzero(best_d[:, k - 1] == best_d[:, k]))
        all_ids.extend(best_i[:, :width])
        all_dists.extend(best_d[:, :width])
    return TopK(ids=all_ids, distances=all_dists, boundary_ties=boundary_ties)
