import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import cache
from itertools import islice

import numpy as np

log = logging.getLogger(__name__)

RECALL_CUTOFFS = (100, 1_000, 10_000, 100_000, 1_000_000)


@dataclass
class Metric:
    """result metrics"""

    # for load cases
    max_load_count: int = 0

    # for both performace and streaming cases
    insert_duration: float = 0.0
    optimize_duration: float = 0.0
    load_duration: float = 0.0  # insert + optimize

    # for performance cases
    qps: float = 0.0
    serial_latency_p99: float = 0.0
    serial_latency_p95: float = 0.0
    serial_latency_p50: float = 0.0
    recall: float = 0.0
    recall_at: dict[int, float] = field(default_factory=dict)
    ndcg: float = 0.0
    conc_num_list: list[int] = field(default_factory=list)
    conc_qps_list: list[float] = field(default_factory=list)
    conc_latency_p99_list: list[float] = field(default_factory=list)
    conc_latency_p95_list: list[float] = field(default_factory=list)
    conc_latency_p50_list: list[float] = field(default_factory=list)
    conc_latency_avg_list: list[float] = field(default_factory=list)
    payload_profile: str = "ids_only"
    payload_estimated_bytes_per_query: int = 0

    inserted_count: int = 0
    insert_rows_per_second: float = 0.0
    insert_completion_seconds: float = 0.0
    searchable_after_insert_seconds: float = 0.0
    indexed_after_searchable_seconds: float = 0.0
    additional_parameters: dict = field(default_factory=dict)

    # for streaming cases
    st_ideal_insert_duration: int = 0
    st_search_stage_list: list[int] = field(default_factory=list)
    st_search_time_list: list[float] = field(default_factory=list)
    st_max_qps_list_list: list[float] = field(default_factory=list)
    st_recall_list: list[float] = field(default_factory=list)
    st_ndcg_list: list[float] = field(default_factory=list)
    st_serial_latency_p99_list: list[float] = field(default_factory=list)
    st_serial_latency_p95_list: list[float] = field(default_factory=list)
    st_conc_failed_rate_list: list[float] = field(default_factory=list)

    # for streaming cases - concurrent latency data per stage
    st_conc_num_list_list: list[list[int]] = field(default_factory=list)
    st_conc_qps_list_list: list[list[float]] = field(default_factory=list)
    st_conc_latency_p99_list_list: list[list[float]] = field(default_factory=list)
    st_conc_latency_p95_list_list: list[list[float]] = field(default_factory=list)
    st_conc_latency_avg_list_list: list[list[float]] = field(default_factory=list)


QURIES_PER_DOLLAR_METRIC = "QP$ (Quries per Dollar)"
LOAD_DURATION_METRIC = "load_duration"
SERIAL_LATENCY_P99_METRIC = "serial_latency_p99"
SERIAL_LATENCY_P95_METRIC = "serial_latency_p95"
SERIAL_LATENCY_P50_METRIC = "serial_latency_p50"
MAX_LOAD_COUNT_METRIC = "max_load_count"
QPS_METRIC = "qps"
RECALL_METRIC = "recall"

metric_unit_map = {
    LOAD_DURATION_METRIC: "s",
    SERIAL_LATENCY_P99_METRIC: "ms",
    SERIAL_LATENCY_P95_METRIC: "ms",
    SERIAL_LATENCY_P50_METRIC: "ms",
    MAX_LOAD_COUNT_METRIC: "K",
    QURIES_PER_DOLLAR_METRIC: "K",
}

lower_is_better_metrics = [
    LOAD_DURATION_METRIC,
    SERIAL_LATENCY_P99_METRIC,
    SERIAL_LATENCY_P95_METRIC,
    SERIAL_LATENCY_P50_METRIC,
]

metric_order = [
    QPS_METRIC,
    RECALL_METRIC,
    LOAD_DURATION_METRIC,
    SERIAL_LATENCY_P99_METRIC,
    SERIAL_LATENCY_P95_METRIC,
    SERIAL_LATENCY_P50_METRIC,
    MAX_LOAD_COUNT_METRIC,
]


def isLowerIsBetterMetric(metric: str) -> bool:
    return metric in lower_is_better_metrics


def calc_recall(count: int, ground_truth: Iterable[int], got: Iterable[int]) -> float:
    if count <= 0:
        return 0.0

    ground_truth_ids = set(ground_truth)
    hits = {result for result in islice(got, count) if result in ground_truth_ids}
    return len(hits) / count


@cache
def get_ideal_dcg(k: int):
    if k <= 0:
        return 0.0

    ranks = np.arange(2, k + 2, dtype=np.float64)
    return float(np.sum(1 / np.log2(ranks)))


def _build_ground_truth_ranks(ground_truth: Iterable[int]) -> dict[int, int]:
    ranks = {}
    for rank, neighbor_id in enumerate(ground_truth):
        ranks.setdefault(neighbor_id, rank)
    return ranks


def _calc_ndcg_from_ranks(ground_truth_ranks: dict[int, int], got: Iterable[int], ideal_dcg: float) -> float:
    if ideal_dcg <= 0:
        return 0.0

    dcg = 0.0
    for got_id in set(got):
        rank = ground_truth_ranks.get(got_id)
        if rank is not None:
            dcg += 1 / np.log2(rank + 2)
    return dcg / ideal_dcg


def calc_ndcg(ground_truth: Iterable[int], got: Iterable[int], ideal_dcg: float) -> float:
    return _calc_ndcg_from_ranks(_build_ground_truth_ranks(ground_truth), got, ideal_dcg)


def calc_vector_metrics(
    count: int,
    ground_truth: Iterable[int],
    got: Sequence[int],
    recall_cutoffs: Iterable[int] = RECALL_CUTOFFS,
) -> tuple[float, float, dict[int, float]]:
    if count <= 0:
        return 0.0, 0.0, {}

    ground_truth_ranks = _build_ground_truth_ranks(islice(ground_truth, count))
    unique_results = set(islice(got, count))
    recall = len(unique_results.intersection(ground_truth_ranks)) / count
    ndcg = _calc_ndcg_from_ranks(ground_truth_ranks, unique_results, get_ideal_dcg(count))

    recall_at = {}
    for cutoff in recall_cutoffs:
        if cutoff <= 0 or cutoff > count:
            continue
        hits = {
            result
            for result in islice(got, cutoff)
            if (rank := ground_truth_ranks.get(result)) is not None and rank < cutoff
        }
        recall_at[cutoff] = len(hits) / cutoff

    return recall, ndcg, recall_at


def calc_recall_fts(k: int, ground_truth: list[int], got: list[int]) -> float:
    if not ground_truth or k <= 0:
        return 0.0
    gt_set = set(ground_truth)
    hits = gt_set & set(got[:k])
    return calc_recall(len(gt_set), gt_set, hits)
