import logging
import time

import numpy as np

from vectordb_bench.backend.filter import Filter, non_filter
from vectordb_bench.backend.payload import PayloadProfile

from ... import config
from ..clients import api

log = logging.getLogger(__name__)


class ColdWarmSearchRunner:
    def __init__(
        self,
        db: api.VectorDB,
        test_data: list[list[float]],
        k: int = 100,
        filters: Filter = non_filter,
        payload_profile: PayloadProfile = PayloadProfile.IDS_ONLY,
        query_count: int = 1000,
    ):
        if query_count <= 0:
            msg = "query_count must be positive"
            raise ValueError(msg)
        if len(test_data) < query_count:
            msg = f"query_count={query_count} exceeds test_data size={len(test_data)}"
            raise ValueError(msg)

        self.db = db
        self.k = k
        self.filters = filters
        self.payload_profile = payload_profile
        self.query_count = query_count
        if not self.db.supports_payload_profile(self.payload_profile):
            msg = f"{self.db.name} does not support payload_profile={self.payload_profile.value}"
            raise NotImplementedError(msg)

        self.test_data = [
            query.tolist() if isinstance(query, np.ndarray) else query for query in test_data[:query_count]
        ]

    def _search_embedding(self, emb: list[float]) -> list[int]:
        if self.payload_profile == PayloadProfile.IDS_ONLY:
            return self.db.search_embedding(emb, self.k)
        return self.db.search_embedding(emb, self.k, payload_profile=self.payload_profile)

    def _get_db_search_res(self, emb: list[float], retry_idx: int = 0) -> list[int]:
        try:
            results = self._search_embedding(emb)
        except Exception as e:
            log.warning(f"Cold/warm search failed, retry_idx={retry_idx}, Exception: {e}")
            if retry_idx < config.MAX_SEARCH_RETRY:
                return self._get_db_search_res(emb=emb, retry_idx=retry_idx + 1)

            msg = f"Cold/warm search failed and retried more than {config.MAX_SEARCH_RETRY} times"
            raise RuntimeError(msg) from e

        return results

    @staticmethod
    def _latency_stats(latencies: list[float]) -> dict[str, float]:
        return {
            "first_query_latency": round(float(latencies[0]), 4),
            "p99_latency": round(float(np.percentile(latencies, 99)), 4),
            "p95_latency": round(float(np.percentile(latencies, 95)), 4),
            "avg_latency": round(float(np.mean(latencies)), 4),
        }

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float:
        if denominator == 0:
            return 0.0
        return round(float(numerator / denominator), 4)

    def _ratio_stats(self, cold_stats: dict[str, float], warm_stats: dict[str, float]) -> dict[str, float]:
        return {
            "first_query_latency_ratio": self._safe_ratio(
                cold_stats["first_query_latency"],
                warm_stats["first_query_latency"],
            ),
            "p99_latency_ratio": self._safe_ratio(cold_stats["p99_latency"], warm_stats["p99_latency"]),
            "p95_latency_ratio": self._safe_ratio(cold_stats["p95_latency"], warm_stats["p95_latency"]),
            "avg_latency_ratio": self._safe_ratio(cold_stats["avg_latency"], warm_stats["avg_latency"]),
        }

    def _run_pass(self, pass_name: str) -> dict[str, float]:
        latencies = []
        for emb in self.test_data:
            start = time.perf_counter()
            self._get_db_search_res(emb)
            latencies.append(time.perf_counter() - start)

            if len(latencies) % 100 == 0:
                log.debug(f"{pass_name} search_count={len(latencies):3}, latest_latency={latencies[-1]}")

        stats = self._latency_stats(latencies)
        log.info(
            f"{pass_name} search pass: "
            f"queries={len(latencies)}, "
            f"first_query_latency={stats['first_query_latency']}, "
            f"avg_latency={stats['avg_latency']}, "
            f"p99={stats['p99_latency']}, "
            f"p95={stats['p95_latency']}"
        )
        return stats

    def run(self) -> dict[str, dict[str, float]]:
        with self.db.init():
            self.db.prepare_filter(self.filters)
            cold_stats = self._run_pass("cold")
            warm_stats = self._run_pass("warm")

        return {
            "cold_stats": cold_stats,
            "warm_stats": warm_stats,
            "cold_warm_ratio": self._ratio_stats(cold_stats, warm_stats),
        }
