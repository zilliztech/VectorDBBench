from enum import StrEnum
from dataclasses import dataclass


class DisplayedMetric(StrEnum):
    db = "db"
    db_name = "db_name"
    search_stage = "search_stage"
    search_time = "search_time"
    qps = "qps"
    recall = "recall"
    ndcg = "ndcg"
    adjusted_recall = "adjusted_recall"
    adjusted_ndcg = "adjusted_ndcg"
    latency_p99 = "latency_p99"
    latency_p95 = "latency_p95"
    # st_ideal_insert_duration = "st_ideal_insert_duration"
    # st_search_time_list = "st_search_time_list"
    insert_duration = "insert_duration"
    optimize_duration = "optimize_duration"


@dataclass
class StreamingData:
    db: str
    db_name: str
    search_stage: int
    search_time: float
    qps: float
    recall: float
    ndcg: float
    adjusted_recall: float
    adjusted_ndcg: float
    latency_p99: float
    latency_p95: float
    ideal_insert_duration: int
    insert_duration: float
    optimize_duration: float

    @property
    def optimized(self) -> bool:
        return self.search_stage > 100


def get_streaming_data(data) -> list[StreamingData]:
    return [
        StreamingData(
            db=d["db"],
            db_name=d["db_name"],
            search_stage=search_stage,
            search_time=d["st_search_time_list"][i],
            qps=d["st_max_qps_list_list"][i],
            recall=d["st_recall_list"][i],
            ndcg=d["st_ndcg_list"][i],
            adjusted_recall=round(d["st_recall_list"][i] / min(search_stage, 100) * 100, 4),
            adjusted_ndcg=round(d["st_ndcg_list"][i] / min(search_stage, 100) * 100, 4),
            latency_p99=round(d["st_serial_latency_p99_list"][i] * 1000, 2),
            latency_p95=round(d["st_serial_latency_p95_list"][i] * 1000, 2) if "st_serial_latency_p95_list" in d and i < len(d["st_serial_latency_p95_list"]) else 0.0,
            ideal_insert_duration=d["st_ideal_insert_duration"],
            insert_duration=d["insert_duration"],
            optimize_duration=d["optimize_duration"],
        )
        for d in data
        for i, search_stage in enumerate(d["st_search_stage_list"])
    ]
