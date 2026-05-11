import pathlib
from datetime import datetime

import ujson

from vectordb_bench import config
from vectordb_bench.backend.cases import CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.frontend.config.dbPrices import DB_DBLABEL_TO_PRICE
from vectordb_bench.interface import benchMarkRunner
from vectordb_bench.models import ResultLabel, TestResult

task_label_to_code = {
    ResultLabel.FAILED: -1,
    ResultLabel.OUTOFRANGE: -2,
    ResultLabel.NORMAL: 1,
}


def format_time(ts: float) -> str:
    default_standard_test_time = datetime(2023, 8, 1)
    t = datetime.fromtimestamp(ts)
    t = max(t, default_standard_test_time)
    return t.strftime("%Y-%m")


def main():
    all_results: list[TestResult] = benchMarkRunner.get_results()

    if all_results is not None:
        data = [
            {
                "db": d.task_config.db.value,
                "db_label": d.task_config.db_config.db_label,
                "db_name": d.task_config.db_name,
                "case": d.task_config.case_config.case_id.case_name(),
                "qps": d.metrics.qps,
                "latency": d.metrics.serial_latency_p99,
                "recall": d.metrics.recall,
                "label": task_label_to_code[d.label],
                "note": d.task_config.db_config.note,
                "version": d.task_config.db_config.version,
                "test_time": format_time(test_result.timestamp),
            }
            for test_result in all_results
            if "standard" in test_result.task_label
            for d in test_result.results
            if d.task_config.case_config.case_id not in {CaseType.CapacityDim128, CaseType.CapacityDim960}
            if d.task_config.db != DB.ZillizCloud or test_result.timestamp >= datetime(2024, 1, 1).timestamp()
        ]

        # compute qp$
        for d in data:
            db = d["db"]
            db_label = d["db_label"]
            qps = d["qps"]
            price = DB_DBLABEL_TO_PRICE.get(db, {}).get(db_label, 0)
            d["qp$"] = (qps / price * 3600) if price > 0 else 0.0

        with pathlib.Path(config.RESULTS_LOCAL_DIR, "leaderboard.json").open("w") as f:
            ujson.dump(data, f)


if __name__ == "__main__":
    main()
