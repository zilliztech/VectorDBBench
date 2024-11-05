from vectordb_bench import config
import ujson
import pathlib
from vectordb_bench.backend.cases import CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.frontend.config.dbPrices import DB_DBLABEL_TO_PRICE
from vectordb_bench.interface import benchMarkRunner
from vectordb_bench.models import ResultLabel, TestResult
from datetime import datetime

taskLabelToCode = {
    ResultLabel.FAILED: -1,
    ResultLabel.OUTOFRANGE: -2,
    ResultLabel.NORMAL: 1,
}


def format_time(ts: float) -> str:
    default_standard_test_time = datetime(2023, 8, 1)
    t = datetime.fromtimestamp(ts)
    if t < default_standard_test_time:
        t = default_standard_test_time
    return t.strftime("%Y-%m")


def main():
    allResults: list[TestResult] = benchMarkRunner.get_results()

    if allResults is not None:
        data = [
            {
                "db": d.task_config.db.value,
                "db_label": d.task_config.db_config.db_label,
                "db_name": d.task_config.db_name,
                "case": d.task_config.case_config.case_id.case_name(),
                "qps": d.metrics.qps,
                "latency": d.metrics.serial_latency_p99,
                "recall": d.metrics.recall,
                "label": taskLabelToCode[d.label],
                "note": d.task_config.db_config.note,
                "version": d.task_config.db_config.version,
                "test_time": format_time(test_result.timestamp),
            }
            for test_result in allResults
            if "standard" in test_result.task_label
            for d in test_result.results
            if d.task_config.case_config.case_id != CaseType.CapacityDim128
            and d.task_config.case_config.case_id != CaseType.CapacityDim960
            if d.task_config.db != DB.ZillizCloud
            or test_result.timestamp >= datetime(2024, 1, 1).timestamp()
        ]

        # compute qp$
        for d in data:
            db = d["db"]
            db_label = d["db_label"]
            qps = d["qps"]
            price = DB_DBLABEL_TO_PRICE.get(db, {}).get(db_label, 0)
            d["qp$"] = (qps / price * 3600) if price > 0 else 0.0

        with open(pathlib.Path(config.RESULTS_LOCAL_DIR, "leaderboard.json"), "w") as f:
            ujson.dump(data, f)


if __name__ == "__main__":
    main()
