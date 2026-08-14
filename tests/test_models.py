import json
import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from vectordb_bench import config
from vectordb_bench.backend.clients import DB, IndexType
from vectordb_bench.backend.clients.api import EmptyDBCaseConfig
from vectordb_bench.backend.payload import PayloadProfile
from vectordb_bench.models import CaseConfig, CaseResult, CaseType, Metric, TaskConfig, TestResult
from vectordb_bench.restful.format_res import format_results
from vectordb_bench.results import getLeaderboardDataV2

log = logging.getLogger("vectordb_bench")


def test_performance_case_config_applies_top_level_payload_without_mutating_custom_case():
    custom_case = {}
    case_config = CaseConfig(
        case_id=CaseType.Performance768D100M,
        custom_case=custom_case,
        payload_profile=PayloadProfile.VECTOR,
    )

    assert case_config.case.payload_profile == PayloadProfile.VECTOR
    assert custom_case == {}


def test_performance_case_config_payload_round_trip_and_hash_identity():
    ids_only = CaseConfig(
        case_id=CaseType.Performance768D100M,
        payload_profile=PayloadProfile.IDS_ONLY,
    )
    vector = CaseConfig(
        case_id=CaseType.Performance768D100M,
        payload_profile=PayloadProfile.VECTOR,
    )

    restored = CaseConfig.model_validate(vector.model_dump(mode="json"))

    assert restored.payload_profile == PayloadProfile.VECTOR
    assert restored.case.payload_profile == PayloadProfile.VECTOR
    assert hash(ids_only) != hash(vector)


def test_case_config_rejects_payload_for_non_performance_case():
    with pytest.raises(ValidationError, match="only supported for PerformanceCase"):
        CaseConfig(
            case_id=CaseType.CapacityDim128,
            payload_profile=PayloadProfile.VECTOR,
        )


class TestModels:
    @pytest.mark.skip("runs locally")
    def test_test_result(self):
        result = CaseResult(
            task_config=TaskConfig(
                db=DB.Milvus,
                db_config=DB.Milvus.config(),
                db_case_config=DB.Milvus.case_config_cls(index=IndexType.Flat)(),
                case_config=CaseConfig(case_id=CaseType.Performance10M),
            ),
            metrics=Metric(),
        )

        test_result = TestResult(run_id=10000, results=[result])
        test_result.flush()

        with pytest.raises(ValueError):
            result = TestResult.read_file("nosuchfile.json")

    def test_test_result_read_write(self):
        result_dir = config.RESULTS_LOCAL_DIR
        for json_file in result_dir.rglob("result*.json"):
            res = TestResult.read_file(json_file)
            res.flush()

    def test_test_result_merge(self):
        result_dir = config.RESULTS_LOCAL_DIR
        all_results = []

        first_result = None
        for json_file in result_dir.glob("*.json"):
            res = TestResult.read_file(json_file)

            for cr in res.results:
                all_results.append(cr)

            if not first_result:
                first_result = res

        tr = TestResult(
            run_id=first_result.run_id,
            task_label="standard",
            results=all_results,
        )
        tr.flush()

    def test_test_result_display(self):
        result_dir = config.RESULTS_LOCAL_DIR
        for json_file in result_dir.rglob("result*.json"):
            log.info(json_file)
            res = TestResult.read_file(json_file)
            res.display()


def test_old_result_defaults_large_topk_metrics(tmp_path: Path):
    test_result = _large_topk_test_result(Metric())
    payload = test_result.model_dump_for_output()
    case_config = payload["results"][0]["task_config"]["case_config"]
    metrics = payload["results"][0]["metrics"]
    case_config.pop("payload_profile", None)
    metrics.pop("serial_latency_p50", None)
    metrics.pop("conc_latency_p50_list", None)
    metrics.pop("recall_at", None)
    metrics.pop("payload_profile", None)
    metrics.pop("payload_estimated_bytes_per_query", None)
    result_file = tmp_path / "old-result.json"
    result_file.write_text(json.dumps(payload), encoding="utf-8")

    loaded = TestResult.read_file(result_file)

    assert loaded.results[0].task_config.case_config.payload_profile is None
    assert loaded.results[0].task_config.case_config.case.payload_profile == PayloadProfile.IDS_ONLY
    assert loaded.results[0].metrics.serial_latency_p50 == 0
    assert loaded.results[0].metrics.conc_latency_p50_list == []
    assert loaded.results[0].metrics.recall_at == {}
    assert loaded.results[0].metrics.payload_profile == "ids_only"
    assert loaded.results[0].metrics.payload_estimated_bytes_per_query == 0


def test_large_topk_metrics_round_trip_and_convert_serial_p50(tmp_path: Path):
    test_result = _large_topk_test_result(
        Metric(
            serial_latency_p50=0.25,
            conc_latency_p50_list=[0.3],
            recall_at={100: 0.9, 1_000: 0.8},
            payload_profile="vector",
            payload_estimated_bytes_per_query=3_092_000_000,
        ),
        payload_profile=PayloadProfile.VECTOR,
    )
    result_file = tmp_path / "large-topk-result.json"
    result_file.write_text(json.dumps(test_result.model_dump_for_output()), encoding="utf-8")

    loaded = TestResult.read_file(result_file, trans_unit=True)

    metrics = loaded.results[0].metrics
    assert loaded.results[0].task_config.case_config.payload_profile == PayloadProfile.VECTOR
    assert metrics.serial_latency_p50 == 250
    assert metrics.conc_latency_p50_list == [0.3]
    assert metrics.recall_at == {100: 0.9, 1_000: 0.8}
    assert metrics.payload_profile == "vector"
    assert metrics.payload_estimated_bytes_per_query == 3_092_000_000


def test_rest_formatter_exports_large_topk_metrics():
    test_result = _large_topk_test_result(
        Metric(
            serial_latency_p50=0.25,
            conc_latency_p50_list=[0.3],
            recall_at={100: 0.9},
            payload_profile="vector",
            payload_estimated_bytes_per_query=3_092_000_000,
        ),
        payload_profile=PayloadProfile.VECTOR,
    )

    formatted = format_results([test_result], task_label="large-topk")[0]

    assert formatted["serial_latency_p50"] == 0.25
    assert formatted["conc_latency_p50_list"] == [0.3]
    assert formatted["recall_at"] == {100: 0.9}
    assert formatted["payload_profile"] == "vector"
    assert formatted["payload_estimated_bytes_per_query"] == 3_092_000_000


def test_legacy_leaderboard_exports_payload_profile(monkeypatch: pytest.MonkeyPatch):
    captured = {}
    result = _large_topk_test_result(
        Metric(qps=1, recall=1, payload_profile="vector"),
        payload_profile=PayloadProfile.VECTOR,
    ).results[0]

    monkeypatch.setattr(getLeaderboardDataV2, "get_standard_2025_results", lambda: [result])
    monkeypatch.setattr(
        getLeaderboardDataV2,
        "save_to_json",
        lambda data, file_name: captured.setdefault(str(file_name), data),
    )

    getLeaderboardDataV2.main()

    performance_rows = next(rows for rows in captured.values() if rows)
    assert performance_rows[0]["payload_profile"] == "vector"


def _large_topk_test_result(
    metric: Metric,
    payload_profile: PayloadProfile | None = None,
) -> TestResult:
    return TestResult(
        run_id="large-topk",
        task_label="large-topk",
        results=[
            CaseResult(
                task_config=TaskConfig(
                    db=DB.Test,
                    db_config=DB.Test.config_cls(),
                    db_case_config=EmptyDBCaseConfig(),
                    case_config=CaseConfig(
                        case_id=CaseType.Performance768D100M,
                        k=1_000_000,
                        payload_profile=payload_profile,
                    ),
                ),
                metrics=metric,
            )
        ],
    )
