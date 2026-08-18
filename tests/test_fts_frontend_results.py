import json
import tomllib
from inspect import signature
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from vectordb_bench.backend.clients import DB
from vectordb_bench.frontend.config.dbCaseConfigs import get_fts_case_items
from vectordb_bench.frontend.pages.full_text_search import (
    _concurrency_rows,
    _draw_filtered_qps_tab,
    _peak_filtered_qps_rows,
    load_full_text_search_rows,
)


def test_oss_opensearch_fts_cases_are_selectable_in_run_test():
    assert all(item.supports_dbs([DB.OSSOpenSearch]) for item in get_fts_case_items())


def test_streamlit_dependency_floor_supports_fts_tab_state_api():
    repo_root = Path(__file__).resolve().parents[1]
    dependencies = tomllib.loads((repo_root / "pyproject.toml").read_text())["project"]["dependencies"]

    assert "streamlit>=1.61,<2" in dependencies
    assert {"default", "key", "on_change"}.issubset(signature(st.tabs).parameters)


def test_frontend_separates_filtered_results_and_expands_concurrency_qps(tmp_path: Path):
    result_file = tmp_path / "result_fts.json"
    result_file.write_text(
        json.dumps(
            {
                "task_label": "fts-results",
                "results": [
                    {
                        "metrics": {
                            "qps": 100.0,
                            "recall": 0.9,
                            "ndcg": 0.8,
                            "mrr": 0.7,
                        },
                        "task_config": {
                            "db": "ZillizCloud",
                            "case_config": {
                                "custom_case": {
                                    "dataset_with_size_type": "HotpotQA Large (5.2M documents)",
                                    "payload_profile": "ids_only",
                                }
                            },
                        },
                    },
                    {
                        "metrics": {
                            "qps": 200.0,
                            "recall": 0.95,
                            "ndcg": 0.85,
                            "mrr": 0.75,
                            "additional_parameters": {
                                "fts_filter": {
                                    "filter_rate": 0.5,
                                    "filter_id_distribution": "affine_permutation_v1",
                                }
                            },
                            "conc_num_list": [60, 80],
                            "conc_qps_list": [1200.0, 1500.0],
                        },
                        "task_config": {
                            "db": "ZillizCloud",
                            "case_config": {
                                "custom_case": {
                                    "dataset_with_size_type": "HotpotQA Large (5.2M documents)",
                                    "payload_profile": "ids_only",
                                    "filter_rate": 0.5,
                                }
                            },
                        },
                    },
                ],
            }
        )
    )

    data = load_full_text_search_rows(tmp_path)
    standard_data = data[~data["is_filtered"]]
    filtered_data = data[data["is_filtered"]]
    concurrency_data = _concurrency_rows(filtered_data)
    peak_data = _peak_filtered_qps_rows(filtered_data)

    assert len(standard_data) == 1
    assert standard_data.iloc[0]["qps"] == 100.0
    assert len(filtered_data) == 1
    assert filtered_data.iloc[0]["filter_rate_label"] == "50%"
    assert "filter_distribution" not in filtered_data.columns
    assert concurrency_data[["concurrency", "qps"]].to_dict("records") == [
        {"concurrency": 60, "qps": 1200.0},
        {"concurrency": 80, "qps": 1500.0},
    ]
    assert peak_data[["filter_rate_label", "concurrency", "qps"]].to_dict("records") == [
        {"filter_rate_label": "50%", "concurrency": 80, "qps": 1500.0}
    ]


def test_checked_in_consolidated_permuted_results_expose_concurrency_qps_for_all_backends():
    repo_root = Path(__file__).resolve().parents[1]
    result_dir = repo_root / "vectordb_bench" / "results" / "FullTextSearch"

    data = load_full_text_search_rows(result_dir)
    filtered_data = data[data["is_filtered"]]
    concurrency_data = _concurrency_rows(filtered_data)
    peak_data = _peak_filtered_qps_rows(filtered_data)

    assert len(data) == 64
    assert len(filtered_data) == 40
    assert len(concurrency_data) == 90
    assert (pd.to_numeric(filtered_data["p99_s"]) > 0).all()
    assert (pd.to_numeric(filtered_data["p95_s"]) > 0).all()
    assert (pd.to_numeric(filtered_data["recall"]) > 0).all()
    assert (pd.to_numeric(filtered_data["ndcg"]) > 0).all()
    assert (pd.to_numeric(filtered_data["mrr"]) > 0).all()
    assert len(peak_data) == 20
    assert set(filtered_data["backend"].astype(str)) == {
        "ElasticSearch",
        "OSSOpenSearch",
        "TurboPuffer",
        "ZillizCloud",
    }
    assert set(concurrency_data["concurrency"]) == {40, 60, 80}
    assert set(peak_data["filter_rate_label"]) == {"50%", "75%", "90%", "95%", "99%"}
    assert set(peak_data["concurrency"]).issubset({60, 80})
    assert set(pd.to_numeric(filtered_data["filter_rate"])) == {0.5, 0.75, 0.9, 0.95, 0.99}
    assert not (result_dir / "ZillizCloud" / "result_20260709_fts_filtered_zillizcloud.json").exists()

    expected_result_files = {
        "ElasticCloud": "result_20260626_fts_standard_elasticcloud.json",
        "OpenSearch": "result_20260708_fts_standard_opensearch.json",
        "TurboPuffer": "result_20260626_fts_standard_turbopuffer.json",
        "ZillizCloud": "result_20260626_fts_standard_zillizcloud.json",
    }
    expected_result_counts = {
        "ElasticCloud": 16,
        "OpenSearch": 16,
        "TurboPuffer": 16,
        "ZillizCloud": 16,
    }
    result_files = sorted(result_dir.glob("*/result_*.json"))
    assert len(result_files) == 4
    assert {path.parent.name: path.name for path in result_files} == expected_result_files

    filtered_results = []
    result_counts = {}
    for result_file in result_files:
        results = json.loads(result_file.read_text())["results"]
        result_counts[result_file.parent.name] = len(results)
        if result_file.parent.name == "ZillizCloud":
            assert all(
                "one persistent segment"
                in json.loads(case_result["task_config"]["db_config"]["note"])["evidence"]["source"]
                for case_result in results
            )
        for case_result in results:
            custom_case = case_result["task_config"]["case_config"].get("custom_case") or {}
            fts_filter = case_result["metrics"].get("additional_parameters", {}).get("fts_filter") or {}
            filter_rate = custom_case.get("filter_rate", fts_filter.get("filter_rate"))
            if filter_rate is not None:
                assert "filter_id_distribution" not in custom_case
                assert fts_filter["filter_id_distribution"] == "affine_permutation_v1"
                filtered_results.append(case_result)

    assert result_counts == expected_result_counts
    assert len(filtered_results) == 40
    expected_serial_fields = {
        "serial_latency_p99",
        "serial_latency_p95",
        "recall",
        "ndcg",
        "mrr",
    }
    for case_result in filtered_results:
        stages = set(case_result["task_config"]["stages"])
        provenance = case_result["metrics"]["additional_parameters"].get("serial_measurement")

        assert {"search_concurrent", "search_serial"}.issubset(stages)
        if provenance is None:
            assert case_result["task_config"]["db"] == "ZillizCloud"
            assert all(case_result["metrics"][field] > 0 for field in expected_serial_fields)
            continue
        assert provenance["composed_from_separate_run"] is True
        assert set(provenance["metric_fields"]) == expected_serial_fields
        assert provenance["source_file"].startswith("result_")
        assert len(provenance["source_sha256"]) == 64
        assert "search_serial" in provenance["source_stages"]
        assert provenance["source_note"]


def test_filtered_qps_chart_groups_backend_bars_by_filter_rate():
    repo_root = Path(__file__).resolve().parents[1]
    result_dir = repo_root / "vectordb_bench" / "results" / "FullTextSearch"
    data = load_full_text_search_rows(result_dir)
    filtered_data = data[data["is_filtered"] & (data["dataset_family"].astype(str) == "MS MARCO")]

    class PlotRecorder:
        figure = None

        def info(self, _message: str) -> None:
            raise AssertionError("Expected filtered QPS chart data")

        def plotly_chart(self, figure: Any, **_kwargs) -> None:
            self.figure = figure

    recorder = PlotRecorder()
    _draw_filtered_qps_tab(recorder, filtered_data)

    assert recorder.figure is not None
    assert recorder.figure.layout.barmode == "group"
    assert {trace.type for trace in recorder.figure.data} == {"bar"}
    assert {trace.name for trace in recorder.figure.data} == {
        "ElasticSearch",
        "OSSOpenSearch",
        "TurboPuffer",
        "ZillizCloud",
    }
    expected_filter_rates = {"50%", "75%", "90%", "95%", "99%"}
    assert all(set(trace.x) == expected_filter_rates for trace in recorder.figure.data)
    assert recorder.figure.layout.xaxis.type == "category"
    assert tuple(recorder.figure.layout.xaxis.categoryarray) == ("50%", "75%", "90%", "95%", "99%")


def test_checked_in_zilliz_standard_results_use_semantic_metrics():
    repo_root = Path(__file__).resolve().parents[1]
    result_dir = repo_root / "vectordb_bench" / "results" / "FullTextSearch"

    data = load_full_text_search_rows(result_dir)
    zilliz_data = data[
        (~data["is_filtered"]) & (data["backend"].astype(str) == "ZillizCloud") & (data["payload"] == "ids_only")
    ]
    expected = {
        "MS MARCO Small": (0.9157, 0.7206, 0.6713),
        "MS MARCO Medium": (0.8261, 0.5298, 0.4572),
        "MS MARCO Large": (0.6283, 0.2763, 0.1889),
        "HotpotQA Small": (0.9225, 0.8459, 0.9485),
        "HotpotQA Medium": (0.8437, 0.7322, 0.8636),
        "HotpotQA Large": (0.7674, 0.6265, 0.7574),
    }

    assert len(zilliz_data) == len(expected)
    for row in zilliz_data.to_dict("records"):
        assert (row["recall"], row["ndcg"], row["mrr"]) == expected[str(row["dataset"])]


def test_checked_in_zilliz_large_standard_results_retain_load_metrics():
    repo_root = Path(__file__).resolve().parents[1]
    result_file = (
        repo_root
        / "vectordb_bench"
        / "results"
        / "FullTextSearch"
        / "ZillizCloud"
        / "result_20260626_fts_standard_zillizcloud.json"
    )
    expected = {
        "HotpotQA Large (5.2M documents)": (5_233_329, 92.5443, 81.9515, 174.4958),
        "MS MARCO Large (8.8M documents)": (8_841_823, 183.6571, 98.8113, 282.4684),
    }

    large_results = {}
    for case_result in json.loads(result_file.read_text())["results"]:
        custom_case = case_result["task_config"]["case_config"].get("custom_case") or {}
        dataset = custom_case.get("dataset_with_size_type")
        if dataset in expected and custom_case.get("filter_rate") is None:
            metrics = case_result["metrics"]
            large_results[dataset] = (
                metrics["inserted_count"],
                metrics["insert_duration"],
                metrics["optimize_duration"],
                metrics["load_duration"],
            )

    assert large_results == expected
