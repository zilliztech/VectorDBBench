import json
from pathlib import Path
from typing import Any

import pandas as pd

from vectordb_bench.frontend.pages.full_text_search import (
    _concurrency_rows,
    _draw_filtered_qps_tab,
    _peak_filtered_qps_rows,
    load_full_text_search_rows,
)


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
                                    "filter_id_distribution": "permuted",
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
    assert filtered_data.iloc[0]["filter_distribution"] == "Permuted"
    assert concurrency_data[["concurrency", "qps"]].to_dict("records") == [
        {"concurrency": 60, "qps": 1200.0},
        {"concurrency": 80, "qps": 1500.0},
    ]
    assert peak_data[["filter_rate_label", "concurrency", "qps"]].to_dict("records") == [
        {"filter_rate_label": "50%", "concurrency": 80, "qps": 1500.0}
    ]


def test_checked_in_permuted_results_expose_concurrency_qps_for_all_backends():
    repo_root = Path(__file__).resolve().parents[1]
    result_dir = repo_root / "vectordb_bench" / "results" / "FullTextSearch"

    data = load_full_text_search_rows(result_dir)
    filtered_data = data[data["is_filtered"]]
    concurrency_data = _concurrency_rows(filtered_data)
    peak_data = _peak_filtered_qps_rows(filtered_data)

    assert len(filtered_data) == 40
    assert len(concurrency_data) == 80
    assert len(peak_data) == 20
    assert set(filtered_data["backend"].astype(str)) == {
        "ElasticSearch",
        "OSSOpenSearch",
        "TurboPuffer",
        "ZillizCloud",
    }
    assert set(filtered_data["filter_distribution"]) == {"Permuted"}
    assert set(concurrency_data["concurrency"]) == {60, 80}
    assert set(peak_data["filter_rate_label"]) == {"50%", "75%", "90%", "95%", "99%"}
    assert set(peak_data["concurrency"]).issubset({60, 80})
    assert set(pd.to_numeric(filtered_data["filter_rate"])) == {0.5, 0.75, 0.9, 0.95, 0.99}
    assert not (result_dir / "ZillizCloud" / "result_20260709_fts_filtered_zillizcloud.json").exists()


def test_filtered_qps_chart_groups_filter_rate_bars_by_backend():
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
    assert {trace.name for trace in recorder.figure.data} == {"50%", "75%", "90%", "95%", "99%"}
    assert {backend for trace in recorder.figure.data for backend in trace.x} == {
        "ElasticSearch",
        "OSSOpenSearch",
        "TurboPuffer",
        "ZillizCloud",
    }


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
