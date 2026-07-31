import json
from pathlib import Path

from vectordb_bench.frontend.pages.full_text_search import _parse_result_file


def test_standard_fts_frontend_ignores_filtered_results(tmp_path: Path):
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
                            "additional_parameters": {"fts_filter": {"filter_rate": 0.5}},
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

    rows = _parse_result_file(result_file)

    assert len(rows) == 1
    assert rows[0]["backend"] == "ZillizCloud"
    assert rows[0]["qps"] == 100.0
