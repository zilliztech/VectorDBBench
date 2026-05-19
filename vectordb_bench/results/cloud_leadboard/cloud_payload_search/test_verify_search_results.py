import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import verify_search_results


def write_raw(path: Path, run_id: str, db_label: str, payload: str = "ids_only") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "results": [
                    {
                        "metrics": {
                            "qps": 12.3456,
                            "serial_latency_p99": 0.22,
                            "serial_latency_p95": 0.11,
                            "recall": 0.951,
                            "ndcg": 0.9617,
                            "conc_num_list": [60, 80],
                            "conc_qps_list": [10.1234, 12.3456],
                            "conc_latency_p99_list": [0.9, 1.1],
                            "conc_latency_p95_list": [0.7, 0.8],
                            "conc_latency_avg_list": [0.5, 0.6],
                            "payload_profile": payload,
                            "payload_estimated_bytes_per_query": 2000,
                        },
                        "task_config": {
                            "db_config": {"db_label": db_label},
                            "case_config": {"custom_case": {"payload_profile": payload}},
                        },
                    }
                ],
            }
        )
    )


class VerifySearchResultsTest(unittest.TestCase):
    def test_verifies_manifest_raw_json_and_report_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            serial_raw = root / "cloud_payload_search/raw_results/zilliz_cloud_tiered_4cu/unfiltered/na/ids_only/serial_recall/result.json"
            concurrent_raw = root / "cloud_payload_search/raw_results/zilliz_cloud_tiered_4cu/unfiltered/na/ids_only/concurrent_qps/result.json"
            write_raw(serial_raw, "run-serial", "label-serial")
            write_raw(concurrent_raw, "run-concurrent", "label-concurrent")
            (root / "cloud_payload_search/raw_results/manifest.jsonl").parent.mkdir(parents=True, exist_ok=True)
            (root / "cloud_payload_search/raw_results/manifest.jsonl").write_text(
                json.dumps(
                    {
                        "case_id": "zilliz_cloud_tiered_4cu__unfiltered__na__ids_only",
                        "product": "zilliz_cloud_tiered_4cu",
                        "filter_type": "unfiltered",
                        "filter_rate": "na",
                        "payload_profile": "ids_only",
                        "phase": "serial_recall",
                        "raw_json": str(serial_raw.relative_to(root)),
                        "run_id": "run-serial",
                        "db_label": "label-serial",
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "case_id": "zilliz_cloud_tiered_4cu__unfiltered__na__ids_only",
                        "product": "zilliz_cloud_tiered_4cu",
                        "filter_type": "unfiltered",
                        "filter_rate": "na",
                        "payload_profile": "ids_only",
                        "phase": "concurrent_qps",
                        "raw_json": str(concurrent_raw.relative_to(root)),
                        "run_id": "run-concurrent",
                        "db_label": "label-concurrent",
                    }
                )
                + "\n"
            )
            (root / "cloud_payload_search/single_tenant_100m_search.md").write_text(
                """## Zilliz Cloud Tiered 4CU
### Unfiltered Search
| Payload | Recall | QPS @60 | QPS @80 | Max QPS | Avg latency @60/@80 | P95 @60/@80 | P99 @60/@80 | Status |
|---|---:|---:|---:|---:|---|---|---|---|
| IDs only | [0.951](raw_results/zilliz_cloud_tiered_4cu/unfiltered/na/ids_only/serial_recall/result.json) | 10.1234 | 12.3456 | [12.3456](raw_results/zilliz_cloud_tiered_4cu/unfiltered/na/ids_only/concurrent_qps/result.json) | 0.5000s / 0.6000s | 0.7000s / 0.8000s | 0.9000s / 1.1000s | measured |
"""
            )

            errors = verify_search_results.verify(root)

            self.assertEqual(errors, [])

    def test_reports_metric_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "cloud_payload_search/raw_results/zilliz_cloud_tiered_4cu/unfiltered/na/ids_only/concurrent_qps/result.json"
            write_raw(raw, "run-1", "label-1")
            (root / "cloud_payload_search/raw_results/manifest.jsonl").parent.mkdir(parents=True, exist_ok=True)
            (root / "cloud_payload_search/raw_results/manifest.jsonl").write_text(
                json.dumps(
                    {
                        "case_id": "zilliz_cloud_tiered_4cu__unfiltered__na__ids_only",
                        "product": "zilliz_cloud_tiered_4cu",
                        "filter_type": "unfiltered",
                        "filter_rate": "na",
                        "payload_profile": "ids_only",
                        "phase": "concurrent_qps",
                        "raw_json": str(raw.relative_to(root)),
                        "run_id": "run-1",
                        "db_label": "label-1",
                    }
                )
                + "\n"
            )
            (root / "cloud_payload_search/single_tenant_100m_search.md").write_text(
                """## Zilliz Cloud Tiered 4CU
### Unfiltered Search
| Payload | Recall | QPS @60 | QPS @80 | Max QPS | Avg latency @60/@80 | P95 @60/@80 | P99 @60/@80 | Status |
|---|---:|---:|---:|---:|---|---|---|---|
| IDs only | pending | 99.0000 | 12.3456 | [12.3456](raw_results/zilliz_cloud_tiered_4cu/unfiltered/na/ids_only/concurrent_qps/result.json) | 0.5000s / 0.6000s | 0.7000s / 0.8000s | 0.9000s / 1.1000s | measured |
"""
            )

            errors = verify_search_results.verify(root)

            self.assertTrue(any("QPS @60" in error for error in errors))

    def test_verifies_non_default_concurrency_headers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "cloud_payload_search/raw_results/pinecone_serverless/unfiltered/na/ids_only/concurrent_qps/result.json"
            write_raw(raw, "run-1", "label-1")
            payload = json.loads(raw.read_text())
            metrics = payload["results"][0]["metrics"]
            metrics["conc_num_list"] = [3, 4]
            metrics["conc_qps_list"] = [5.3979, 4.1521]
            metrics["conc_latency_avg_list"] = [0.0863, 0.0691]
            metrics["conc_latency_p95_list"] = [0.1523, 0.1443]
            metrics["conc_latency_p99_list"] = [0.2210, 0.1801]
            metrics["qps"] = 5.3979
            raw.write_text(json.dumps(payload))
            (root / "cloud_payload_search/raw_results/manifest.jsonl").parent.mkdir(parents=True, exist_ok=True)
            (root / "cloud_payload_search/raw_results/manifest.jsonl").write_text(
                json.dumps(
                    {
                        "case_id": "pinecone_serverless__unfiltered__na__ids_only",
                        "product": "pinecone_serverless",
                        "filter_type": "unfiltered",
                        "filter_rate": "na",
                        "payload_profile": "ids_only",
                        "phase": "concurrent_qps",
                        "raw_json": str(raw.relative_to(root)),
                        "run_id": "run-1",
                        "db_label": "label-1",
                    }
                )
                + "\n"
            )
            (root / "cloud_payload_search/single_tenant_100m_search.md").write_text(
                """## Pinecone Serverless
### Unfiltered Search
| Payload | Recall | QPS @3 | QPS @4 | Max QPS | Avg latency @3/@4 | P95 @3/@4 | P99 @3/@4 | Status |
|---|---:|---:|---:|---:|---|---|---|---|
| IDs only | pending | 9.9999 | 4.1521 | [5.3979](raw_results/pinecone_serverless/unfiltered/na/ids_only/concurrent_qps/result.json) | 0.0863s / 0.0691s | 0.1523s / 0.1443s | 0.2210s / 0.1801s | measured; 429 observed |
"""
            )

            errors = verify_search_results.verify(root)

            self.assertTrue(any("QPS @3" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
