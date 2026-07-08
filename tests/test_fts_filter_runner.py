import threading

from vectordb_bench.backend.cases import CaseLabel
from vectordb_bench.backend.data_source import DatasetSource
from vectordb_bench.backend.dataset import FtsDocument
from vectordb_bench.backend.runner.concurrent_runner import ConcurrentInsertRunner
from vectordb_bench.backend.task_runner import CaseRunner
from vectordb_bench.models import TaskStage


def test_concurrent_insert_runner_fts_batch_includes_filter_ids():
    runner = ConcurrentInsertRunner.__new__(ConcurrentInsertRunner)
    runner._deadline = None
    runner._stop_event = None
    runner._iter_lock = threading.Lock()
    runner._prefetched_fts_batch = None
    runner._dataset_iter = iter(
        [
            [
                FtsDocument(doc_id="d1", text="alpha", filter_id=0),
                FtsDocument(doc_id="d2", text="beta", filter_id=5),
            ]
        ]
    )

    assert runner._next_fts_batch() == {
        "texts": ["alpha", "beta"],
        "doc_ids": ["d1", "d2"],
        "filter_ids": [0, 5],
    }


def test_fts_pre_run_passes_filters_to_dataset(monkeypatch):
    filter_obj = object()

    class Dataset:
        def __init__(self):
            self.calls = []

        def prepare(self, source, filters=None):
            self.calls.append((source, filters))

    class Case:
        label = CaseLabel.FullTextSearchPerformance
        is_multitenant = False
        dataset = Dataset()
        filters = filter_obj

    config_obj = type("Config", (), {"stages": [TaskStage.LOAD]})()
    runner = CaseRunner.construct(ca=Case(), config=config_obj, dataset_source=DatasetSource.S3)
    init_calls = []
    monkeypatch.setattr(CaseRunner, "init_db", lambda self, drop_old=True: init_calls.append(drop_old))

    runner._pre_run(drop_old=False)

    assert runner.ca.dataset.calls == [(DatasetSource.S3, filter_obj)]
    assert init_calls == [False]


def test_fts_perf_metric_includes_dataset_filter_stats():
    class Dataset:
        filter_stats = {
            "filter_field": "filter_id",
            "filter_value": 90,
            "filtered_query_count": 12,
        }

    class Case:
        label = CaseLabel.FullTextSearchPerformance
        dataset = Dataset()

    config_obj = type("Config", (), {"stages": []})()
    runner = CaseRunner.construct(ca=Case(), config=config_obj)

    metric = runner._run_perf_case(drop_old=False)

    assert metric.additional_parameters["fts_filter"] == Dataset.filter_stats
