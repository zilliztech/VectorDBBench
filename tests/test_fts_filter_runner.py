import threading

from vectordb_bench.backend.cases import CaseLabel
from vectordb_bench.backend.data_source import DatasetSource
from vectordb_bench.backend.dataset import FtsDocument, FtsFilterIdDistribution, FtsQuery
from vectordb_bench.backend.filter import non_filter
from vectordb_bench.backend.payload import PayloadProfile
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

        def prepare(self, source, filters=None, filter_id_distribution=None):
            self.calls.append((source, filters, filter_id_distribution))

    class Case:
        label = CaseLabel.FullTextSearchPerformance
        is_multitenant = False
        dataset = Dataset()
        filters = filter_obj
        filter_id_distribution = FtsFilterIdDistribution.Sequential

    config_obj = type("Config", (), {"stages": [TaskStage.LOAD]})()
    runner = CaseRunner.construct(ca=Case(), config=config_obj, dataset_source=DatasetSource.S3)
    init_calls = []
    monkeypatch.setattr(CaseRunner, "init_db", lambda self, drop_old=True: init_calls.append(drop_old))

    runner._pre_run(drop_old=False)

    assert runner.ca.dataset.calls == [
        (DatasetSource.S3, filter_obj, FtsFilterIdDistribution.Sequential),
    ]
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


class DummyDb:
    name = "dummy"

    def supports_payload_profile(self, payload_profile):
        return True

    def supports_document_payload_profile(self, payload_profile):
        return True

    def search_documents(self, query, k, payload_profile=None):
        return []


class FtsConcurrencyConfig:
    num_concurrency = [60, 80]
    concurrency_duration = 30
    concurrency_timeout = 300
    serial_cooldown = 0


class FtsCaseConfig:
    k = 10
    concurrency_search_config = FtsConcurrencyConfig()


def test_fts_search_runners_use_full_queries_for_concurrency_and_filtered_queries_for_recall():
    class Dataset:
        queries_data = [
            FtsQuery(query_id="q1", text="full query one"),
            FtsQuery(query_id="q2", text="full query two"),
        ]
        gt_data = [{"d1": 1}, {"d2": 1}]
        recall_queries_data = [FtsQuery(query_id="q2", text="full query two")]
        recall_gt_data = [{"d2": 1}]
        recall_skipped = False
        recall_skip_reason = None

    class Case:
        label = CaseLabel.FullTextSearchPerformance
        dataset = Dataset()
        filters = non_filter
        payload_profile = PayloadProfile.IDS_ONLY

    config_obj = type(
        "Config",
        (),
        {
            "stages": [TaskStage.SEARCH_SERIAL, TaskStage.SEARCH_CONCURRENT],
            "case_config": FtsCaseConfig(),
        },
    )()
    runner = CaseRunner.construct(ca=Case(), config=config_obj, db=DummyDb())

    runner._init_fts_search_runner()

    assert runner.search_runner.test_data == ["full query one", "full query two"]
    assert runner.serial_search_runner.test_data == ["full query two"]
    assert runner.serial_search_runner.ground_truth == [{"d2": 1}]


def test_fts_perf_metric_marks_recall_skipped_without_blocking_case(monkeypatch):
    class Dataset:
        filter_stats = {
            "filter_field": "filter_id",
            "filter_value": 99,
            "filtered_query_count": 0,
        }
        queries_data = [
            FtsQuery(query_id="q1", text="full query one"),
            FtsQuery(query_id="q2", text="full query two"),
        ]
        recall_queries_data = []
        recall_skipped = True
        recall_skip_reason = "no_positive_qrels_after_filter"

    class Case:
        label = CaseLabel.FullTextSearchPerformance
        dataset = Dataset()

    config_obj = type(
        "Config",
        (),
        {
            "stages": [TaskStage.SEARCH_SERIAL],
            "case_config": FtsCaseConfig(),
        },
    )()
    runner = CaseRunner.construct(ca=Case(), config=config_obj)
    monkeypatch.setattr(CaseRunner, "_init_search_runners", lambda self: None)

    metric = runner._run_perf_case(drop_old=False)

    assert metric.recall == 0.0
    assert metric.ndcg == 0.0
    assert metric.mrr == 0.0
    assert metric.additional_parameters["fts_recall"] == {
        "skipped": True,
        "reason": "no_positive_qrels_after_filter",
        "serial_query_count": 0,
        "full_query_count": 2,
    }
