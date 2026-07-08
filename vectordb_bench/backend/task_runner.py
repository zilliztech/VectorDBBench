import concurrent
import hashlib
import logging
import re
import time
import traceback
from enum import Enum, auto

import numpy as np
from pydantic import PrivateAttr

from .. import config
from ..base import BaseModel
from ..metric import Metric
from ..models import PerformanceTimeoutError, TaskConfig, TaskStage
from . import utils
from .cases import Case, CaseLabel, StreamingPerformanceCase
from .clients import DB, MetricType, api
from .data_source import DatasetSource
from .runner import (
    ColdWarmSearchRunner,
    ConcurrentInsertRunner,
    MultiProcessingSearchRunner,
    ReadWriteRunner,
    SerialInsertRunner,
    SerialSearchRunner,
)
from .utils import kill_proc_tree
from .workload import WorkloadKind

log = logging.getLogger(__name__)


class RunningStatus(Enum):
    PENDING = auto()
    FINISHED = auto()


class CaseRunner(BaseModel):
    """DataSet, filter_rate, db_class with db config

    Fields:
        run_id(str): run_id of this case runner,
            indicating which task does this case belong to.
        config(TaskConfig): task configs of this case runner.
        ca(Case): case for this case runner.
        status(RunningStatus): RunningStatus of this case runner.

        db(api.VectorDB): The vector database for this case runner.
    """

    run_id: str
    config: TaskConfig
    ca: Case
    status: RunningStatus
    dataset_source: DatasetSource

    db: api.VectorDB | None = None
    test_emb: list[list[float]] | None = None
    test_texts: list[str] | None = None
    serial_search_runner: SerialSearchRunner | None = None
    search_runner: MultiProcessingSearchRunner | None = None
    final_search_runner: MultiProcessingSearchRunner | None = None
    read_write_runner: ReadWriteRunner | None = None
    cold_warm_search_runner: ColdWarmSearchRunner | None = None

    _fts_manifest_report: dict = PrivateAttr(default_factory=dict)

    def __eq__(self, obj: any):
        if isinstance(obj, CaseRunner):
            key = self.load_reuse_key()
            return key is not None and key == obj.load_reuse_key()
        return False

    def __hash__(self) -> int:
        """Hash method to maintain consistency with __eq__ method."""
        return hash(self.load_reuse_key())

    def load_reuse_key(self) -> tuple | None:
        if self.ca.label != CaseLabel.Performance:
            return None
        return (
            self.config.db.value,
            self._db_config_hash_key(),
            self._db_case_config_hash_key(),
            self._collection_name_hash_key(),
            self._dataset_hash_key(),
            self.ca.with_scalar_labels,
            self.ca.is_multitenant,
            self._multitenant_routing_hash_key(),
        )

    @classmethod
    def _hashable_value(cls, value: object) -> object:
        if isinstance(value, dict):
            hashable = tuple(sorted((str(k), cls._hashable_value(v)) for k, v in value.items()))
        elif isinstance(value, (list, tuple)):
            hashable = tuple(cls._hashable_value(v) for v in value)
        elif isinstance(value, (set, frozenset)):
            hashable = tuple(sorted((cls._hashable_value(v) for v in value), key=repr))
        elif isinstance(value, Enum):
            hashable = value.value
        elif hasattr(value, "model_dump"):
            hashable = cls._hashable_value(value.model_dump(mode="json"))
        elif hasattr(value, "get_secret_value"):
            hashable = value.get_secret_value()
        else:
            hashable = value
        return hashable

    def _db_config_hash_key(self) -> object:
        db_config = self.config.db_config
        if hasattr(db_config, "to_dict"):
            return self._hashable_value(db_config.to_dict())
        return self._hashable_value(db_config)

    def _db_case_config_hash_key(self) -> object:
        return self._hashable_value(self.config.db_case_config)

    def _collection_name_hash_key(self) -> str | None:
        return self._doris_collection_name()

    def _dataset_hash_key(self) -> object:
        return self._hashable_value(self.ca.dataset.data)

    def _multitenant_routing_hash_key(self) -> tuple | None:
        if not self.ca.is_multitenant:
            return None
        return (
            getattr(self.ca, "tenant_count", None),
            getattr(self.ca, "tenant_prefix", None),
            getattr(self.ca, "tenant_id_width", None),
            getattr(self.ca, "tenant_distribution", None),
        )

    def _doris_collection_name(self) -> str | None:
        if self.config.db != DB.Doris:
            return None
        case_type_name = self.config.case_config.case_id.name
        base = f"{case_type_name.lower()}"
        base = re.sub(r"[^a-z0-9_]+", "_", base).strip("_")
        if len(base) > 63:
            h = hashlib.md5(base.encode(), usedforsecurity=False).hexdigest()[:6]
            base = f"{base[:(63-7)]}_{h}"
        return base

    def display(self) -> dict:
        dataset_include = {
            "name": True,
            "size": True,
            "label": True,
            "metric_type": True,
        }
        if self.ca.label != CaseLabel.FullTextSearchPerformance:
            dataset_include["dim"] = True
        c_dict = self.ca.dict(
            include={
                "label": True,
                "name": True,
                "filters": True,
                "dataset": {
                    "data": dataset_include,
                },
            },
        )
        c_dict["db"] = self.config.db_name
        return c_dict

    @property
    def normalize(self) -> bool:
        assert self.db
        if self.is_fts:
            return False
        return self.db.need_normalize_cosine() and self.ca.dataset.data.metric_type == MetricType.COSINE

    @property
    def workload_kind(self) -> WorkloadKind:
        if getattr(self.ca, "label", None) == CaseLabel.FullTextSearchPerformance:
            return WorkloadKind.FULL_TEXT
        return WorkloadKind.VECTOR

    @property
    def is_fts(self) -> bool:
        return self.workload_kind == WorkloadKind.FULL_TEXT

    def init_db(self, drop_old: bool = True) -> None:
        db_cls = self.config.db.init_cls
        # Compose a compact, case-unique collection/table name for Doris to avoid cross-case interference
        collection_name = None
        try:
            collection_name = self._doris_collection_name()
        except Exception:
            # If anything goes wrong, fall back silently; Doris will use its default name logic
            collection_name = None

        # Check if collection_name is in the db_config (e.g., for Zilliz, Milvus)
        db_config_dict = self.config.db_config.to_dict()
        if "collection_name" in db_config_dict and not collection_name:
            collection_name = db_config_dict.pop("collection_name")

        extra_db_kwargs = {}
        if collection_name:
            extra_db_kwargs["collection_name"] = collection_name
        if self.ca.is_multitenant:
            extra_db_kwargs["multitenant_tenant_labels"] = self.ca.tenant_labels()

        self.db = db_cls(
            dim=getattr(self.ca.dataset.data, "dim", 0),
            db_config=db_config_dict,
            db_case_config=self.config.db_case_config,
            drop_old=drop_old,
            with_scalar_labels=self.ca.with_scalar_labels,
            **extra_db_kwargs,
        )

    def _apply_fts_manifest_params(self) -> None:
        bm25_params = dict(getattr(self.ca.dataset, "bm25_params", {}) or {})
        analyzer_params = dict(getattr(self.ca.dataset, "analyzer_params", {}) or {})
        self.config.db_case_config, manifest_report = self.config.db_case_config.apply_fts_manifest(
            bm25_params=bm25_params,
            analyzer_params=analyzer_params,
        )
        self._fts_manifest_report = {
            "fts_manifest": {
                "bm25": bm25_params,
                "analyzer": analyzer_params,
            },
            **manifest_report,
        }

    def _fts_manifest_additional_parameters(self) -> dict:
        return dict(self._fts_manifest_report)

    def _pre_run(self, drop_old: bool = True):
        try:
            self._validate_cloud_cold_latency_config(drop_old)
            creates_multitenant_collection = (
                TaskStage.DROP_OLD in self.config.stages or TaskStage.LOAD in self.config.stages
            )
            if (
                self.ca.is_multitenant
                and self.config.db in {DB.Milvus, DB.ZillizCloud}
                and creates_multitenant_collection
                and not getattr(self.config.db_case_config, "use_partition_key", False)
            ):
                msg = "CloudMultiTenantSearchCase requires use_partition_key=True for Milvus/ZillizCloud"
                raise ValueError(msg)

            if self.is_fts:
                self.ca.dataset.prepare(self.dataset_source)
                self._apply_fts_manifest_params()
                self.init_db(drop_old)
                return

            self.init_db(drop_old)
            if self.ca.is_multitenant and self.db is not None:
                if not self.db.supports_multitenant():
                    msg = f"{self.config.db_name} does not support CloudMultiTenantSearchCase"
                    raise NotImplementedError(msg)
                self.db.set_multitenant_context(self.ca.tenant_labels())
                if self.config.db in {DB.Milvus, DB.ZillizCloud} and not creates_multitenant_collection:
                    self.db.validate_multitenant_schema()
            self.ca.dataset.prepare(
                self.dataset_source,
                filters=self.ca.filters,
                with_train_files=TaskStage.LOAD in self.config.stages,
                with_scalar_labels=self.ca.with_scalar_labels,
            )
        except ModuleNotFoundError as e:
            log.warning(f"pre run case error: please install client for db: {self.config.db}, error={e}")
            raise e from None

    def _validate_cloud_cold_latency_config(self, drop_old: bool) -> None:
        if getattr(self.ca, "label", None) != CaseLabel.CloudColdLatency:
            return
        if drop_old:
            msg = (
                "CloudColdLatencyCase requires an existing cold collection. "
                "Run with --skip-drop-old and --skip-load."
            )
            raise ValueError(msg)
        if TaskStage.LOAD in self.config.stages:
            msg = "CloudColdLatencyCase is search-only. Run with --skip-load."
            raise ValueError(msg)

    def run(self, drop_old: bool = True) -> Metric:
        log.info("Starting run")

        self._pre_run(drop_old)

        if self.ca.label == CaseLabel.Load:
            return self._run_capacity_case()
        if self.ca.label in {CaseLabel.Performance, CaseLabel.FullTextSearchPerformance}:
            return self._run_perf_case(drop_old)
        if self.ca.label == CaseLabel.Streaming:
            return self._run_streaming_case()
        if self.ca.label == CaseLabel.CloudInsert:
            return self._run_cloud_insert_case()
        if self.ca.label == CaseLabel.CloudColdLatency:
            return self._run_cloud_cold_latency_case(drop_old)
        msg = f"unknown case type: {self.ca.label}"
        log.warning(msg)
        raise ValueError(msg)

    def _run_capacity_case(self) -> Metric:
        """run capacity cases

        Returns:
            Metric: the max load count
        """
        assert self.db is not None
        log.info("Start capacity case")
        try:
            runner = SerialInsertRunner(
                self.db,
                self.ca.dataset,
                self.normalize,
                self.ca.filters,
                self.ca.load_timeout,
            )
            count = runner.run_endlessness()
        except Exception as e:
            log.warning(f"Failed to run capacity case, reason = {e}")
            raise e from None
        else:
            log.info(f"Capacity case loading dataset reaches VectorDB's limit: max capacity = {count}")
            return Metric(max_load_count=count)

    def _run_perf_case(self, drop_old: bool = True) -> Metric:
        """run performance cases

        Returns:
            Metric: load_duration, recall, serial_latency_p99, and, qps
        """

        log.info("Start performance case")
        try:
            m = Metric()
            if drop_old:
                if TaskStage.LOAD in self.config.stages:
                    count, load_dur = self._load_data()
                    build_dur = self._optimize()
                    m.inserted_count = count
                    m.insert_duration = round(load_dur, 4)
                    m.optimize_duration = round(build_dur, 4)
                    m.load_duration = round(load_dur + build_dur, 4)
                    m.additional_parameters.update(
                        {
                            "num_per_batch": config.NUM_PER_BATCH,
                            "load_concurrency": self.config.load_concurrency,
                        }
                    )
                    log.info(
                        f"Finish loading the entire dataset into VectorDB,"
                        f" insert_duration={load_dur}, optimize_duration={build_dur}"
                        f" load_duration(insert + optimize) = {m.load_duration}"
                    )
                else:
                    log.info("Data loading skipped")
            if TaskStage.SEARCH_SERIAL in self.config.stages or TaskStage.SEARCH_CONCURRENT in self.config.stages:
                self._init_search_runners()
                if TaskStage.SEARCH_CONCURRENT in self.config.stages:
                    search_results = self._conc_search()
                    (
                        m.qps,
                        m.conc_num_list,
                        m.conc_qps_list,
                        m.conc_latency_p99_list,
                        m.conc_latency_p95_list,
                        m.conc_latency_avg_list,
                    ) = search_results
                if TaskStage.SEARCH_SERIAL in self.config.stages:
                    search_results = self._serial_search()
                    if self.is_fts:
                        m.recall, m.ndcg, m.mrr, m.serial_latency_p99, m.serial_latency_p95 = search_results
                    else:
                        m.recall, m.ndcg, m.serial_latency_p99, m.serial_latency_p95 = search_results
            if hasattr(self.ca, "payload_profile"):
                m.payload_profile = self.ca.payload_profile.value
                m.payload_estimated_bytes_per_query = self.ca.estimated_payload_bytes_per_query(
                    self.config.case_config.k
                )
            if self.is_fts:
                m.additional_parameters.update(self._fts_manifest_additional_parameters())

        except Exception as e:
            log.warning(f"Failed to run performance case, reason = {e}")
            traceback.print_exc()
            raise e from None
        else:
            log.info(f"Performance case got result: {m}")
            return m

    def _run_streaming_case(self) -> Metric:
        log.info("Start streaming case")
        try:
            self._init_read_write_runner()
            m = self.read_write_runner.run_read_write()
        except Exception as e:
            log.warning(f"Failed to run streaming case, reason = {e}")
            traceback.print_exc()
            raise e from None
        else:
            log.info(f"Streaming case got result: {m}")
            return m

    def _run_cloud_insert_case(self) -> Metric:
        assert self.db is not None
        started = time.perf_counter()
        runner_kwargs = {}
        if self.ca.is_multitenant:
            runner_kwargs["tenant_case"] = self.ca
        runner = ConcurrentInsertRunner(
            self.db,
            self.ca.dataset,
            self.normalize,
            self.ca.filters,
            max_workers=self.config.load_concurrency or None,
            batch_size=self.ca.batch_size,
            duration=self.ca.duration,
            **runner_kwargs,
        )
        count = runner.task()
        insert_done = time.perf_counter()
        readiness_timeout = self.ca.readiness_timeout
        readiness_poll_interval = self.ca.readiness_poll_interval
        readiness_deadline = None if readiness_timeout is None else time.perf_counter() + readiness_timeout
        with self.db.init():
            status = self.db.poll_insert_readiness(count)
            searchable_started = time.perf_counter()
            while not status["fully_searchable"]:
                if readiness_deadline is not None and time.perf_counter() >= readiness_deadline:
                    msg = (
                        "Cloud insert readiness timed out waiting for fully_searchable "
                        f"after {readiness_timeout}s; last_status={status}"
                    )
                    raise TimeoutError(msg)
                time.sleep(readiness_poll_interval)
                status = self.db.poll_insert_readiness(count)
            indexed_started = time.perf_counter()
            while not status["fully_indexed"]:
                if readiness_deadline is not None and time.perf_counter() >= readiness_deadline:
                    msg = (
                        "Cloud insert readiness timed out waiting for fully_indexed "
                        f"after {readiness_timeout}s; last_status={status}"
                    )
                    raise TimeoutError(msg)
                time.sleep(readiness_poll_interval)
                status = self.db.poll_insert_readiness(count)
        return Metric(
            inserted_count=count,
            insert_rows_per_second=round(count / max(insert_done - started, 0.001), 4),
            insert_completion_seconds=round(insert_done - started, 4),
            searchable_after_insert_seconds=round(indexed_started - searchable_started, 4),
            indexed_after_searchable_seconds=round(time.perf_counter() - indexed_started, 4),
            additional_parameters=status.get("additional_parameters", {}),
        )

    def _init_cold_warm_search_runner(self) -> None:
        if self.normalize:
            test_emb = np.stack(self.ca.dataset.test_data)
            test_emb = test_emb / np.linalg.norm(test_emb, axis=1)[:, np.newaxis]
            self.test_emb = test_emb.tolist()
        else:
            self.test_emb = self.ca.dataset.test_data

        self.cold_warm_search_runner = ColdWarmSearchRunner(
            db=self.db,
            test_data=self.test_emb,
            filters=self.ca.filters,
            k=self.config.case_config.k,
            payload_profile=self.ca.payload_profile,
            query_count=self.ca.query_count,
        )

    def _run_cloud_cold_latency_case(self, drop_old: bool = True) -> Metric:
        log.info("Start cloud cold latency case")
        try:
            self._validate_cloud_cold_latency_config(drop_old)
            m = Metric()
            if drop_old:
                if TaskStage.LOAD in self.config.stages:
                    _, load_dur = self._load_train_data()
                    build_dur = self._optimize()
                    m.insert_duration = round(load_dur, 4)
                    m.optimize_duration = round(build_dur, 4)
                    m.load_duration = round(load_dur + build_dur, 4)
                else:
                    log.info("Data loading skipped")

            self._init_cold_warm_search_runner()
            m.additional_parameters = {
                "cold_latency": self.cold_warm_search_runner.run(),
            }
            m.payload_profile = self.ca.payload_profile.value
            m.payload_estimated_bytes_per_query = self.ca.estimated_payload_bytes_per_query(self.config.case_config.k)
        except Exception as e:
            log.warning(f"Failed to run cloud cold latency case, reason = {e}")
            traceback.print_exc()
            raise e from None
        else:
            log.info(f"Cloud cold latency case got result: {m}")
            return m

    @utils.time_it
    def _load_data(self):
        return self._load_train_data()

    def _load_train_data(self):
        """Insert vector or FTS train data concurrently and get insert duration."""
        try:
            runner_kwargs = {}
            if self.ca.is_multitenant:
                runner_kwargs["tenant_case"] = self.ca
            runner = ConcurrentInsertRunner(
                self.db,
                self.ca.dataset,
                self.normalize,
                self.ca.filters,
                self.ca.load_timeout,
                max_workers=self.config.load_concurrency or None,
                with_scalar_labels=self.ca.with_scalar_labels,
                workload_kind=self.workload_kind,
                **runner_kwargs,
            )
            return runner.run()
        except Exception as e:
            raise e from None
        finally:
            runner = None

    def _serial_search(self) -> tuple[float, ...]:
        """Performance serial tests, search the entire test data once,
        calculate the recall, serial_latency_p99, serial_latency_p95

        Returns:
            tuple[float, ...]: vector cases return recall, ndcg, p99, p95;
                FTS cases return recall, p99, p95.
        """
        try:
            results, _ = self.serial_search_runner.run()
        except Exception as e:
            log.warning(f"search error: {e!s}, {e}")
            self.stop()
            raise e from e
        else:
            return results

    def _conc_search(self):
        """Performance concurrency tests, search the test data endlessness
        for 30s in several concurrencies

        Returns:
            float: the largest qps in all concurrencies
        """
        try:
            return self.search_runner.run()
        except Exception as e:
            log.warning(f"search error: {e!s}, {e}")
            raise e from None
        finally:
            self.stop()

    @utils.time_it
    def _optimize_task(self) -> None:
        with self.db.init():
            self.db.optimize(data_size=self.ca.dataset.data.size)

    def _optimize(self) -> float:
        if self.is_fts:
            try:
                with utils.timeout(self.ca.optimize_timeout, PerformanceTimeoutError):
                    _, duration = self._optimize_task()
                    return duration
            except PerformanceTimeoutError:
                log.warning(f"VectorDB optimize timeout in {self.ca.optimize_timeout}")
                raise
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._optimize_task)
            try:
                return future.result(timeout=self.ca.optimize_timeout)[1]
            except TimeoutError as e:
                log.warning(f"VectorDB optimize timeout in {self.ca.optimize_timeout}")
                kill_proc_tree(pids=list(executor._processes.keys()))
                raise PerformanceTimeoutError from e
            except Exception as e:
                log.warning(f"VectorDB optimize error: {e}")
                raise e from None

    def _init_search_runners(self):
        if self.is_fts:
            return self._init_fts_search_runner()
        return self._init_search_runner()

    def _init_search_runner(self):
        if self.normalize:
            test_emb = np.stack(self.ca.dataset.test_data)
            test_emb = test_emb / np.linalg.norm(test_emb, axis=1)[:, np.newaxis]
            self.test_emb = test_emb.tolist()
        else:
            self.test_emb = self.ca.dataset.test_data

        tenant_labels = self.ca.tenant_labels() if self.ca.is_multitenant else None
        measure_recall = getattr(self.ca, "measure_recall", True)
        gt_df = self.ca.dataset.gt_data if measure_recall else None

        if TaskStage.SEARCH_SERIAL in self.config.stages:
            self.serial_search_runner = SerialSearchRunner(
                db=self.db,
                test_data=self.test_emb,
                ground_truth=gt_df,
                filters=self.ca.filters,
                k=self.config.case_config.k,
                payload_profile=self.ca.payload_profile,
                tenant_labels=tenant_labels,
                measure_recall=measure_recall,
                workload_kind=WorkloadKind.VECTOR,
            )
        if TaskStage.SEARCH_CONCURRENT in self.config.stages:
            self.search_runner = MultiProcessingSearchRunner(
                db=self.db,
                test_data=self.test_emb,
                filters=self.ca.filters,
                concurrencies=self.config.case_config.concurrency_search_config.num_concurrency,
                duration=self.config.case_config.concurrency_search_config.concurrency_duration,
                concurrency_timeout=self.config.case_config.concurrency_search_config.concurrency_timeout,
                k=self.config.case_config.k,
                payload_profile=self.ca.payload_profile,
                tenant_labels=tenant_labels,
                workload_kind=WorkloadKind.VECTOR,
            )

    def _init_fts_search_runner(self):
        fts_dataset = self.ca.dataset

        if fts_dataset.queries_data is None or fts_dataset.gt_data is None:
            msg = "FTS dataset is missing queries or ground truth. Call prepare() before initializing search."
            raise ValueError(msg)
        test_texts = [q.text for q in fts_dataset.queries_data]
        ground_truth = fts_dataset.gt_data
        if len(test_texts) != len(ground_truth):
            msg = f"FTS query count {len(test_texts)} does not match ground truth row count {len(ground_truth)}"
            raise ValueError(msg)

        log.info(f"FTS test will use {len(test_texts)} queries for testing")
        self.test_texts = test_texts

        if TaskStage.SEARCH_SERIAL in self.config.stages:
            self.serial_search_runner = SerialSearchRunner(
                db=self.db,
                test_data=test_texts,
                ground_truth=ground_truth,
                filters=self.ca.filters,
                k=self.config.case_config.k,
                payload_profile=self.ca.payload_profile,
                workload_kind=WorkloadKind.FULL_TEXT,
            )
        if TaskStage.SEARCH_CONCURRENT in self.config.stages:
            self.search_runner = MultiProcessingSearchRunner(
                db=self.db,
                test_data=test_texts,
                filters=self.ca.filters,
                concurrencies=self.config.case_config.concurrency_search_config.num_concurrency,
                duration=self.config.case_config.concurrency_search_config.concurrency_duration,
                concurrency_timeout=self.config.case_config.concurrency_search_config.concurrency_timeout,
                k=self.config.case_config.k,
                payload_profile=self.ca.payload_profile,
                workload_kind=WorkloadKind.FULL_TEXT,
            )

    def _init_read_write_runner(self):
        ca: StreamingPerformanceCase = self.ca
        self.read_write_runner = ReadWriteRunner(
            db=self.db,
            dataset=ca.dataset,
            insert_rate=ca.insert_rate,
            search_stages=ca.search_stages,
            optimize_after_write=ca.optimize_after_write,
            read_dur_after_write=ca.read_dur_after_write,
            concurrencies=ca.concurrencies,
            k=self.config.case_config.k,
            normalize=self.normalize,
        )

    def stop(self):
        if self.search_runner:
            self.search_runner.stop()


DATA_FORMAT = " %-14s | %-12s %-20s %7s | %-10s"
TITLE_FORMAT = (" %-14s | %-12s %-20s %7s | %-10s") % (
    "DB",
    "CaseType",
    "Dataset",
    "Filter",
    "task_label",
)


class TaskRunner(BaseModel):
    run_id: str
    task_label: str
    case_runners: list[CaseRunner]

    def num_cases(self) -> int:
        return len(self.case_runners)

    def num_finished(self) -> int:
        return self._get_num_by_status(RunningStatus.FINISHED)

    def set_finished(self, idx: int) -> None:
        self.case_runners[idx].status = RunningStatus.FINISHED

    def _get_num_by_status(self, status: RunningStatus) -> int:
        return sum([1 for c in self.case_runners if c.status == status])

    def display(self) -> None:
        fmt = [TITLE_FORMAT]
        fmt.append(DATA_FORMAT % ("-" * 11, "-" * 12, "-" * 20, "-" * 7, "-" * 7))

        for f in self.case_runners:
            filters = f.ca.filters.filter_rate

            ds_str = f"{f.ca.dataset.data.name}-{f.ca.dataset.data.label}-{utils.numerize(f.ca.dataset.data.size)}"
            fmt.append(
                DATA_FORMAT
                % (
                    f.config.db_name,
                    f.ca.label.name,
                    ds_str,
                    filters,
                    self.task_label,
                ),
            )

        tmp_logger = logging.getLogger("no_color")
        for f in fmt:
            tmp_logger.info(f)
