"""Wrapper around Apache Pinot vector database over VectorDB"""

import json
import logging
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import requests

from ...filter import Filter, FilterOp
from ..api import DBCaseConfig, MetricType, VectorDB

log = logging.getLogger(__name__)

# Rows accumulated before flushing a segment to Pinot.
# Large enough to avoid thousands of tiny segments (which breaks IVF training
# and hurts query performance), yet small enough to keep memory use bounded.
# For 768-dim float32 vectors: 100K rows ≈ 300 MB in-memory.
DEFAULT_INGEST_BATCH_SIZE = 100_000


class Pinot(VectorDB):
    """Apache Pinot vector database client for VectorDBBench."""

    name = "Pinot"
    # thread_safe=True: each flush uses a fresh requests.Session (not a shared one),
    # and each worker thread has its own row buffer via threading.local().
    # This lets the framework spawn multiple load workers that flush segments in parallel.
    thread_safe: bool = True
    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig,
        collection_name: str = "VectorBenchCollection",
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        self.dim = dim
        self.case_config = db_case_config
        self.table_name = collection_name
        self._primary_field = "id"
        self._vector_field = "embedding"
        self._label_field = "labels"
        self.with_scalar_labels = with_scalar_labels
        self._filter_where: str = ""  # set by prepare_filter(); applied in search_embedding()

        controller_host = db_config["controller_host"]
        controller_port = db_config["controller_port"]
        broker_host = db_config["broker_host"]
        broker_port = db_config["broker_port"]
        self._controller_url = f"http://{controller_host}:{controller_port}"
        self._broker_url = f"http://{broker_host}:{broker_port}"

        self._auth = None
        if db_config.get("username") and db_config.get("password"):
            self._auth = (db_config["username"], db_config["password"])

        # Per-thread row buffers — each worker thread accumulates rows independently
        # and flushes its own segment when the threshold is reached.
        # _registered_buffers tracks all thread-local objects so init() teardown
        # can flush any remaining rows from every worker thread.
        self._thread_local = threading.local()
        self._registered_buffers: list = []
        self._buffers_lock = threading.Lock()
        self._ingest_batch_size: int = db_config.get("ingest_batch_size", DEFAULT_INGEST_BATCH_SIZE)

        self.session = None

        with requests.Session() as setup_session:
            if self._auth:
                setup_session.auth = self._auth

            if drop_old:
                self._delete_table(setup_session)
                self._delete_schema(setup_session)

            if not self._schema_exists(setup_session):
                self._create_schema(setup_session)

            if not self._table_exists(setup_session):
                self._create_table(setup_session)

    def _schema_exists(self, session: requests.Session) -> bool:
        resp = session.get(f"{self._controller_url}/schemas/{self.table_name}")
        return resp.status_code == 200

    def _table_exists(self, session: requests.Session) -> bool:
        resp = session.get(f"{self._controller_url}/tables/{self.table_name}")
        if resp.status_code != 200:
            return False
        data = resp.json()
        return bool(data.get("OFFLINE") or data.get("tables"))

    def _delete_table(self, session: requests.Session):
        resp = session.delete(f"{self._controller_url}/tables/{self.table_name}?type=offline")
        if resp.status_code not in (200, 404):
            log.warning(f"Failed to delete Pinot table {self.table_name}: {resp.text}")
        else:
            log.info(f"Deleted Pinot table: {self.table_name}")
        # Wait for Pinot to finish cleaning up the external view
        for _ in range(30):
            check = session.get(f"{self._controller_url}/tables/{self.table_name}/externalview")
            if check.status_code == 404 or not check.json():
                break
            log.info(f"Waiting for Pinot external view cleanup for {self.table_name}...")
            time.sleep(2)
        else:
            log.warning(f"External view for {self.table_name} did not clear within 60s")

    def _delete_schema(self, session: requests.Session):
        resp = session.delete(f"{self._controller_url}/schemas/{self.table_name}")
        if resp.status_code not in (200, 404):
            log.warning(f"Failed to delete Pinot schema {self.table_name}: {resp.text}")
        else:
            log.info(f"Deleted Pinot schema: {self.table_name}")

    def _create_schema(self, session: requests.Session):
        dimension_fields = [
            {"name": self._primary_field, "dataType": "INT"},
            {"name": self._vector_field, "dataType": "FLOAT", "singleValueField": False},
        ]
        if self.with_scalar_labels:
            dimension_fields.append({"name": self._label_field, "dataType": "STRING"})

        schema = {
            "schemaName": self.table_name,
            "dimensionFieldSpecs": dimension_fields,
        }
        resp = session.post(
            f"{self._controller_url}/schemas",
            json=schema,
            headers={"Content-Type": "application/json"},
        )
        if not resp.ok:
            log.error(f"Failed to create Pinot schema: {resp.text}")
            resp.raise_for_status()
        log.info(f"Created Pinot schema: {self.table_name}")

    def _create_table(self, session: requests.Session):
        metric_str = self._get_index_metric_str()
        index_params = self.case_config.index_param()

        # Pull vectorIndexType out of index_params; remaining entries are type-specific properties.
        vector_index_type = index_params.pop("vectorIndexType", "HNSW")
        properties: dict = {
            "vectorIndexType": vector_index_type,
            "vectorDimension": str(self.dim),
            "vectorDistanceFunction": metric_str,
            "version": "1",
        }
        properties.update(index_params)

        table_config = {
            "tableName": self.table_name,
            "tableType": "OFFLINE",
            "segmentsConfig": {
                "replication": "1",
                "schemaName": self.table_name,
            },
            "tenants": {},
            "tableIndexConfig": {
                "loadMode": "MMAP",
                # Inverted index on id for fast equality lookups; range index for >=/<= filters.
                "invertedIndexColumns": [self._primary_field],
                "rangeIndexColumns": [self._primary_field],
            },
            "fieldConfigList": [
                {
                    "encodingType": "RAW",
                    "indexType": "VECTOR",
                    "name": self._vector_field,
                    "properties": properties,
                }
            ],
            "ingestionConfig": {
                "batchIngestionConfig": {
                    "segmentIngestionType": "APPEND",
                    "segmentIngestionFrequency": "DAILY",
                }
            },
            "metadata": {},
        }
        resp = session.post(
            f"{self._controller_url}/tables",
            json=table_config,
            headers={"Content-Type": "application/json"},
        )
        if not resp.ok:
            log.error(f"Failed to create Pinot table: {resp.text}")
            resp.raise_for_status()
        log.info(f"Created Pinot table: {self.table_name}")

    def _get_index_metric_str(self) -> str:
        if self.case_config.metric_type == MetricType.COSINE:
            return "COSINE"
        if self.case_config.metric_type == MetricType.IP:
            return "INNER_PRODUCT"
        return "L2"

    def _get_query_distance_fn(self) -> tuple[str, str]:
        """Returns (sql_function_name, sort_order) for vector search."""
        if self.case_config.metric_type == MetricType.COSINE:
            return "cosineDistance", "ASC"
        if self.case_config.metric_type == MetricType.IP:
            return "innerProduct", "DESC"
        return "l2Distance", "ASC"

    def prepare_filter(self, filters: Filter):
        """Pre-compute the SQL WHERE fragment for the given filter condition."""
        if filters.type == FilterOp.NonFilter:
            self._filter_where = ""
        elif filters.type == FilterOp.NumGE:
            self._filter_where = f"{filters.int_field} >= {filters.int_value}"
        elif filters.type == FilterOp.StrEqual:
            self._filter_where = f"{self._label_field} = '{filters.label_value}'"

    def __getstate__(self):
        # threading.local and Lock cannot be pickled; the framework pickles the DB
        # instance to send it to the load subprocess, so we must exclude them here
        # and recreate them in __setstate__ after unpickling.
        state = self.__dict__.copy()
        state.pop("_thread_local", None)
        state.pop("_buffers_lock", None)
        state.pop("_registered_buffers", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._thread_local = threading.local()
        self._buffers_lock = threading.Lock()
        self._registered_buffers = []

    def _get_thread_buffer(self) -> list:
        """Return this thread's row buffer, creating and registering it on first access.

        The list itself (not the threading.local container) is registered so that the
        teardown in init() can access buffered rows from the main thread, which has its
        own (empty) thread-local slot and cannot see other threads' data through the
        threading.local object.
        """
        if not hasattr(self._thread_local, "pending_rows"):
            self._thread_local.pending_rows = []
            with self._buffers_lock:
                # Register the list itself, not self._thread_local, so teardown can
                # read the contents from any thread (main thread included).
                self._registered_buffers.append(self._thread_local.pending_rows)
        return self._thread_local.pending_rows

    @contextmanager
    def init(self):
        self.session = requests.Session()
        if self._auth:
            self.session.auth = self._auth
        # Reset buffer registry so teardown only flushes buffers from this init() scope.
        self._registered_buffers = []
        try:
            yield
        finally:
            # Flush any rows that were buffered but not yet sent to Pinot.
            # Each worker thread may have its own partially-filled buffer.
            # This must happen here — not in optimize() — because optimize() runs
            # in a separate subprocess where the buffers are always empty.
            for pending in self._registered_buffers:
                if pending:
                    log.info(f"Pinot init teardown: flushing {len(pending)} remaining buffered rows")
                    _, err = self._flush_rows(pending)
                    if err:
                        log.warning(f"Pinot init teardown: flush error: {err}")
            self.session.close()
            self.session = None

    def _flush_rows(self, rows: list) -> tuple[int, Exception | None]:
        """Flush the given row list to Pinot as one segment using a fresh HTTP session.

        Using a fresh session (not self.session) makes this method safe to call
        from multiple threads concurrently.  On success the list is cleared in-place.
        Returns (rows_flushed, error). On error the list is left intact so the caller
        can decide whether to retry.
        """
        if not rows:
            return 0, None

        n = len(rows)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, prefix="pinot_ingest_") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
            tmp_path = f.name

        try:
            batch_config = json.dumps({"inputFormat": "json"})
            params = {
                "tableNameWithType": f"{self.table_name}_OFFLINE",
                "batchConfigMapStr": batch_config,
            }
            last_err = None
            with requests.Session() as session:
                if self._auth:
                    session.auth = self._auth
                for attempt in range(3):
                    try:
                        with Path(tmp_path).open("rb") as f:
                            resp = session.post(
                                f"{self._controller_url}/ingestFromFile",
                                params=params,
                                files={"file": (Path(tmp_path).name, f, "application/json")},
                                timeout=1800,  # HNSW index building for 100K x 768D can take 10+ min
                            )
                        if resp.ok:
                            rows.clear()
                            log.debug(f"Pinot: flushed segment with {n} rows")
                            return n, None
                        last_err = Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")
                        log.warning(f"Pinot flush attempt {attempt + 1} failed: {last_err}")
                        time.sleep(1 + attempt)
                    except Exception as e:
                        last_err = e
                        log.warning(f"Pinot flush attempt {attempt + 1} error: {e}")
                        time.sleep(1 + attempt)
            return 0, last_err
        finally:
            Path(tmp_path).unlink()

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[int, Exception]:
        # Each thread has its own buffer; no locking needed here.
        pending = self._get_thread_buffer()

        for i, (emb, meta) in enumerate(zip(embeddings, metadata, strict=False)):
            row = {self._primary_field: meta, self._vector_field: list(emb)}
            if self.with_scalar_labels and labels_data is not None:
                row[self._label_field] = labels_data[i]
            pending.append(row)

        if len(pending) >= self._ingest_batch_size:
            _flushed, err = self._flush_rows(pending)
            if err:
                log.warning(f"Failed to flush Pinot buffer: {err}")
                return 0, err

        return len(embeddings), None

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
    ) -> list[int]:
        assert self.session is not None, "Session not initialized"

        query_arr = ",".join(str(v) for v in query)
        dist_fn, order = self._get_query_distance_fn()

        search_params = self.case_config.search_param()
        nprobe = search_params.get("nprobe")

        filter_clause = self._filter_where

        if nprobe is not None:
            # IVF-based index: set probe count via session option, use ORDER BY for top-k
            where = f"WHERE {filter_clause} " if filter_clause else ""
            sql = (
                f"set vectorNprobe={nprobe}; "
                f"SELECT {self._primary_field} "
                f"FROM {self.table_name} "
                f"{where}"
                f"ORDER BY {dist_fn}({self._vector_field}, ARRAY[{query_arr}]) {order} "
                f"LIMIT {k}"
            )
        else:
            # HNSW index: WHERE vectorSimilarity(..., ef) triggers Lucene HNSW graph search
            # with a candidate list of size ef (defaults to k).  Larger ef → better recall.
            # ORDER BY dist re-ranks the ANN candidates for correct final recall.
            ef = search_params.get("ef") or k
            extra = f" AND {filter_clause}" if filter_clause else ""
            sql = (
                f"SELECT {self._primary_field} "
                f"FROM {self.table_name} "
                f"WHERE vectorSimilarity({self._vector_field}, ARRAY[{query_arr}], {ef}){extra} "
                f"ORDER BY {dist_fn}({self._vector_field}, ARRAY[{query_arr}]) {order} "
                f"LIMIT {k}"
            )

        resp = self.session.post(
            f"{self._broker_url}/query/sql",
            json={"sql": sql},
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        resp.raise_for_status()
        result = resp.json()

        rows = result.get("resultTable", {}).get("rows", [])
        return [row[0] for row in rows]

    def optimize(self, data_size: int | None = None):
        """Wait for all ingested data to be queryable in Pinot.

        Remaining buffered rows are flushed on init() teardown (in the insert
        subprocess), so by the time optimize() runs they are already in Pinot.
        """
        if self.session is None:
            return

        if data_size is None:
            time.sleep(5)
            return

        max_wait = 600
        check_interval = 10
        start = time.time()

        while time.time() - start < max_wait:
            try:
                resp = self.session.post(
                    f"{self._broker_url}/query/sql",
                    json={"sql": f"SELECT COUNT(*) FROM {self.table_name}"},
                    headers={"Content-Type": "application/json"},
                )
                if resp.status_code == 200:
                    rows = resp.json().get("resultTable", {}).get("rows", [])
                    current_count = rows[0][0] if rows else 0
                    if current_count >= data_size:
                        log.info(f"Pinot: all {data_size} rows are queryable")
                        return
                    log.info(f"Pinot: {current_count}/{data_size} rows queryable, waiting...")
            except Exception as e:
                log.warning(f"Pinot optimize check error: {e}")

            time.sleep(check_interval)

        log.warning(f"Pinot optimize timed out after {max_wait}s")
