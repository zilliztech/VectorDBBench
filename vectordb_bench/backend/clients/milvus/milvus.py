"""Wrapper around the Milvus vector database over VectorDB"""

import logging
import time
from collections.abc import Iterable
from contextlib import contextmanager

from pymilvus import DataType, MilvusClient, MilvusException

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import MilvusIndexConfig

log = logging.getLogger(__name__)

MILVUS_LOAD_REQS_SIZE = 1.5 * 1024 * 1024


class Milvus(VectorDB):
    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: MilvusIndexConfig,
        collection_name: str = "VDBBench",
        drop_old: bool = False,
        name: str = "Milvus",
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        """Initialize wrapper around the milvus vector database."""
        self.name = name
        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = collection_name
        self.batch_size = int(MILVUS_LOAD_REQS_SIZE / (dim * 4))
        self.with_scalar_labels = with_scalar_labels

        self._primary_field = "pk"
        self._scalar_id_field = "id"
        self._scalar_label_field = "label"
        self._vector_field = "vector"
        self._vector_index_name = "vector_idx"
        self._scalar_id_index_name = "id_sort_idx"
        self._scalar_labels_index_name = "labels_idx"

        client = MilvusClient(
            uri=self.db_config.get("uri"),
            user=self.db_config.get("user"),
            password=self.db_config.get("password"),
            timeout=30,
        )

        if drop_old and client.has_collection(self.collection_name):
            log.info(f"{self.name} client drop_old collection: {self.collection_name}")
            client.drop_collection(self.collection_name)

        if not client.has_collection(self.collection_name):
            schema = MilvusClient.create_schema()
            schema.add_field(self._primary_field, DataType.INT64, is_primary=True)
            schema.add_field(self._scalar_id_field, DataType.INT64)
            schema.add_field(self._vector_field, DataType.FLOAT_VECTOR, dim=dim)

            if self.with_scalar_labels:
                is_partition_key = db_case_config.use_partition_key
                log.info(f"with_scalar_labels, add a new varchar field, as partition_key: {is_partition_key}")
                schema.add_field(
                    self._scalar_label_field,
                    DataType.VARCHAR,
                    max_length=256,
                    is_partition_key=is_partition_key,
                )

            log.info(f"{self.name} create collection: {self.collection_name}")

            index_params = self._build_index_params()
            client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                num_shards=self.db_config.get("num_shards", 1),
                consistency_level="Session",
            )
            client.create_index(self.collection_name, index_params)
            client.load_collection(
                self.collection_name,
                replica_number=self.db_config.get("replica_number", 1),
            )

        client.close()

    def _build_index_params(self):
        index_params = MilvusClient.prepare_index_params()
        vec_idx = self.case_config.index_param()
        index_params.add_index(
            field_name=self._vector_field,
            index_name=self._vector_index_name,
            index_type=vec_idx.get("index_type", ""),
            metric_type=vec_idx.get("metric_type", ""),
            params=vec_idx.get("params", {}),
        )
        index_params.add_index(
            field_name=self._scalar_id_field,
            index_name=self._scalar_id_index_name,
            index_type="STL_SORT",
        )
        if self.with_scalar_labels:
            index_params.add_index(
                field_name=self._scalar_label_field,
                index_name=self._scalar_labels_index_name,
                index_type="BITMAP",
            )
        return index_params

    @contextmanager
    def init(self):
        """
        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding()
        """
        self.client: MilvusClient | None = None
        self.client = MilvusClient(
            uri=self.db_config.get("uri"),
            user=self.db_config.get("user"),
            password=self.db_config.get("password"),
            timeout=60,
        )
        yield
        self.client.close()
        self.client = None

    def _wait_for_segments_sorted(self):
        while True:
            segments = self.client.list_persistent_segments(self.collection_name)
            unsorted = [s for s in segments if not s.is_sorted]
            if not unsorted:
                log.info(f"{self.name} all persistent segments are sorted.")
                break
            log.debug(f"{self.name} waiting for {len(unsorted)} segments to be sorted...")
            time.sleep(5)

    def _wait_for_index(self):
        while True:
            info = self.client.describe_index(self.collection_name, self._vector_index_name)
            if info.get("pending_index_rows", -1) == 0:
                break
            time.sleep(5)

    def _wait_for_compaction(self, compaction_id: int):
        while True:
            state = self.client.get_compaction_state(compaction_id)
            if state == "Completed":
                break
            time.sleep(0.5)

    def _optimize(self):
        log.info(f"{self.name} optimizing before search")
        try:
            self.client.flush(self.collection_name)

            if self.case_config.is_gpu_index:
                log.debug("skip force merge compaction for gpu index type.")
            else:
                try:
                    # wait for sort, index, compact
                    self._wait_for_segments_sorted()
                    self._wait_for_index()
                    compaction_id = self.client.compact(self.collection_name, target_size=(2**63 - 1))
                    if compaction_id > 0:
                        self._wait_for_compaction(compaction_id)
                    log.info(f"{self.name} force merge compaction completed.")
                except Exception as e:
                    log.warning(f"{self.name} compact or list segments error: {e}")
                    if hasattr(e, "code") and e.code().name == "PERMISSION_DENIED":
                        log.warning("Skip compact due to list segments or compact permission denied.")
                    else:
                        raise e from None

            # wait for index no matter what
            self._wait_for_index()
            self.client.refresh_load(self.collection_name)
        except Exception as e:
            log.warning(f"{self.name} optimize error: {e}")
            raise e from None

    def optimize(self, data_size: int | None = None):
        assert self.client, "Please call self.init() before"
        self._optimize()

    def need_normalize_cosine(self) -> bool:
        """Wheather this database need to normalize dataset to support COSINE"""
        if self.case_config.is_gpu_index:
            log.info("current gpu_index only supports IP / L2, cosine dataset need normalize.")
            return True

        return False

    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception]:
        """Insert embeddings into Milvus. should call self.init() first"""
        assert self.client is not None
        assert len(embeddings) == len(metadata)
        insert_count = 0
        try:
            for batch_start_offset in range(0, len(embeddings), self.batch_size):
                batch_end_offset = min(batch_start_offset + self.batch_size, len(embeddings))
                batch_data = []
                for i in range(batch_start_offset, batch_end_offset):
                    row = {
                        self._primary_field: metadata[i],
                        self._scalar_id_field: metadata[i],
                        self._vector_field: embeddings[i],
                    }
                    if self.with_scalar_labels:
                        row[self._scalar_label_field] = labels_data[i]
                    batch_data.append(row)
                res = self.client.insert(self.collection_name, batch_data)
                insert_count += res["insert_count"]
        except MilvusException as e:
            log.info(f"Failed to insert data: {e}")
            return insert_count, e
        return insert_count, None

    def prepare_filter(self, filters: Filter):
        if filters.type == FilterOp.NonFilter:
            self.expr = ""
        elif filters.type == FilterOp.NumGE:
            self.expr = f"{self._scalar_id_field} >= {filters.int_value}"
        elif filters.type == FilterOp.StrEqual:
            self.expr = f"{self._scalar_label_field} == '{filters.label_value}'"
        else:
            msg = f"Not support Filter for Milvus - {filters}"
            raise ValueError(msg)

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        timeout: int | None = None,
    ) -> list[int]:
        """Perform a search on a query embedding and return results."""
        assert self.client is not None

        res = self.client.search(
            collection_name=self.collection_name,
            data=[query],
            anns_field=self._vector_field,
            search_params=self.case_config.search_param(),
            limit=k,
            filter=self.expr,
        )

        return [result[self._primary_field] for result in res[0]]
