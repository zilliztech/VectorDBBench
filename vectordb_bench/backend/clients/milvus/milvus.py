"""Wrapper around the Milvus vector database over VectorDB"""

import logging
import time
from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any

from pymilvus import DataType, Function, FunctionType, MilvusClient, MilvusException

from vectordb_bench.backend.filter import Filter, FilterOp
from vectordb_bench.backend.payload import PayloadProfile

from ..api import VectorDB
from .config import MilvusFtsConfig, MilvusIndexConfig

log = logging.getLogger(__name__)

MILVUS_LOAD_REQS_SIZE = 1.5 * 1024 * 1024
MILVUS_FTS_BATCH_SIZE = 1000
MILVUS_FORCE_MERGE_TARGET_SIZE_MB = ((1 << 63) - 1) // (1024**2)


class Milvus(VectorDB):
    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    @classmethod
    def supports_full_text_search(cls) -> bool:
        return True

    def has_text_field(self) -> bool:
        return bool(getattr(self, "_is_fts", False) and getattr(self, "_text_field", None))

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: MilvusIndexConfig | MilvusFtsConfig,
        collection_name: str = "VDBBench",
        drop_old: bool = False,
        name: str = "Milvus",
        with_scalar_labels: bool = False,
        fts_batch_size: int | None = None,
        **kwargs,
    ):
        """Initialize wrapper around the milvus vector database."""
        self.name = name
        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = collection_name
        self.with_scalar_labels = with_scalar_labels

        self._scalar_label_field = "label"
        self._scalar_payload_label_field = self._scalar_label_field
        self._multitenant_partition_key_field = self._scalar_label_field
        self._scalar_labels_index_name = "labels_idx"
        self._is_fts = isinstance(self.case_config, MilvusFtsConfig)

        if self._is_fts:
            self.batch_size = fts_batch_size or MILVUS_FTS_BATCH_SIZE
            self._primary_field = "doc_id"
            self._text_field = "text"
            self._sparse_field = "sparse_vector"
            self._main_index_name = "sparse_vector_idx"
            self._sort_index_name = "doc_id_sort_idx"
            self._sort_index_field = self._primary_field
        else:
            self.batch_size = int(MILVUS_LOAD_REQS_SIZE / (dim * 4))
            self._primary_field = "pk"
            self._scalar_id_field = "id"
            self._vector_field = "vector"
            self._vector_index_name = "vector_idx"
            self._scalar_id_index_name = "id_sort_idx"
            self._main_index_name = self._vector_index_name
            self._sort_index_name = self._scalar_id_index_name
            self._sort_index_field = self._scalar_id_field

        self.multitenant_tenant_labels: list[str] = kwargs.get("multitenant_tenant_labels", [])
        if self.multitenant_tenant_labels:
            self._multitenant_partition_key_field = "labels"
            if self.with_scalar_labels:
                self._scalar_payload_label_field = "scalar_label"

        client = MilvusClient(
            uri=self.db_config.get("uri"),
            user=self.db_config.get("user"),
            password=self.db_config.get("password"),
            token=self.db_config.get("token", ""),
            timeout=30,
        )

        if drop_old and client.has_collection(self.collection_name):
            log.info(f"{self.name} client drop_old collection: {self.collection_name}")
            client.drop_collection(self.collection_name)

        if not client.has_collection(self.collection_name):
            schema = MilvusClient.create_schema()
            if self._is_fts:
                analyzer_params = (
                    self.case_config.analyzer_param()
                    if hasattr(self.case_config, "analyzer_param")
                    else self.case_config.index_param().get("analyzer_params", {"type": "english"})
                )
                schema.add_field(self._primary_field, DataType.VARCHAR, max_length=512, is_primary=True)
                schema.add_field(
                    self._text_field,
                    DataType.VARCHAR,
                    max_length=65535,
                    enable_analyzer=True,
                    enable_match=True,
                    analyzer_params=analyzer_params,
                )
                schema.add_field(self._sparse_field, DataType.SPARSE_FLOAT_VECTOR)
                if self.with_scalar_labels:
                    schema.add_field(self._scalar_label_field, DataType.VARCHAR, max_length=256)
                schema.add_function(
                    Function(
                        name="text_bm25_emb",
                        function_type=FunctionType.BM25,
                        input_field_names=[self._text_field],
                        output_field_names=[self._sparse_field],
                        params={},
                    )
                )
            else:
                schema.add_field(self._primary_field, DataType.INT64, is_primary=True)
                schema.add_field(self._scalar_id_field, DataType.INT64)
                schema.add_field(self._vector_field, DataType.FLOAT_VECTOR, dim=dim)

                if self.multitenant_tenant_labels:
                    schema.add_field(
                        self._multitenant_partition_key_field,
                        DataType.VARCHAR,
                        max_length=256,
                        is_partition_key=True,
                    )

                if self.with_scalar_labels:
                    is_partition_key = db_case_config.use_partition_key
                    log.info(f"with_scalar_labels, add a new varchar field, as partition_key: {is_partition_key}")
                    if not self.multitenant_tenant_labels or (
                        self._scalar_payload_label_field != self._multitenant_partition_key_field
                    ):
                        schema.add_field(
                            self._scalar_payload_label_field,
                            DataType.VARCHAR,
                            max_length=256,
                            is_partition_key=is_partition_key and not self.multitenant_tenant_labels,
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
        if self._is_fts:
            sparse_idx = (
                self.case_config.sparse_index_param()
                if hasattr(self.case_config, "sparse_index_param")
                else self.case_config.index_param()
            )
            index_params.add_index(
                field_name=self._sparse_field,
                index_name=self._main_index_name,
                index_type=sparse_idx.get("index_type", ""),
                metric_type=sparse_idx.get("metric_type", ""),
                params=sparse_idx.get("params", {}),
            )
        else:
            vec_idx = self.case_config.index_param()
            index_params.add_index(
                field_name=self._vector_field,
                index_name=self._vector_index_name,
                index_type=vec_idx.get("index_type", ""),
                metric_type=vec_idx.get("metric_type", ""),
                params=vec_idx.get("params", {}),
            )
        index_params.add_index(
            field_name=self._sort_index_field,
            index_name=self._sort_index_name,
            index_type="STL_SORT",
        )
        if self.with_scalar_labels:
            index_params.add_index(
                field_name=self._scalar_payload_label_field,
                index_name=self._scalar_labels_index_name,
                index_type="BITMAP",
            )
        return index_params

    def supports_multitenant(self) -> bool:
        return True

    def validate_multitenant_schema(self) -> None:
        client = MilvusClient(
            uri=self.db_config.get("uri"),
            user=self.db_config.get("user"),
            password=self.db_config.get("password"),
            token=self.db_config.get("token", ""),
            timeout=30,
        )
        try:
            desc = client.describe_collection(self.collection_name)
            fields = desc.get("fields", []) if isinstance(desc, dict) else []
            fields_by_name = {self._field_property(field, "name"): field for field in fields}
            partition_key_field = self._find_multitenant_partition_key_field(fields_by_name)
            if partition_key_field is None:
                label_field = fields_by_name.get(self._scalar_label_field)
                if label_field is None:
                    msg = f"{self.name} multitenant collection {self.collection_name} is missing tenant label field"
                    raise ValueError(msg)
                msg = f"{self.name} multitenant collection {self.collection_name} label field is not a partition key"
                raise ValueError(msg)
            self._multitenant_partition_key_field = partition_key_field
            if "scalar_label" in fields_by_name:
                self._scalar_payload_label_field = "scalar_label"
        finally:
            client.close()

    def _find_multitenant_partition_key_field(self, fields_by_name: dict[str, dict | object]) -> str | None:
        for field_name in [self._scalar_label_field, "labels"]:
            field = fields_by_name.get(field_name)
            if field is not None and self._field_property(field, "is_partition_key", False):
                return field_name
        return None

    @staticmethod
    def _field_property(field: dict | object, name: str, default: Any = None):
        if isinstance(field, dict):
            if name in field:
                return field[name]
            params = field.get("params")
            if isinstance(params, dict) and name in params:
                return params[name]
            return default
        return getattr(field, name, default)

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
            token=self.db_config.get("token", ""),
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
            info = self.client.describe_index(self.collection_name, self._main_index_name)
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

            if getattr(self.case_config, "is_gpu_index", False):
                log.debug("skip force merge compaction for gpu index type.")
            else:
                try:
                    # wait for sort, index, compact
                    self._wait_for_segments_sorted()
                    self._wait_for_index()
                    compaction_id = self.client.compact(
                        self.collection_name, target_size=MILVUS_FORCE_MERGE_TARGET_SIZE_MB
                    )
                    if compaction_id > 0:
                        self._wait_for_compaction(compaction_id)
                    log.info(f"{self.name} force merge compaction completed.")
                except Exception as e:
                    log.warning(f"{self.name} compact or list segments error: {e}")
                    if getattr(getattr(e, "code", None), "name", None) == "PERMISSION_DENIED":
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
        if self._is_fts:
            return False
        if self.case_config.is_gpu_index:
            log.info("current gpu_index only supports IP / L2, cosine dataset need normalize.")
            return True

        return False

    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        tenant_labels_data: list[str] | None = None,
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
                    if tenant_labels_data is not None:
                        row[self._multitenant_partition_key_field] = tenant_labels_data[i]
                    if self.with_scalar_labels:
                        row[self._scalar_payload_label_field] = labels_data[i]
                    batch_data.append(row)
                res = self.client.insert(self.collection_name, batch_data)
                insert_count += res["insert_count"]
        except MilvusException as e:
            log.info(f"Failed to insert data: {e}")
            return insert_count, e
        return insert_count, None

    def insert_documents(
        self,
        texts: Iterable[str],
        doc_ids: list[str],
        **kwargs,
    ) -> tuple[int, Exception | None]:
        """Insert documents into a Milvus BM25 full-text collection."""
        if not self._is_fts:
            msg = "insert_documents is only valid in FTS mode"
            raise RuntimeError(msg)
        assert self.client is not None

        docs = list(texts)
        if len(docs) != len(doc_ids):
            msg = f"Mismatch between texts ({len(docs)}) and doc_ids ({len(doc_ids)}) lengths"
            raise ValueError(msg)

        batch_size = kwargs.get("batch_size", self.batch_size)
        labels_data = kwargs.get("labels_data")

        insert_count = 0
        try:
            for batch_start_offset in range(0, len(docs), batch_size):
                batch_end_offset = min(batch_start_offset + batch_size, len(docs))
                rows = []
                for i in range(batch_start_offset, batch_end_offset):
                    row = {
                        self._primary_field: str(doc_ids[i]),
                        self._text_field: docs[i],
                    }
                    if self.with_scalar_labels:
                        row[self._scalar_label_field] = labels_data[i] if labels_data is not None else ""
                    rows.append(row)

                res = self.client.insert(self.collection_name, rows)
                insert_count += res["insert_count"]
                if batch_start_offset // batch_size % 10 == 0:
                    log.debug(
                        f"{self.name} batch insert progress: {batch_end_offset}/{len(docs)} "
                        f"({batch_end_offset / len(docs) * 100:.1f}%)"
                    )
        except MilvusException as e:
            log.info(f"{self.name} insert error: {e}")
            return insert_count, e
        return insert_count, None

    def prepare_filter(self, filters: Filter):
        if self._is_fts:
            self.expr = ""
            return
        if filters.type == FilterOp.NonFilter:
            self.expr = ""
        elif filters.type == FilterOp.NumGE:
            self.expr = f"{self._scalar_id_field} >= {filters.int_value}"
        elif filters.type == FilterOp.StrEqual:
            self.expr = f"{self._scalar_payload_label_field} == '{filters.label_value}'"
        else:
            msg = f"Not support Filter for Milvus - {filters}"
            raise ValueError(msg)

    def supports_payload_profile(self, payload_profile: PayloadProfile) -> bool:
        return payload_profile in {
            PayloadProfile.IDS_ONLY,
            PayloadProfile.VECTOR,
            PayloadProfile.SCALAR_LABEL,
        }

    def poll_insert_readiness(self, expected_count: int) -> dict:
        assert self.client is not None
        self.client.flush(self.collection_name)
        stats = self.client.get_collection_stats(self.collection_name)
        count = int(stats.get("row_count", stats.get("num_entities", 0)))
        progress = self.client.describe_index(self.collection_name, self._vector_index_name)
        return {
            "fully_searchable": count >= expected_count,
            "fully_indexed": progress.get("pending_index_rows", -1) == 0,
            "additional_parameters": {},
        }

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        timeout: int | None = None,
        payload_profile: PayloadProfile = PayloadProfile.IDS_ONLY,
        tenant: str | None = None,
    ) -> list[int]:
        """Perform a search on a query embedding and return results."""
        assert self.client is not None

        output_fields = None
        if payload_profile == PayloadProfile.VECTOR:
            output_fields = [self._vector_field]
        elif payload_profile == PayloadProfile.SCALAR_LABEL:
            output_fields = [getattr(self, "_scalar_payload_label_field", self._scalar_label_field)]

        expr = self.expr
        if tenant is not None:
            tenant_field = getattr(self, "_multitenant_partition_key_field", self._scalar_label_field)
            tenant_expr = f"{tenant_field} == '{tenant}'"
            expr = tenant_expr if not expr else f"({expr}) and ({tenant_expr})"

        search_kwargs = {
            "collection_name": self.collection_name,
            "data": [query],
            "anns_field": self._vector_field,
            "search_params": self.case_config.search_param(),
            "limit": k,
            "filter": expr,
            "output_fields": output_fields,
        }
        res = self.client.search(**search_kwargs)

        return [result[self._primary_field] for result in res[0]]

    def search_documents(
        self,
        query: str,
        k: int = 100,
        timeout: int | None = None,
        payload_profile: PayloadProfile = PayloadProfile.IDS_ONLY,
    ) -> list[str]:
        """Search a Milvus BM25 full-text collection and return document IDs."""
        if not self._is_fts:
            msg = "search_documents only valid in FTS mode"
            raise RuntimeError(msg)
        if not self.supports_document_payload_profile(payload_profile):
            msg = f"{getattr(self, 'name', 'Milvus')} does not support document payload_profile={payload_profile.value}"
            raise NotImplementedError(msg)
        assert self.client is not None

        output_fields = [self._primary_field]
        if payload_profile == PayloadProfile.TEXT:
            output_fields.append(self._text_field)

        res = self.client.search(
            collection_name=self.collection_name,
            data=[str(query)],
            anns_field=self._sparse_field,
            search_params=self.case_config.search_param(),
            limit=k,
            output_fields=output_fields,
        )

        hits = res[0] if res else []
        doc_ids = []
        for hit in hits:
            entity = hit.get("entity", hit) if isinstance(hit, dict) else hit
            if isinstance(entity, dict):
                doc_ids.append(str(entity.get(self._primary_field)))
            else:
                doc_ids.append(str(getattr(entity, self._primary_field)))
        return doc_ids
