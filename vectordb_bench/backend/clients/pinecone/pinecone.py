"""Wrapper around the Pinecone vector database over VectorDB"""

import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import Any

import pinecone

from vectordb_bench.backend.filter import Filter, FilterOp
from vectordb_bench.backend.payload import PayloadProfile

from ..api import DBCaseConfig, VectorDB

log = logging.getLogger(__name__)

PINECONE_MAX_NUM_PER_BATCH = 1000
PINECONE_MAX_SIZE_PER_BATCH = 2 * 1024 * 1024  # 2MB
PINECONE_QUERY_MAX_RETRIES_ENV = "PINECONE_QUERY_MAX_RETRIES"
PINECONE_QUERY_RETRY_SLEEP_ENV = "PINECONE_QUERY_RETRY_SLEEP_SECONDS"
PINECONE_QUERY_DEFAULT_MAX_RETRIES = 10
PINECONE_QUERY_DEFAULT_RETRY_SLEEP_SECONDS = 0.5


class Pinecone(VectorDB):
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
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        """Initialize wrapper around the milvus vector database."""
        self.index_name = db_config.get("index_name", "")
        self.api_key = db_config.get("api_key", "")
        self.multitenant_namespace_prefix = db_config.get("multitenant_namespace_prefix", "vdbbench_mt_")
        self.multitenant_tenant_labels: list[str] = kwargs.get("multitenant_tenant_labels", [])
        self._multitenant_insert_counts: dict[str, int] = {}
        self.batch_size = int(
            min(PINECONE_MAX_SIZE_PER_BATCH / (dim * 5), PINECONE_MAX_NUM_PER_BATCH),
        )
        self._last_write_lsn: int | None = None
        self._last_write_lsn_lock = threading.Lock()
        self._readiness_probe_vector = [0.0] * dim

        pc = pinecone.Pinecone(api_key=self.api_key)
        index = pc.Index(self.index_name)

        self.with_scalar_labels = with_scalar_labels
        self.expr = None
        if drop_old and self.multitenant_tenant_labels:
            for tenant in self.multitenant_tenant_labels:
                namespace = self._namespace_for_tenant(tenant)
                log.info(f"Pinecone index delete multitenant namespace: {namespace}")
                index.delete(delete_all=True, namespace=namespace)
        elif drop_old:
            index_stats = index.describe_index_stats()
            index_dim = index_stats["dimension"]
            if index_dim != dim:
                msg = f"Pinecone index {self.index_name} dimension mismatch, expected {index_dim} got {dim}"
                raise ValueError(msg)
            for namespace in index_stats["namespaces"]:
                log.info(f"Pinecone index delete namespace: {namespace}")
                index.delete(delete_all=True, namespace=namespace)

        self._scalar_id_field = "meta"
        self._scalar_label_field = "label"

    @contextmanager
    def init(self):
        pc = pinecone.Pinecone(api_key=self.api_key)
        self.index = pc.Index(self.index_name)
        yield

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_last_write_lsn_lock", None)
        return state

    def optimize(self, data_size: int | None = None):
        pass

    def supports_payload_profile(self, payload_profile: PayloadProfile) -> bool:
        return payload_profile in {
            PayloadProfile.IDS_ONLY,
            PayloadProfile.SCALAR_LABEL,
            PayloadProfile.VECTOR,
        }

    def supports_multitenant(self) -> bool:
        return True

    def set_multitenant_context(self, tenant_labels: list[str]) -> None:
        self.multitenant_tenant_labels = tenant_labels

    def _namespace_for_tenant(self, tenant: str | None) -> str | None:
        if tenant is None:
            return None
        return f"{self.multitenant_namespace_prefix}{tenant}"

    def poll_insert_readiness(self, expected_count: int) -> dict:
        stats = self.index.describe_index_stats()
        if getattr(self, "multitenant_tenant_labels", []):
            namespaces = stats.get("namespaces", {})
            expected_by_tenant = self._expected_multitenant_counts(expected_count)
            count_ready = True
            for tenant, expected_tenant_count in expected_by_tenant.items():
                namespace = self._namespace_for_tenant(tenant)
                namespace_stats = namespaces.get(namespace, {})
                namespace_count = namespace_stats.get("vector_count", 0)
                count_ready = count_ready and namespace_count >= expected_tenant_count
            fresh = self._multitenant_lsn_ready(expected_by_tenant)
            return {
                "fully_searchable": count_ready and fresh,
                "fully_indexed": count_ready and fresh,
                "additional_parameters": {},
            }

        count = stats.get("total_vector_count", 0)
        count_ready = count >= expected_count
        last_write_lsn = getattr(self, "_last_write_lsn", None)
        if last_write_lsn is None:
            return {
                "fully_searchable": count_ready,
                "fully_indexed": count_ready,
                "additional_parameters": {},
            }
        query_res = self.index.query(vector=self._readiness_probe_vector, top_k=1)
        indexed_lsn = self._extract_lsn(query_res, "x-pinecone-max-indexed-lsn")
        if indexed_lsn is None:
            return {
                "fully_searchable": count_ready,
                "fully_indexed": count_ready,
                "additional_parameters": {},
            }
        fresh = indexed_lsn >= last_write_lsn
        return {
            "fully_searchable": count_ready and fresh,
            "fully_indexed": count_ready and fresh,
            "additional_parameters": {},
        }

    def _expected_multitenant_counts(self, expected_count: int) -> dict[str, int]:
        insert_counts = getattr(self, "_multitenant_insert_counts", {})
        if insert_counts:
            return dict(insert_counts)
        tenant_labels = self.multitenant_tenant_labels
        tenant_count = len(tenant_labels)
        base_count = expected_count // tenant_count if tenant_count else 0
        remainder = expected_count % tenant_count if tenant_count else 0
        return {tenant: base_count + (1 if idx < remainder else 0) for idx, tenant in enumerate(tenant_labels)}

    def _multitenant_lsn_ready(self, expected_by_tenant: dict[str, int]) -> bool:
        last_write_lsn = getattr(self, "_last_write_lsn", None)
        if last_write_lsn is None:
            return True
        for tenant, expected_tenant_count in expected_by_tenant.items():
            if expected_tenant_count <= 0:
                continue
            query_res = self.index.query(
                vector=self._readiness_probe_vector,
                top_k=1,
                namespace=self._namespace_for_tenant(tenant),
            )
            indexed_lsn = self._extract_lsn(query_res, "x-pinecone-max-indexed-lsn")
            if indexed_lsn is not None and indexed_lsn < last_write_lsn:
                return False
        return True

    @staticmethod
    def _extract_lsn(response: Any, header_name: str) -> int | None:
        response_info = getattr(response, "_response_info", None)
        if not response_info:
            return None
        raw_headers = response_info.get("raw_headers", {})
        value = raw_headers.get(header_name.lower()) or raw_headers.get(header_name)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _record_write_lsn(self, write_lsn: int) -> None:
        if not hasattr(self, "_last_write_lsn_lock"):
            self._last_write_lsn_lock = threading.Lock()
        with self._last_write_lsn_lock:
            self._last_write_lsn = max(getattr(self, "_last_write_lsn", 0) or 0, write_lsn)

    @staticmethod
    def _matches(response: Any) -> list:
        if isinstance(response, dict):
            return response["matches"]
        return response.matches

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        tenant_labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception]:
        assert len(embeddings) == len(metadata)
        insert_count = 0
        try:
            for batch_start_offset in range(0, len(embeddings), self.batch_size):
                batch_end_offset = min(batch_start_offset + self.batch_size, len(embeddings))
                insert_datas = []
                for i in range(batch_start_offset, batch_end_offset):
                    metadata_dict = {self._scalar_id_field: metadata[i]}
                    if self.with_scalar_labels:
                        metadata_dict[self._scalar_label_field] = labels_data[i]
                    insert_data = (
                        str(metadata[i]),
                        embeddings[i],
                        metadata_dict,
                    )
                    insert_datas.append(insert_data)
                if tenant_labels_data is None:
                    upsert_res = self.index.upsert(insert_datas)
                    write_lsn = self._extract_lsn(upsert_res, "x-pinecone-request-lsn")
                    if write_lsn is not None:
                        self._record_write_lsn(write_lsn)
                else:
                    batch_tenant_labels = tenant_labels_data[batch_start_offset:batch_end_offset]
                    for tenant in sorted(set(batch_tenant_labels)):
                        tenant_insert_datas = [
                            insert_data
                            for insert_data, tenant_label in zip(insert_datas, batch_tenant_labels, strict=True)
                            if tenant_label == tenant
                        ]
                        self._multitenant_insert_counts[tenant] = self._multitenant_insert_counts.get(tenant, 0) + len(
                            tenant_insert_datas
                        )
                        upsert_res = self.index.upsert(
                            tenant_insert_datas,
                            namespace=self._namespace_for_tenant(tenant),
                        )
                        write_lsn = self._extract_lsn(upsert_res, "x-pinecone-request-lsn")
                        if write_lsn is not None:
                            self._record_write_lsn(write_lsn)
                insert_count += batch_end_offset - batch_start_offset
        except Exception as e:
            return insert_count, e
        return len(embeddings), None

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        timeout: int | None = None,
        payload_profile: PayloadProfile = PayloadProfile.IDS_ONLY,
        tenant: str | None = None,
    ) -> list[int]:
        pinecone_filters = self.expr
        include_metadata = payload_profile == PayloadProfile.SCALAR_LABEL
        include_values = payload_profile == PayloadProfile.VECTOR
        max_retries = int(os.getenv(PINECONE_QUERY_MAX_RETRIES_ENV, PINECONE_QUERY_DEFAULT_MAX_RETRIES))
        retry_sleep = float(
            os.getenv(PINECONE_QUERY_RETRY_SLEEP_ENV, PINECONE_QUERY_DEFAULT_RETRY_SLEEP_SECONDS),
        )
        for retry_idx in range(max_retries + 1):
            try:
                query_res = self.index.query(
                    top_k=k,
                    vector=query,
                    filter=pinecone_filters,
                    include_metadata=include_metadata,
                    include_values=include_values,
                    namespace=self._namespace_for_tenant(tenant),
                )
                res = self._matches(query_res)
                break
            except Exception as exc:
                if not self._is_rate_limited(exc) or retry_idx >= max_retries:
                    raise
                time.sleep(retry_sleep)
        return [int(one_res["id"]) for one_res in res]

    @staticmethod
    def _is_rate_limited(exc: Exception) -> bool:
        return getattr(exc, "status", None) == 429

    def prepare_filter(self, filters: Filter):
        if filters.type == FilterOp.NonFilter:
            self.expr = None
        elif filters.type == FilterOp.NumGE:
            self.expr = {self._scalar_id_field: {"$gte": filters.int_value}}
        elif filters.type == FilterOp.StrEqual:
            # both "in" and "==" are supported
            # for example, self.expr = {self._scalar_label_field: {"$in": [filters.label_value]}}
            self.expr = {self._scalar_label_field: {"$eq": filters.label_value}}
        else:
            msg = f"Not support Filter for Pinecone - {filters}"
            raise ValueError(msg)
