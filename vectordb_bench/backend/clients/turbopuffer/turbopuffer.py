"""Wrapper around the TurboPuffer vector database over VectorDB"""

import logging
import time
from contextlib import contextmanager

import turbopuffer as tpuf

from vectordb_bench.backend.clients.turbopuffer.config import TurboPufferIndexConfig
from vectordb_bench.backend.filter import Filter, FilterOp
from vectordb_bench.backend.payload import PayloadProfile

from ..api import VectorDB

log = logging.getLogger(__name__)
TURBOPUFFER_SEARCHABLE_UNINDEXED_BYTES = 2 * 1024 * 1024 * 1024


class TurboPuffer(VectorDB):
    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: TurboPufferIndexConfig,
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        self.api_key = db_config.get("api_key", "")
        self.region = db_config.get("region", "")
        self.api_base_url = db_config.get("api_base_url")
        self.namespace = db_config.get("namespace", "")
        self.multitenant_namespace_prefix = db_config.get("multitenant_namespace_prefix", "vdbbench_mt_")
        self.multitenant_tenant_labels: list[str] = kwargs.get("multitenant_tenant_labels", [])
        self._multitenant_touched_tenants: set[str] = set()
        self._ns_cache = {}
        self.db_case_config = db_case_config
        self.metric = db_case_config.parse_metric()

        self._vector_field = "vector"
        self._scalar_id_field = "id"
        self._scalar_label_field = "label"
        self._scalar_payload_label_field = db_config.get("scalar_payload_label_field", self._scalar_label_field)

        self.with_scalar_labels = with_scalar_labels
        self.expr = None

        if drop_old:
            tmp_client = self._create_client()
            if self.multitenant_tenant_labels:
                for tenant in self.multitenant_tenant_labels:
                    try:
                        tmp_client.namespace(self._namespace_name_for_tenant(tenant)).delete_all()
                    except Exception as e:
                        log.warning(f"Failed to delete multitenant namespace {tenant}. Error: {e}")
            else:
                log.info(f"Drop old. delete the namespace: {self.namespace}")
                ns = tmp_client.namespace(self.namespace)
                try:
                    ns.delete_all()
                except Exception as e:
                    log.warning(f"Failed to delete all. Error: {e}")
            tmp_client = None

    def _create_client(self) -> tpuf.Turbopuffer:
        client_kwargs = {"api_key": self.api_key, "region": self.region}
        if self.api_base_url:
            client_kwargs["base_url"] = self.api_base_url
        return tpuf.Turbopuffer(**client_kwargs)

    @contextmanager
    def init(self):
        self.client = self._create_client()
        self._ns_cache = {}
        self.ns = self.client.namespace(self.namespace)
        yield

    def supports_multitenant(self) -> bool:
        return True

    def set_multitenant_context(self, tenant_labels: list[str]) -> None:
        self.multitenant_tenant_labels = tenant_labels

    def _namespace_name_for_tenant(self, tenant: str | None) -> str:
        if tenant is None:
            return self.namespace
        return f"{self.multitenant_namespace_prefix}{tenant}"

    def _namespace_for_tenant(self, tenant: str | None):
        name = self._namespace_name_for_tenant(tenant)
        ns = self._ns_cache.get(name)
        if ns is None:
            ns = self.client.namespace(name)
            self._ns_cache[name] = ns
        return ns

    def optimize(self, data_size: int | None = None):
        # turbopuffer responds to the request
        #   once the cache warming operation has been started.
        # It does not wait for the operation to complete,
        #   which can take multiple minutes for large namespaces.
        self.ns.hint_cache_warm()
        log.info(f"warming up but no api waiting for complete. just sleep {self.db_case_config.time_wait_warmup}s")
        time.sleep(self.db_case_config.time_wait_warmup)

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        tenant_labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception]:
        vectors = [embedding.tolist() if hasattr(embedding, "tolist") else embedding for embedding in embeddings]
        try:
            if tenant_labels_data is not None:
                inserted = 0
                for tenant in sorted(set(tenant_labels_data)):
                    self._multitenant_touched_tenants.add(tenant)
                    idxs = [i for i, label in enumerate(tenant_labels_data) if label == tenant]
                    upsert_columns = {
                        self._scalar_id_field: [metadata[i] for i in idxs],
                        self._vector_field: [vectors[i] for i in idxs],
                    }
                    self._namespace_for_tenant(tenant).write(
                        upsert_columns=upsert_columns,
                        distance_metric=self.metric,
                        disable_backpressure=self.db_case_config.disable_backpressure,
                    )
                    inserted += len(idxs)
                return inserted, None
            if self.with_scalar_labels:
                self.ns.write(
                    upsert_columns={
                        self._scalar_id_field: metadata,
                        self._vector_field: vectors,
                        self._scalar_label_field: labels_data,
                    },
                    distance_metric=self.metric,
                    disable_backpressure=self.db_case_config.disable_backpressure,
                )
            else:
                self.ns.write(
                    upsert_columns={
                        self._scalar_id_field: metadata,
                        self._vector_field: vectors,
                    },
                    distance_metric=self.metric,
                    disable_backpressure=self.db_case_config.disable_backpressure,
                )
        except Exception as e:
            log.warning(f"Failed to insert. Error: {e}")
            return 0, e
        return len(embeddings), None

    def supports_payload_profile(self, payload_profile: PayloadProfile) -> bool:
        return payload_profile in {
            PayloadProfile.IDS_ONLY,
            PayloadProfile.SCALAR_LABEL,
            PayloadProfile.VECTOR,
        }

    def poll_insert_readiness(self, expected_count: int) -> dict:
        if getattr(self, "multitenant_tenant_labels", []):
            unindexed_by_tenant = {}
            tenant_labels = (
                sorted(getattr(self, "_multitenant_touched_tenants", set())) or self.multitenant_tenant_labels
            )
            for tenant in tenant_labels:
                metadata = self._namespace_for_tenant(tenant).metadata()
                if not isinstance(metadata, dict):
                    metadata = metadata.model_dump() if hasattr(metadata, "model_dump") else vars(metadata)
                index = metadata.get("index", {})
                if not isinstance(index, dict):
                    index = index.model_dump() if hasattr(index, "model_dump") else vars(index)
                unindexed_by_tenant[tenant] = metadata.get("unindexed_bytes", index.get("unindexed_bytes", 0))
            max_unindexed_bytes = max(unindexed_by_tenant.values(), default=0)
            return {
                "fully_searchable": max_unindexed_bytes <= TURBOPUFFER_SEARCHABLE_UNINDEXED_BYTES,
                "fully_indexed": max_unindexed_bytes == 0,
                "additional_parameters": {
                    "disable_backpressure": self.db_case_config.disable_backpressure,
                    "max_unindexed_bytes": max_unindexed_bytes,
                },
            }
        metadata = self.ns.metadata()
        if not isinstance(metadata, dict):
            metadata = metadata.model_dump() if hasattr(metadata, "model_dump") else vars(metadata)
        index = metadata.get("index", {})
        if not isinstance(index, dict):
            index = index.model_dump() if hasattr(index, "model_dump") else vars(index)
        unindexed_bytes = metadata.get("unindexed_bytes", index.get("unindexed_bytes", 0))
        return {
            "fully_searchable": unindexed_bytes <= TURBOPUFFER_SEARCHABLE_UNINDEXED_BYTES,
            "fully_indexed": unindexed_bytes == 0,
            "additional_parameters": {"disable_backpressure": self.db_case_config.disable_backpressure},
        }

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        timeout: int | None = None,
        payload_profile: PayloadProfile = PayloadProfile.IDS_ONLY,
        tenant: str | None = None,
    ) -> list[int]:
        query_kwargs = {
            "rank_by": ("vector", "ANN", query),
            "top_k": k,
            "filters": self.expr,
        }
        if payload_profile == PayloadProfile.VECTOR:
            query_kwargs["include_attributes"] = [self._vector_field]
        elif payload_profile == PayloadProfile.SCALAR_LABEL:
            query_kwargs["include_attributes"] = [self._scalar_payload_label_field]
        res = self._namespace_for_tenant(tenant).query(**query_kwargs)
        return [int(row.id) for row in res.rows] if res.rows is not None else []

    def prepare_filter(self, filters: Filter):
        if filters.type == FilterOp.NonFilter:
            self.expr = None
        elif filters.type == FilterOp.NumGE:
            self.expr = (self._scalar_id_field, "Gte", filters.int_value)
        elif filters.type == FilterOp.StrEqual:
            self.expr = (self._scalar_payload_label_field, "Eq", filters.label_value)
        else:
            msg = f"Not support Filter for TurboPuffer - {filters}"
            raise ValueError(msg)
