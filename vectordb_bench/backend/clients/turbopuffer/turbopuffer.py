"""Wrapper around the TurboPuffer vector database over VectorDB"""

import logging
import time
from contextlib import contextmanager
from json import dumps, loads
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

import turbopuffer as tpuf

from vectordb_bench.backend.clients.turbopuffer.config import (
    TurboPufferFtsConfig,
    TurboPufferIndexConfig,
    TurboPufferMultitenantWarmupPolicy,
)
from vectordb_bench.backend.filter import Filter, FilterOp
from vectordb_bench.backend.payload import PayloadProfile

from ..api import PartialInsertError, VectorDB

log = logging.getLogger(__name__)
TURBOPUFFER_SEARCHABLE_UNINDEXED_BYTES = 2 * 1024 * 1024 * 1024
PINNING_POLL_INTERVAL = 10
PINNING_TIMEOUT = 45 * 60


def namespace_metadata_request(
    api_key: str,
    region: str,
    namespace: str,
    method: str,
    payload: dict[str, Any] | None = None,
    api_base_url: str | None = None,
) -> dict:
    base_url = api_base_url or f"https://{region}.turbopuffer.com"
    url = f"{base_url.rstrip('/')}/v1/namespaces/{quote(namespace, safe='')}/metadata"
    req = Request(  # noqa: S310
        url,
        data=dumps(payload).encode() if payload is not None else None,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(req, timeout=60) as resp:  # noqa: S310
            return loads(resp.read().decode() or "{}")
    except HTTPError as e:
        detail = e.read().decode(errors="replace")
        msg = f"Failed to update TurboPuffer namespace metadata: {e.code} {detail}"
        raise RuntimeError(msg) from e


def wait_for_namespace_pinning(
    api_key: str,
    region: str,
    namespace: str,
    replicas: int | None,
    api_base_url: str | None = None,
    timeout: int = PINNING_TIMEOUT,
) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        meta = namespace_metadata_request(api_key, region, namespace, "GET", api_base_url=api_base_url)
        pinning = meta.get("pinning")
        if replicas is None:
            if pinning is None:
                return meta
        else:
            status = pinning.get("status", {}) if isinstance(pinning, dict) else {}
            if pinning and pinning.get("replicas") == replicas and status.get("ready_replicas") == replicas:
                return meta
        if time.monotonic() >= deadline:
            msg = f"Timed out waiting for TurboPuffer pinning state on namespace {namespace}"
            raise TimeoutError(msg)
        log.info("Waiting for TurboPuffer pinning state on %s: %s", namespace, pinning)
        time.sleep(PINNING_POLL_INTERVAL)


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
        db_case_config: TurboPufferIndexConfig | TurboPufferFtsConfig,
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
        self.pin_namespace = db_config.get("pin_namespace", False)
        self.pin_replicas = db_config.get("pin_replicas", 1)
        self.pin_timeout = db_config.get("pin_timeout", PINNING_TIMEOUT)
        self._pinning_applied = False
        self.db_case_config = db_case_config
        self._is_fts = isinstance(db_case_config, TurboPufferFtsConfig)
        self.metric = None if self._is_fts else db_case_config.parse_metric()

        self._vector_field = "vector"
        self._scalar_id_field = "id"
        self._scalar_label_field = "label"
        self._scalar_payload_label_field = db_config.get("scalar_payload_label_field", self._scalar_label_field)
        self._text_field = "text"

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

    def _apply_namespace_pinning(self):
        if not self.pin_namespace or self._pinning_applied:
            return
        for namespace in self._target_namespaces_for_pinning():
            namespace_metadata_request(
                self.api_key,
                self.region,
                namespace,
                "PATCH",
                {"pinning": {"replicas": self.pin_replicas}},
                self.api_base_url,
            )
            meta = wait_for_namespace_pinning(
                self.api_key,
                self.region,
                namespace,
                self.pin_replicas,
                self.api_base_url,
                self.pin_timeout,
            )
            pinning = meta.get("pinning", {})
            status = pinning.get("status", {}) if isinstance(pinning, dict) else {}
            log.info(
                "TurboPuffer pinning requested for %s: replicas=%s ready_replicas=%s",
                namespace,
                pinning.get("replicas", self.pin_replicas) if isinstance(pinning, dict) else self.pin_replicas,
                status.get("ready_replicas"),
            )
        self._pinning_applied = True

    def _target_namespaces_for_pinning(self) -> list[str]:
        if self.multitenant_tenant_labels:
            return [self._namespace_name_for_tenant(tenant) for tenant in self.multitenant_tenant_labels]
        return [self.namespace]

    @classmethod
    def supports_full_text_search(cls) -> bool:
        return True

    @contextmanager
    def init(self):
        self.client = self._create_client()
        self._ns_cache = {}
        self.ns = self.client.namespace(self.namespace)
        self._apply_namespace_pinning()
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
        warmed_namespaces = self._warmup_target_namespaces()
        for namespace in warmed_namespaces:
            self._namespace_for_tenant(namespace).hint_cache_warm()
        if not warmed_namespaces:
            log.info("TurboPuffer cache warmup skipped")
            return
        log.info(f"warming up but no api waiting for complete. just sleep {self.db_case_config.time_wait_warmup}s")
        time.sleep(self.db_case_config.time_wait_warmup)

    def _warmup_target_namespaces(self) -> list[str | None]:
        if not self.multitenant_tenant_labels:
            return [None]
        policy = getattr(
            self.db_case_config,
            "multitenant_warmup_policy",
            TurboPufferMultitenantWarmupPolicy.NONE,
        )
        if policy == TurboPufferMultitenantWarmupPolicy.ALL:
            return self.multitenant_tenant_labels
        return []

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        tenant_labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception]:
        vectors = [embedding.tolist() if hasattr(embedding, "tolist") else embedding for embedding in embeddings]
        if tenant_labels_data is not None:
            inserted = 0
            successful_tenants: dict[str, int] = {}
            for tenant in sorted(set(tenant_labels_data)):
                self._multitenant_touched_tenants.add(tenant)
                idxs = [i for i, label in enumerate(tenant_labels_data) if label == tenant]
                try:
                    upsert_columns = {
                        self._scalar_id_field: [metadata[i] for i in idxs],
                        self._vector_field: [vectors[i] for i in idxs],
                    }
                    if self.with_scalar_labels:
                        upsert_columns[self._scalar_payload_label_field] = [labels_data[i] for i in idxs]
                    self._namespace_for_tenant(tenant).write(
                        upsert_columns=upsert_columns,
                        distance_metric=self.metric,
                        disable_backpressure=self.db_case_config.disable_backpressure,
                    )
                except Exception as e:
                    msg = (
                        "TurboPuffer multitenant insert failed for "
                        f"tenant={tenant} after writing {inserted} rows; "
                        f"successful_tenants={successful_tenants}; "
                        f"failed_tenant_count={len(idxs)}"
                    )
                    err = PartialInsertError(
                        msg,
                        inserted_count=inserted,
                        successful_tenants=successful_tenants,
                        failed_tenant=tenant,
                        failed_tenant_count=len(idxs),
                        cause=e,
                    )
                    log.warning(f"Failed to insert. Error: {err}")
                    return inserted, err
                inserted += len(idxs)
                successful_tenants[tenant] = len(idxs)
            return inserted, None
        try:
            if self.with_scalar_labels:
                self.ns.write(
                    upsert_columns={
                        self._scalar_id_field: metadata,
                        self._vector_field: vectors,
                        self._scalar_payload_label_field: labels_data,
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

    def insert_documents(
        self,
        texts: list[str],
        doc_ids: list[str],
        **kwargs,
    ) -> tuple[int, Exception | None]:
        if not getattr(self, "_is_fts", False):
            msg = "TurboPuffer full-text insert requires TurboPufferFtsConfig"
            raise RuntimeError(msg)
        assert self.ns is not None, "should self.init() first"

        docs = list(texts)
        if len(docs) != len(doc_ids):
            msg = f"Mismatch between texts ({len(docs)}) and doc_ids ({len(doc_ids)}) lengths"
            raise ValueError(msg)

        text_field = self._text_field
        try:
            self.ns.write(
                upsert_columns={
                    self._scalar_id_field: [str(doc_id) for doc_id in doc_ids],
                    text_field: docs,
                },
                schema={
                    text_field: {
                        "type": "string",
                        "full_text_search": True,
                    }
                },
            )
        except Exception as e:
            log.warning(f"Failed to insert FTS docs. Error: {e}")
            return 0, e
        return len(docs), None

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

    def search_documents(
        self,
        query: str,
        k: int = 100,
        **kwargs,
    ) -> list[str]:
        if not getattr(self, "_is_fts", False):
            msg = "TurboPuffer full-text search requires TurboPufferFtsConfig"
            raise RuntimeError(msg)
        assert self.ns is not None, "should self.init() first"

        res = self.ns.query(
            rank_by=(self._text_field, "BM25", query),
            top_k=k,
        )
        rows = getattr(res, "rows", None) or []
        return [str(row.id) for row in rows if getattr(row, "id", None) is not None]

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
