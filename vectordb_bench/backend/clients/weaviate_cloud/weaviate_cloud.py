"""Wrapper around the Weaviate vector database over VectorDB"""

import logging
from collections.abc import Iterable
from contextlib import contextmanager

import weaviate
from weaviate.exceptions import WeaviateBaseError
from weaviate.classes.config import Configure, DataType, Property, VectorDistances, Reconfigure

from ..api import DBCaseConfig, VectorDB
from .config import WeaviateConfig
from pydantic import SecretStr

log = logging.getLogger(__name__)


class WeaviateCloud(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config,
        db_case_config: DBCaseConfig,
        collection_name: str = "VectorDBBenchCollection",
        drop_old: bool = False,
        **kwargs,
    ):
        """Initialize wrapper around the Weaviate vector database (v4 client)."""
        # Normalize config to WeaviateConfig model (accept dict for backward compatibility)
        self.cfg = self._ensure_cfg(db_config)
        self.case_config = db_case_config
        self.collection_name = collection_name

        self._scalar_field = "key"
        self._vector_field = "vector"
        self._index_name = "vector_idx"

        # Open a short-lived admin connection to ensure collection exists
        http_host, http_port = self.cfg.host_port()
        grpc_pair = getattr(self.cfg, "grpc_host_port", None)
        grpc_host, grpc_port = self.cfg.grpc_host_port() if grpc_pair else (None, None)

        client = weaviate.connect_to_custom(
            http_host=http_host,
            http_port=http_port,
            grpc_host=grpc_host,
            grpc_port=grpc_port,
            http_secure=False,
            grpc_secure=False,
        )
        try:
            if drop_old and client.collections.exists(self.collection_name):
                log.info(f"weaviate client drop_old collection: {self.collection_name}")
                client.collections.delete(self.collection_name)
            self._create_collection(client)
        finally:
            client.close()

    @contextmanager
    def init(self) -> None:
        """
        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding()
        """
        http_host, http_port = self.cfg.host_port()
        grpc_pair = getattr(self.cfg, "grpc_host_port", None)
        grpc_host, grpc_port = self.cfg.grpc_host_port() if grpc_pair else (None, None)

        self.client = weaviate.connect_to_custom(
            http_host=http_host,
            http_port=http_port,
            grpc_host=grpc_host,
            grpc_port=grpc_port,
            http_secure=False,
            grpc_secure=False,
        )
        try:
            yield
        finally:
            self.client.close()
            self.client = None

    @staticmethod
    def _ensure_cfg(db_config) -> WeaviateConfig:
        """Accept either a WeaviateConfig instance or a plain dict (legacy path).

        When a dict is provided, reconstruct WeaviateConfig and normalize fields.
        """
        if isinstance(db_config, WeaviateConfig):
            return db_config
        if isinstance(db_config, dict):
            # Support both keys: 'api_key' and legacy 'auth_client_secret'
            api_key_val = db_config.get("api_key") or db_config.get("auth_client_secret") or "-"
            url_val = db_config.get("url", "")
            grpc_val = db_config.get("grpc_url")
            no_auth_val = bool(db_config.get("no_auth", False))

            # Ensure HTTP scheme for url
            if isinstance(url_val, str) and not (url_val.startswith("http://") or url_val.startswith("https://")):
                url_val = f"http://{url_val}" if url_val else "http://localhost:8080"

            return WeaviateConfig(
                db_label=db_config.get("db_label", "weaviate"),
                api_key=SecretStr(str(api_key_val) if api_key_val is not None else "-"),
                url=SecretStr(url_val),
                no_auth=no_auth_val,
                grpc_url=SecretStr(str(grpc_val)) if grpc_val else None,
            )
        # Unexpected type; raise for clarity
        raise TypeError(f"Unsupported db_config type for WeaviateCloud: {type(db_config)}")

    def optimize(self, data_size: int | None = None):
        col = self.client.collections.get(self.collection_name)
        # Update search ef when provided using v4 Configure helper
        try:
            ef_val = self.case_config.search_param().get("ef")
        except Exception:
            ef_val = None
        if ef_val is not None:
            col.config.update(
                vector_config=Reconfigure.Vectors.update(
                    name="default",
                    vector_index_config=Reconfigure.VectorIndex.hnsw(
                        ef=ef_val,
                    ),
                ),
            )

    def _create_collection(self, client) -> None:
        if not client.collections.exists(self.collection_name):
            log.info(f"Create collection: {self.collection_name}")

            # Map metric to Weaviate v4 distance enum
            try:
                metric = self.case_config.parse_metric()
            except Exception:
                metric = "cosine"
            distance_enum = {
                "cosine": VectorDistances.COSINE,
                "dot": VectorDistances.DOT,
                "l2-squared": VectorDistances.L2_SQUARED,
            }.get(metric, VectorDistances.COSINE)

            # Optional HNSW params
            ef_construction = getattr(self.case_config, "efConstruction", None)
            max_connections = getattr(self.case_config, "maxConnections", None)

            vector_index_cfg = Configure.VectorIndex.hnsw(
                distance_metric=distance_enum,
                **({"ef_construction": ef_construction} if ef_construction is not None else {}),
                **({"max_connections": max_connections} if max_connections is not None else {}),
            )

            # Build v4.18.3-compliant vector_config using helper for self-provided vectors
            vector_config = Configure.Vectors.self_provided(vector_index_config=vector_index_cfg)

            client.collections.create(
                name=self.collection_name,
                properties=[Property(name=self._scalar_field, data_type=DataType.INT)],
                vector_config=vector_config,
            )

    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> tuple[int, Exception | None]:
        """Insert embeddings into Weaviate"""
        col = self.client.collections.get(self.collection_name)
        try:
            # v4.18 expects vectors to be supplied separately from properties.
            props_list = [{self._scalar_field: metadata[i]} for i in range(len(metadata))]
            vecs_list = [embeddings[i] for i in range(len(metadata))]

            # Stream with fixed-size batches to avoid large single RPCs and reduce backpressure
            batch_size = kwargs.get("_weaviate_batch_size", 1000)
            concurrent_requests = kwargs.get("_weaviate_concurrency", 2)

            inserted = 0
            idx = 0
            total = len(props_list)
            while idx < total:
                end = min(idx + batch_size, total)
                with col.batch.fixed_size(batch_size=(end - idx), concurrent_requests=concurrent_requests) as batch:
                    for i in range(idx, end):
                        batch.add_object(properties=props_list[i], vector=vecs_list[i])
                        inserted += 1
                # Lightweight progress log every 50k rows
                if inserted % 50000 == 0:
                    log.info(f"Weaviate inserted {inserted}/{total} objects...")
                idx = end
            return (inserted, None)
        except WeaviateBaseError as e:
            log.warning(f"Failed to insert data, error: {e!s}")
            return (0, e)

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
    ) -> list[int]:
        """Perform a search on a query embedding and return result keys.
        Should call self.init() first.
        """
        col = self.client.collections.get(self.collection_name)

        where = None
        if filters and "id" in filters:
            where = {
                "operator": "GreaterThanEqual",
                "path": [self._scalar_field],
                "valueInt": int(filters["id"]),
            }

        # weaviate-client v4.18.3: pass the query vector positionally
        res = col.query.near_vector(query, limit=k, filters=where)
        return [obj.properties.get(self._scalar_field) for obj in res.objects]
