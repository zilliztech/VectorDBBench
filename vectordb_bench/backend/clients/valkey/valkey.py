import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import numpy as np
from glide_sync import (
    AdvancedGlideClientConfiguration,
    AdvancedGlideClusterClientConfiguration,
    Batch,
    ClusterBatch,
    ClusterScanCursor,
    DataType,
    DistanceMetricType,
    FtCreateOptions,
    FtSearchLimit,
    FtSearchOptions,
    GlideClient,
    GlideClientConfiguration,
    GlideClusterClient,
    GlideClusterClientConfiguration,
    NodeAddress,
    NumericField,
    ServerCredentials,
    TagField,
    VectorAlgorithm,
    VectorField,
    VectorFieldAttributesHnsw,
    VectorType,
    ft,
)

from vectordb_bench import config
from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import ValkeyHNSWConfig

log = logging.getLogger(__name__)
ValkeyConnection = GlideClient | GlideClusterClient
ValkeyBatch = Batch | ClusterBatch


class Valkey(VectorDB):
    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: ValkeyHNSWConfig,
        collection_name: str = "vdbbench_valkey",
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        self.name = "Valkey"
        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = collection_name
        self.key_prefix = f"{self.collection_name}:"
        self.with_scalar_labels = with_scalar_labels
        self.filter_query = "*"
        self.cluster_mode = not self.db_config["cmd"]
        self.conn: ValkeyConnection | None = None

        conn = self._create_client()
        try:
            if drop_old:
                self._drop_index(conn)
            self._make_index(dim, conn)
        finally:
            conn.close()

    def _create_client(self) -> ValkeyConnection:
        password = self.db_config["password"]
        config_kwargs = {
            "addresses": [NodeAddress(self.db_config["host"], self.db_config["port"])],
            "use_tls": self.db_config["ssl"],
            "credentials": ServerCredentials(password=password) if password else None,
            "request_timeout": self.db_config["request_timeout_ms"],
        }
        connection_timeout = self.db_config["connection_timeout_ms"]
        if self.cluster_mode:
            config_kwargs["advanced_config"] = AdvancedGlideClusterClientConfiguration(
                connection_timeout=connection_timeout
            )
            return GlideClusterClient.create(GlideClusterClientConfiguration(**config_kwargs))
        config_kwargs["advanced_config"] = AdvancedGlideClientConfiguration(connection_timeout=connection_timeout)
        return GlideClient.create(GlideClientConfiguration(database_id=0, **config_kwargs))

    def _drop_index(self, conn: ValkeyConnection) -> None:
        if not self._index_exists(conn):
            log.info("Valkey index did not exist: %s", self.collection_name)
        else:
            ft.dropindex(conn, self.collection_name)
            log.info("Valkey dropped old index: %s", self.collection_name)
        self._delete_prefixed_documents(conn)

    def _index_exists(self, conn: ValkeyConnection) -> bool:
        encoded_name = self.collection_name.encode()
        return any(index_name in (self.collection_name, encoded_name) for index_name in ft.list(conn))

    def _new_batch(self) -> ValkeyBatch:
        return ClusterBatch(is_atomic=False) if self.cluster_mode else Batch(is_atomic=False)

    def _scan_keys(self, conn: ValkeyConnection) -> Generator[bytes, None, None]:
        match = f"{self.key_prefix}*"
        if self.cluster_mode:
            cursor = ClusterScanCursor()
            while not cursor.is_finished():
                cursor, keys = conn.scan(cursor, match=match, count=config.NUM_PER_BATCH)
                yield from keys
            return

        cursor: str | bytes | int = "0"
        while True:
            cursor, keys = conn.scan(cursor, match=match, count=config.NUM_PER_BATCH)
            yield from keys
            if cursor in {"0", b"0", 0}:
                return

    def _delete_prefixed_documents(self, conn: ValkeyConnection) -> None:
        deleted = 0
        pending = 0
        batch = self._new_batch()
        for key in self._scan_keys(conn):
            batch.unlink([key])
            deleted += 1
            pending += 1
            if pending == config.NUM_PER_BATCH:
                conn.exec(batch, raise_on_error=True)
                batch = self._new_batch()
                pending = 0
        if pending:
            conn.exec(batch, raise_on_error=True)
        if deleted:
            log.info("Valkey deleted %s documents with prefix %s", deleted, self.key_prefix)

    def _make_index(self, vector_dimensions: int, conn: ValkeyConnection) -> None:
        if self._index_exists(conn):
            return

        index_param = self.case_config.index_param()
        schema = [
            NumericField("metadata"),
            VectorField(
                "vector",
                VectorAlgorithm.HNSW,
                VectorFieldAttributesHnsw(
                    dimensions=vector_dimensions,
                    distance_metric=DistanceMetricType(index_param["metric_type"]),
                    type=VectorType.FLOAT32,
                    number_of_edges=index_param["params"]["M"],
                    vectors_examined_on_construction=index_param["params"]["efConstruction"],
                ),
            ),
        ]
        if self.with_scalar_labels:
            schema.append(TagField("label"))

        ft.create(
            conn,
            self.collection_name,
            schema,
            FtCreateOptions(data_type=DataType.HASH, prefixes=[self.key_prefix]),
        )

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        conn = self._create_client()
        self.conn = conn
        try:
            yield
        finally:
            self.conn = None
            conn.close()

    def optimize(self, data_size: int | None = None) -> None:
        """Valkey does not require a post-load optimization step."""
        return

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[int, Exception | None]:
        assert self.conn is not None, "Please call self.init() before insert_embeddings"
        if len(embeddings) != len(metadata):
            msg = "Embeddings and metadata must have the same length"
            raise ValueError(msg)
        if self.with_scalar_labels and (labels_data is None or len(labels_data) != len(embeddings)):
            msg = "Scalar labels must be provided for every embedding"
            raise ValueError(msg)
        if not embeddings:
            return 0, None

        try:
            batch = self._new_batch()
            for i, embedding in enumerate(embeddings):
                doc_id = metadata[i]
                mapping = {
                    "metadata": str(doc_id),
                    "vector": np.asarray(embedding, dtype=np.float32).tobytes(),
                }
                if self.with_scalar_labels and labels_data is not None:
                    mapping["label"] = labels_data[i]
                batch.hset(f"{self.key_prefix}{doc_id}", mapping)
            self.conn.exec(batch, raise_on_error=True)
        except Exception as e:
            return 0, e

        return len(embeddings), None

    def prepare_filter(self, filters: Filter) -> None:
        if filters.type == FilterOp.NonFilter:
            self.filter_query = "*"
        elif filters.type == FilterOp.NumGE:
            self.filter_query = f"@metadata:[{filters.int_value} +inf]"
        elif filters.type == FilterOp.StrEqual:
            label = filters.label_value.replace("\\", "\\\\").replace(" ", "\\ ")
            self.filter_query = f"@label:{{{label}}}"
        else:
            msg = f"Unsupported filter for Valkey: {filters}"
            raise ValueError(msg)

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> list[int]:
        assert self.conn is not None, "Please call self.init() before search_embedding"

        ef_runtime = self.case_config.search_param()["params"]["ef"]
        knn = f"KNN {k} @vector $vec EF_RUNTIME {ef_runtime}"
        query_vector = np.asarray(query, dtype=np.float32).tobytes()
        result = ft.search(
            self.conn,
            self.collection_name,
            f"{self.filter_query}=>[{knn}]",
            FtSearchOptions(
                nocontent=True,
                timeout=timeout,
                params={"vec": query_vector},
                limit=FtSearchLimit(0, k),
                dialect=2,
            ),
        )
        documents = result[1] if len(result) > 1 else {}
        return [self._parse_document_id(document_id) for document_id in documents]

    def _parse_document_id(self, document_id: str | bytes) -> int:
        if isinstance(document_id, bytes):
            document_id = document_id.decode()
        if document_id.startswith(self.key_prefix):
            document_id = document_id[len(self.key_prefix) :]
        return int(document_id)
