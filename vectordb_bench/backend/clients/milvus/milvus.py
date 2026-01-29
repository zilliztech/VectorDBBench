"""Wrapper around the Milvus vector database over VectorDB"""

import logging
import time
from collections.abc import Iterable
from contextlib import contextmanager

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    MilvusException,
    utility,
)

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import MilvusFtsConfig, MilvusIndexConfig

log = logging.getLogger(__name__)

MILVUS_LOAD_REQS_SIZE = 1.5 * 1024 * 1024
MILVUS_FTS_BATCH_SIZE = 1000


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
        self._scalar_labels_index_name = "labels_idx"

        # Configure based on FTS or vector search
        self._is_fts = isinstance(self.case_config, MilvusFtsConfig)
        if self._is_fts:
            self.batch_size = fts_batch_size or MILVUS_FTS_BATCH_SIZE
            self._primary_field, self._text_field, self._sparse_field = "doc_id", "text", "sparse_vector"
            self._main_index_name, self._sort_index_name, self._sort_index_field = (
                "sparse_vector_idx",
                "doc_id_sort_idx",
                "doc_id",
            )
        else:
            self.batch_size = int(MILVUS_LOAD_REQS_SIZE / (dim * 4))
            self._primary_field, self._scalar_id_field, self._vector_field = "pk", "id", "vector"
            self._main_index_name, self._sort_index_name, self._sort_index_field = "vector_idx", "id_sort_idx", "id"

        from pymilvus import connections

        connections.connect(
            uri=self.db_config.get("uri"),
            user=self.db_config.get("user"),
            password=self.db_config.get("password"),
            timeout=30,
        )
        if drop_old and utility.has_collection(self.collection_name):
            log.info(f"{self.name} client drop_old collection: {self.collection_name}")
            utility.drop_collection(self.collection_name)

        if not utility.has_collection(self.collection_name):
            if self._is_fts:
                fields = [
                    FieldSchema(name=self._primary_field, dtype=DataType.INT64, is_primary=True),
                    FieldSchema(
                        name=self._text_field,
                        dtype=DataType.VARCHAR,
                        max_length=65535,
                        enable_analyzer=True,
                        enable_match=True,
                        analyzer_params={"type": "english"},
                    ),
                    FieldSchema(name=self._sparse_field, dtype=DataType.SPARSE_FLOAT_VECTOR),
                ]
                if self.with_scalar_labels:
                    fields.append(
                        FieldSchema(
                            self._scalar_label_field,
                            DataType.VARCHAR,
                            max_length=256,
                        )
                    )
                bm25_function = Function(
                    name="text_bm25_emb",
                    function_type=FunctionType.BM25,
                    input_field_names=[self._text_field],
                    output_field_names=[self._sparse_field],
                    params={},  # BM25 function does not accept parameters
                )
                schema = CollectionSchema(fields=fields, functions=[bm25_function])
                collection_kwargs = {"name": self.collection_name, "schema": schema}
            else:
                fields = [
                    FieldSchema(self._primary_field, DataType.INT64, is_primary=True),
                    FieldSchema(self._scalar_id_field, DataType.INT64),
                    FieldSchema(self._vector_field, DataType.FLOAT_VECTOR, dim=dim),
                ]
                if self.with_scalar_labels:
                    is_partition_key = db_case_config.use_partition_key
                    log.info(f"with_scalar_labels, add a new varchar field, as partition_key: {is_partition_key}")
                    fields.append(
                        FieldSchema(
                            self._scalar_label_field,
                            DataType.VARCHAR,
                            max_length=256,
                            is_partition_key=is_partition_key,
                        )
                    )
                schema = CollectionSchema(fields)
                collection_kwargs = {
                    "name": self.collection_name,
                    "schema": schema,
                    "consistency_level": "Session",
                    "num_shards": self.db_config.get("num_shards", 1),
                }

            log.info(f"{self.name} create collection: {self.collection_name}")
            col = Collection(**collection_kwargs)

            self.create_index()
            col.load(replica_number=self.db_config.get("replica_number", 1))

        connections.disconnect("default")

    def create_index(self):
        col = Collection(self.collection_name)
        if self._is_fts:
            col.create_index(
                self._sparse_field,
                index_params=self.case_config.index_param(),
                index_name=self._main_index_name,
            )
        else:
            col.create_index(
                self._vector_field,
                self.case_config.index_param(),
                index_name=self._main_index_name,
            )
        # Create sort index (primary field for FTS, scalar_id for vector search)
        col.create_index(
            self._sort_index_field,
            index_params={
                "index_type": "STL_SORT",
            },
            index_name=self._sort_index_name,
        )
        # scalar index for varchar (label-filter)
        if self.with_scalar_labels:
            col.create_index(
                self._scalar_label_field,
                index_params={
                    "index_type": "BITMAP",
                },
                index_name=self._scalar_labels_index_name,
            )

    @contextmanager
    def init(self):
        """
        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding()
        """
        from pymilvus import connections

        self.col: Collection | None = None

        connections.connect(**self.db_config, timeout=60)
        # Grab the existing colection with connections
        self.col = Collection(self.collection_name)

        try:
            yield
        finally:
            connections.disconnect("default")
            self.col = None

    def _optimize(self):
        log.info(f"{self.name} optimizing before search")
        self._post_insert()
        try:
            self.col.load(refresh=True)
        except Exception as e:
            log.warning(f"{self.name} optimize error: {e}")
            raise e from None

    def _post_insert(self):
        try:
            self.col.flush()
            # wait for index done and load refresh
            self.create_index()

            # Determine compaction behavior based on mode
            index_name = self._main_index_name
            skip_compaction = False if self._is_fts else self.case_config.is_gpu_index

            # Wait for index building to complete
            utility.wait_for_index_building_complete(self.collection_name, index_name=index_name)

            def wait_index():
                while True:
                    progress = utility.index_building_progress(self.collection_name, index_name=index_name)
                    if progress.get("pending_index_rows", -1) == 0:
                        break
                    time.sleep(5)

            wait_index()

            # Perform compaction if not skipped
            if skip_compaction:
                log.debug("skip compaction for gpu index type.")
            else:
                try:
                    self.col.compact()
                    self.col.wait_for_compaction_completed()
                    log.info("compactation completed. waiting for the rest of index buliding.")
                except Exception as e:
                    log.warning(f"{self.name} compact error: {e}")
                    if hasattr(e, "code"):
                        if e.code().name == "PERMISSION_DENIED":
                            log.warning("Skip compact due to permission denied.")
                    else:
                        raise e from e
                wait_index()
        except Exception as e:
            log.warning(f"{self.name} optimize error: {e}")
            raise e from None

    def optimize(self, data_size: int | None = None):
        if self.col is None:
            msg = "Collection not initialized. Call init() first."
            raise RuntimeError(msg)
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
        **kwargs,
    ) -> tuple[int, Exception]:
        """Insert embeddings into Milvus. should call self.init() first"""
        # use the first insert_embeddings to init collection
        if self.col is None:
            msg = "Collection not initialized. Call init() first."
            raise RuntimeError(msg)
        if len(embeddings) != len(metadata):
            msg = f"Mismatch between embeddings ({len(embeddings)}) and metadata ({len(metadata)}) lengths"
            raise ValueError(msg)
        insert_count = 0
        try:
            for batch_start_offset in range(0, len(embeddings), self.batch_size):
                batch_end_offset = min(batch_start_offset + self.batch_size, len(embeddings))
                insert_data = [
                    metadata[batch_start_offset:batch_end_offset],
                    metadata[batch_start_offset:batch_end_offset],
                    embeddings[batch_start_offset:batch_end_offset],
                ]
                if self.with_scalar_labels:
                    insert_data.append(labels_data[batch_start_offset:batch_end_offset])
                res = self.col.insert(insert_data)
                insert_count += len(res.primary_keys)
        except MilvusException as e:
            log.info(f"Failed to insert data: {e}")
            return insert_count, e
        return insert_count, None

    def insert_fulltext(
        self,
        texts: Iterable[str],
        doc_ids: list[int],
        **kwargs,
    ) -> tuple[int, Exception | None]:
        """Insert fulltext documents into Milvus FTS collection. should call self.init() first"""
        if not self._is_fts:
            msg = "insert_fulltext is only valid in FTS mode"
            raise RuntimeError(msg)
        if self.col is None:
            msg = "Collection not initialized. Call init() first."
            raise RuntimeError(msg)

        docs = list(texts)
        if len(docs) != len(doc_ids):
            msg = f"Mismatch between texts ({len(docs)}) and doc_ids ({len(doc_ids)}) lengths"
            raise ValueError(msg)

        batch_size = kwargs.get("batch_size", self.batch_size)

        insert_count = 0
        try:
            for batch_start_offset in range(0, len(docs), batch_size):
                batch_end_offset = min(batch_start_offset + batch_size, len(docs))
                batch_metadata = doc_ids[batch_start_offset:batch_end_offset]
                batch_docs = docs[batch_start_offset:batch_end_offset]

                insert_data = [batch_metadata, batch_docs]
                if self.with_scalar_labels:
                    labels_data = kwargs.get("labels_data")
                    if labels_data is not None:
                        batch_labels = labels_data[batch_start_offset:batch_end_offset]
                    else:
                        batch_labels = ["" for _ in range(len(batch_docs))]
                    insert_data.append(batch_labels)
                res = self.col.insert(insert_data)
                insert_count += len(res.primary_keys)
                if batch_start_offset // batch_size % 10 == 0:
                    log.debug(
                        f"{self.name} batch insert progress: {batch_end_offset}/{len(docs)} "
                        f"({batch_end_offset/len(docs)*100:.1f}%)"
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
        if self.col is None:
            msg = "Collection not initialized. Call init() first."
            raise RuntimeError(msg)

        # Perform the search.
        res = self.col.search(
            data=[query],
            anns_field=self._vector_field,
            param=self.case_config.search_param(),
            limit=k,
            expr=self.expr,
        )

        # Organize results.
        return [result.id for result in res[0]]

    def search_fulltext(
        self,
        query: str,
        k: int = 100,
        timeout: int | None = None,
    ) -> list[int]:
        """Perform a fulltext search and return results."""
        if not self._is_fts:
            msg = "search_fulltext only valid in FTS mode"
            raise RuntimeError(msg)

        if self.col is None:
            msg = "Collection not initialized. Call init() first."
            raise RuntimeError(msg)

        query_text = str(query)

        res = self.col.search(
            data=[query_text],
            anns_field=self._sparse_field,
            param=self.case_config.search_param(),
            limit=k,
            output_fields=[self._primary_field],
        )

        hits = res[0] if res else []
        return [hit.entity.get(self._primary_field) for hit in hits]
