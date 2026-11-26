import logging
import uuid
from collections.abc import Iterable
from contextlib import contextmanager

from endee import endee

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import DBCaseConfig, EmptyDBCaseConfig, VectorDB
from .config import EndeeConfig

log = logging.getLogger(__name__)

_VECTOR_FIELD_NAME = "dense"
# Max objects per upsert request accepted by the endee SDK (MAX_VECTORS_PER_BATCH).
_MAX_UPSERT_BATCH = 10_000


class Endee(VectorDB):
    """
    VectorDBBench client implementation for Endee VectorDB (collections-based API).
    """

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
        collection_name: str | None = None,
        **kwargs,
    ):
        self.token = db_config.get("token", "")
        self.base_url = db_config.get("base_url")

        self.collection_name = (
            collection_name or db_config.get("collection_name") or f"endee_bench_{uuid.uuid4().hex[:8]}"
        )

        self.space_type = db_config.get("space_type", "cosine")
        self.precision = db_config.get("precision")
        self.M = db_config.get("m")
        self.ef_con = db_config.get("ef_con")
        self.ef_search = db_config.get("ef_search")
        self.prefilter_cardinality_threshold = db_config.get("prefilter_cardinality_threshold")
        self.filter_boost_percentage = db_config.get("filter_boost_percentage")
        self.with_scalar_labels = with_scalar_labels

        self.filter_expr = None
        self._scalar_id_field = "id"
        self._scalar_label_field = "label"

        self.nd = endee.Endee(token=self.token)

        if self.base_url:
            self.nd.set_base_url(self.base_url)
            log.info(f"Targeting server: {self.base_url}")

        try:
            collections = self.nd.list_collections()
            existing_names = [c["name"] for c in collections] if collections else []

            if drop_old and self.collection_name in existing_names:
                log.info(f"Dropping existing Endee collection: '{self.collection_name}'")
                self._drop_collection(self.collection_name)

            try:
                self.collection = self.nd.get_collection(name=self.collection_name)
                log.info(f"Connected to existing Endee collection: '{self.collection_name}'")

            except Exception:
                log.warning(f"Collection '{self.collection_name}' not found. Creating new collection...")
                try:
                    self._create_collection(dim)
                    self.collection = self.nd.get_collection(name=self.collection_name)
                    log.info(f"Successfully created and connected to collection: '{self.collection_name}'")

                except Exception as create_error:
                    if "already exists" in str(create_error).lower() or "conflict" in str(create_error).lower():
                        log.warning(f"Collection '{self.collection_name}' already exists. Fetching it again.")
                        self.collection = self.nd.get_collection(name=self.collection_name)
                    else:
                        log.exception("Failed to create Endee collection")
                        raise
        except Exception:
            log.exception(f"Error accessing or creating Endee collection '{self.collection_name}'")
            raise

    def _create_collection(self, dim: int):
        try:
            params: dict = {
                "dimension": dim,
                "space_type": self.space_type,
            }
            if self.precision is not None:
                params["precision"] = self.precision
            if self.M is not None:
                params["M"] = self.M
            if self.ef_con is not None:
                params["ef_con"] = self.ef_con

            resp = self.nd.create_collection(
                name=self.collection_name,
                fields=[
                    {
                        "name": _VECTOR_FIELD_NAME,
                        "type": "vector",
                        "params": params,
                    }
                ],
            )
            log.info(f"Created new Endee collection: {resp}")
        except Exception:
            log.exception("Failed to create Endee collection")
            raise

    def _drop_collection(self, collection_name: str):
        try:
            res = self.nd.delete_collection(collection_name)
            log.info(res)
        except Exception:
            log.exception("Failed to drop Endee collection")
            raise

    @classmethod
    def config_cls(cls) -> type[EndeeConfig]:
        return EndeeConfig

    @classmethod
    def case_config_cls(cls, index_type: str | None = None) -> type[DBCaseConfig]:
        return EmptyDBCaseConfig

    @contextmanager
    def init(self):
        """
        Context manager for initializing the client connection.
        """
        try:
            nd = endee.Endee(token=self.token)
            if self.base_url:
                nd.set_base_url(self.base_url)
            self.nd = nd
            self.collection = nd.get_collection(name=self.collection_name)
            yield
        except Exception as e:
            msg = "Error initializing Endee client"
            if hasattr(e, "response") and e.response is not None:
                msg += f" (HTTP Status: {e.response.status_code}, Body: {e.response.text})"
            log.exception(msg)
            raise

    def optimize(self, data_size: int | None = None):
        """
        Optimization step after insertion.
        """

    def prepare_filter(self, filters: Filter):
        """
        Translate VectorDBBench standard filters to Endee specific filter expressions.
        """
        if filters.type == FilterOp.NonFilter:
            self.filter_expr = None

        elif filters.type == FilterOp.NumGE:
            self.filter_expr = [{self._scalar_id_field: {"$gte": filters.int_value}}]

        elif filters.type == FilterOp.StrEqual:
            self.filter_expr = [{self._scalar_label_field: {"$eq": filters.label_value}}]

        else:
            msg = f"Not support Filter for Endee - {filters}"
            raise ValueError(msg)

    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception | None]:
        """
        Insert embeddings with associated metadata and labels.
        """
        assert len(embeddings) == len(metadata)
        insert_count = 0
        try:
            objects = []
            for i in range(len(embeddings)):
                obj = {
                    "id": str(metadata[i]),
                    "meta": {"id": metadata[i]},
                    "filter": {self._scalar_id_field: metadata[i]},
                    "fields": {_VECTOR_FIELD_NAME: embeddings[i]},
                }

                if self.with_scalar_labels and labels_data is not None:
                    obj["filter"][self._scalar_label_field] = labels_data[i]

                objects.append(obj)

            for start in range(0, len(objects), _MAX_UPSERT_BATCH):
                chunk = objects[start : start + _MAX_UPSERT_BATCH]
                self.collection.upsert(chunk)
                insert_count += len(chunk)

        except Exception as e:
            log.exception("Failed to insert data")
            return insert_count, e

        return (len(embeddings), None)

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        timeout: int | None = None,
        **kwargs,
    ) -> list[int]:
        """
        Perform vector search with optional filters.
        """
        try:
            search_kwargs = {
                "fields": {_VECTOR_FIELD_NAME: {"query": query, "limit": k}},
                "filter": self.filter_expr,
            }
            if self.ef_search is not None:
                search_kwargs["ef_search"] = self.ef_search
            if self.filter_expr is not None:
                if self.prefilter_cardinality_threshold is not None:
                    search_kwargs["prefilter_cardinality_threshold"] = self.prefilter_cardinality_threshold
                if self.filter_boost_percentage is not None:
                    search_kwargs["filter_boost_percentage"] = self.filter_boost_percentage
            response = self.collection.search(**search_kwargs)

            results = response.get("results", {})
            hits = results.get(_VECTOR_FIELD_NAME, []) if isinstance(results, dict) else results
            return [int(hit["id"]) for hit in hits]

        except Exception as e:
            log.warning(f"Error querying Endee collection: {e}")
            raise

    def describe_index(self) -> dict:
        """
        Get information about the current collection.
        """
        try:
            collections = self.nd.list_collections()
        except Exception as e:
            log.warning(f"Error describing Endee collection: {e}")
            return {}

        for col in collections:
            if col.get("name") == self.collection_name:
                return col
        return {}
