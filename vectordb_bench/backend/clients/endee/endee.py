import logging
from contextlib import contextmanager
from collections.abc import Iterable

from endee import endee

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import DBCaseConfig, EmptyDBCaseConfig, IndexType, VectorDB
from .config import EndeeConfig

log = logging.getLogger(__name__)


class Endee(VectorDB):
    """
    VectorDBBench client implementation for Endee VectorDB.
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
        **kwargs,
    ):
        self.token = db_config.get("token", "")
        self.region = db_config.get("region", "as1")
        self.base_url = db_config.get("base_url")

        self.collection_name = db_config.get("collection_name") or db_config.get("index_name")
        if not self.collection_name:
            self.collection_name = f"endee_bench_{uuid.uuid4().hex[:8]}"

        self.space_type = db_config.get("space_type", "cosine")
        self.precision = db_config.get("precision")
        self.version = db_config.get("version")
        self.M = db_config.get("m")
        self.ef_con = db_config.get("ef_con")
        self.ef_search = db_config.get("ef_search")
        self.with_scalar_labels = with_scalar_labels

        self.filter_expr = None
        self._scalar_id_field = "id"
        self._scalar_label_field = "label"

        self.nd = endee.Endee(token=self.token)

        if self.base_url:
            self.nd.set_base_url(self.base_url)
            log.info(f"Targeting server: {self.base_url}")

        try:
            index_name = self.collection_name
            indices = self.nd.list_indexes().get("indices", [])
            # Check if index exists by name
            _ = [index["name"] for index in indices] if indices else []

            try:
                self.index = self.nd.get_index(name=index_name)
                log.info(f"Connected to existing Endee index: '{index_name}'")

            except Exception:
                log.warning(f"Index '{index_name}' not found. Creating new index...")
                try:
                    self._create_index(dim)
                    self.index = self.nd.get_index(name=index_name)
                    log.info(f"Successfully created and connected to index: '{index_name}'")

                except Exception as create_error:
                    if "already exists" in str(create_error).lower() or "conflict" in str(create_error).lower():
                        log.warning(f"Index '{index_name}' already exists despite previous error. Fetching it again.")
                        self.index = self.nd.get_index(name=index_name)
                    else:
                        log.error(f"Failed to create Endee index: {create_error}")
                        raise
        except Exception as e:
            log.error(f"Error accessing or creating Endee index '{self.collection_name}': {e}")
            raise

    def _create_index(self, dim: int):
        try:
            resp = self.nd.create_index(
                name=self.collection_name,
                dimension=dim,
                space_type=self.space_type,
                precision=self.precision,
                version=self.version,
                M=self.M,
                ef_con=self.ef_con,
            )
            log.info(f"Created new Endee index: {resp}")
        except Exception as e:
            log.error(f"Failed to create Endee index: {e}")
            raise

    def _drop_index(self, collection_name: str):
        try:
            res = self.nd.delete_index(collection_name)
            log.info(res)
        except Exception as e:
            log.error(f"Failed to drop Endee index: {e}")
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
            self.index = nd.get_index(name=self.collection_name)
            yield
        except Exception as e:
            if hasattr(e, "response") and e.response is not None:
                log.error(f"HTTP Status: {e.response.status_code}, Body: {e.response.text}")
            log.error(f"Error initializing Endee client: {e}")
            raise
        finally:
            pass

    def optimize(self, data_size: int | None = None):
        """
        Optimization step after insertion.
        """
        pass

    def prepare_filter(self, filters: Filter):
        """
        Translate VectorDBBench standard filters to Endee specific filter expressions.
        """
        if filters.type == FilterOp.NonFilter:
            self.filter_expr = None

        elif filters.type == FilterOp.NumGE:
            # Endee supports $range, assuming upper bound of 1M for benchmark dataset
            self.filter_expr = [{self._scalar_id_field: {"$range": [filters.int_value, 1_000_000]}}]

        elif filters.type == FilterOp.StrEqual:
            self.filter_expr = [{self._scalar_label_field: {"$eq": filters.label_value}}]

        else:
            raise ValueError(f"Not support Filter for Endee - {filters}")

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
            batch_vectors = []
            for i in range(len(embeddings)):
                vector_data = {
                    "id": str(metadata[i]),
                    "vector": embeddings[i],
                    "meta": {"id": metadata[i]},
                    "filter": {self._scalar_id_field: metadata[i]},
                }

                if self.with_scalar_labels and labels_data is not None:
                    vector_data["filter"][self._scalar_label_field] = labels_data[i]

                batch_vectors.append(vector_data)

            self.index.upsert(batch_vectors)
            insert_count = len(batch_vectors)

        except Exception as e:
            log.error(f"Failed to insert data: {e}")
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
            results = self.index.query(
                vector=query, top_k=k, filter=self.filter_expr, ef=self.ef_search, include_vectors=False
            )

            return [int(result["id"]) for result in results]

        except Exception as e:
            log.warning(f"Error querying Endee index: {e}")
            raise

    def describe_index(self) -> dict:
        """
        Get information about the current index.
        """
        try:
            all_indices = self.nd.list_indexes().get("indices", [])
            for idx in all_indices:
                if idx.get("name") == self.collection_name:
                    return idx
            return {}
        except Exception as e:
            log.warning(f"Error describing Endee index: {e}")
            return {}
