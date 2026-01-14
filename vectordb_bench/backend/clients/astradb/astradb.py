import logging
import time
from contextlib import contextmanager

from astrapy import DataAPIClient
from astrapy.constants import VectorMetric
from astrapy.info import CollectionDefinition

from ..api import VectorDB
from .config import AstraDBIndexConfig

log = logging.getLogger(__name__)


class AstraDBError(Exception):
    """Custom exception class for AstraDB client errors."""


class AstraDB(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: AstraDBIndexConfig,
        collection_name: str = "vdb_bench_collection",
        id_field: str = "id",
        vector_field: str = "vector",
        drop_old: bool = False,
        **kwargs,
    ):
        self.dim = dim
        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = collection_name
        self.id_field = id_field
        self.vector_field = vector_field
        self.drop_old = drop_old

        # Get index parameters
        index_params = self.case_config.index_param()
        log.info(f"index params: {index_params}")
        self.index_params = index_params

        # Initialize client - will be properly set in init()
        self.client = None
        self.db = None
        self.collection = None

        # Initialize and drop collection if needed
        temp_client = DataAPIClient(self.db_config["token"])
        temp_db = temp_client.get_database(
            api_endpoint=self.db_config["api_endpoint"],
            keyspace=self.db_config["namespace"]
        )

        if self.drop_old:
            try:
                temp_db.drop_collection(self.collection_name)
                log.info(f"AstraDB client dropped old collection: {self.collection_name}")
            except Exception:
                log.info(f"Collection {self.collection_name} does not exist, skipping drop")

    @contextmanager
    def init(self):
        """Initialize AstraDB client and cleanup when done"""
        try:
            self.client = DataAPIClient(self.db_config["token"])
            self.db = self.client.get_database(
                api_endpoint=self.db_config["api_endpoint"],
                keyspace=self.db_config["namespace"]
            )

            # Create or get collection with vector configuration
            metric_str = self.case_config.parse_metric()

            # Map metric string to VectorMetric constant
            metric_map = {
                "euclidean": VectorMetric.EUCLIDEAN,
                "dot_product": VectorMetric.DOT_PRODUCT,
                "cosine": VectorMetric.COSINE,
            }
            metric = metric_map.get(metric_str, VectorMetric.COSINE)

            # Create collection with new API
            # Note: check_exists is no longer needed - API handles conflicts automatically
            self.collection = self.db.create_collection(
                name=self.collection_name,
                definition=(
                    CollectionDefinition.builder()
                    .set_vector_dimension(self.dim)
                    .set_vector_metric(metric)
                    .build()
                ),
            )
            log.info(f"Created/accessed collection: {self.collection_name} with metric: {metric_str}")

            yield
        finally:
            if self.client is not None:
                self.client = None
                self.db = None
                self.collection = None

    def need_normalize_cosine(self) -> bool:
        return False

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> (int, Exception | None):
        """Insert embeddings into AstraDB"""

        # Prepare documents in bulk
        documents = [
            {
                "_id": str(id_),
                "$vector": embedding,
            }
            for id_, embedding in zip(metadata, embeddings, strict=False)
        ]

        # Insert documents in batches
        try:
            result = self.collection.insert_many(documents, ordered=False)
            return len(result.inserted_ids), None
        except Exception as e:
            log.exception("Error inserting embeddings")
            return 0, e

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        **kwargs,
    ) -> list[int]:
        """Search for similar vectors"""

        # Build filter if specified
        search_filter = None
        if filters:
            log.info(f"Applying filter: {filters}")
            search_filter = {
                self.id_field: {"$gte": filters["id"]},
            }

        # Perform vector search
        try:
            results = self.collection.find(
                filter=search_filter,
                sort={"$vector": query},
                limit=k,
                include_similarity=True,
            )

            # Extract IDs from results
            return [int(doc["_id"]) for doc in results]
        except Exception:
            log.exception("Error searching embeddings")
            return []

    def optimize(self, data_size: int | None = None) -> None:
        """AstraDB vector indexes are automatically managed"""
        log.info("optimize for search - AstraDB manages indexes automatically")

    def ready_to_load(self) -> None:
        """AstraDB is always ready to load"""
