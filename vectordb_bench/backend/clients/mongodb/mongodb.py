import logging
import time
from contextlib import contextmanager

from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

from ..api import VectorDB
from .config import MongoDBIndexConfig

log = logging.getLogger(__name__)


class MongoDBError(Exception):
    """Custom exception class for MongoDB client errors."""


class MongoDB(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: MongoDBIndexConfig,
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

        # Update index dimensions
        index_params = self.case_config.index_param()
        log.info(f"index params: {index_params}")
        index_params["fields"][0]["numDimensions"] = dim
        self.index_params = index_params

        # Initialize  - they'll also be set in init()
        uri = self.db_config["connection_string"]
        self.client = MongoClient(uri)
        self.db = self.client[self.db_config["database"]]
        self.collection = self.db[self.collection_name]
        if self.drop_old and self.collection_name in self.db.list_collection_names():
            log.info(f"MongoDB client dropping old collection: {self.collection_name}")
            self.db.drop_collection(self.collection_name)
        self.client = None
        self.db = None
        self.collection = None

    @contextmanager
    def init(self):
        """Initialize MongoDB client and cleanup when done"""
        try:
            uri = self.db_config["connection_string"]
            self.client = MongoClient(uri)
            self.db = self.client[self.db_config["database"]]
            self.collection = self.db[self.collection_name]

            yield
        finally:
            if self.client is not None:
                self.client.close()
                self.client = None
                self.db = None
                self.collection = None

    def _create_index(self) -> None:
        """Create vector search index"""
        index_name = "vector_index"
        index_params = self.index_params
        log.info(f"index params {index_params}")
        # drop index if already exists
        if self.collection.list_indexes():
            all_indexes = self.collection.list_search_indexes()
            if any(idx.get("name") == index_name for idx in all_indexes):
                log.info(f"Drop index: {index_name}")
                try:
                    self.collection.drop_search_index(index_name)
                    while True:
                        indices = list(self.collection.list_search_indexes())
                        indices = [idx for idx in indices if idx["name"] == index_name]
                        log.debug(f"index status {indices}")
                        if len(indices) == 0:
                            break
                        log.info(f"index deleting {indices}")
                except Exception:
                    log.exception(f"Error dropping index {index_name}")
        try:
            # Create vector search index
            search_index = SearchIndexModel(definition=index_params, name=index_name, type="vectorSearch")

            self.collection.create_search_index(search_index)
            log.info(f"Created vector search index: {index_name}")
            self._wait_for_index_ready(index_name)

            # Create regular index on id field for faster lookups
            self.collection.create_index(self.id_field)
            log.info(f"Created index on {self.id_field} field")

        except Exception:
            log.exception(f"Error creating index {index_name}")
            raise

    def _wait_for_index_ready(self, index_name: str, check_interval: int = 5) -> None:
        """Wait for index to be ready"""
        while True:
            indices = list(self.collection.list_search_indexes())
            log.debug(f"index status {indices}")
            if indices and any(idx.get("name") == index_name and idx.get("queryable") for idx in indices):
                break
            for idx in indices:
                if idx.get("name") == index_name and idx.get("status") == "FAILED":
                    error_msg = f"Index {index_name} failed to build"
                    raise MongoDBError(error_msg)

            time.sleep(check_interval)
        log.info(f"Index {index_name} is ready")

    def need_normalize_cosine(self) -> bool:
        return False

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> (int, Exception | None):
        """Insert embeddings into MongoDB"""

        # Prepare documents in bulk
        documents = [
            {
                self.id_field: id_,
                self.vector_field: embedding,
            }
            for id_, embedding in zip(metadata, embeddings, strict=False)
        ]

        # Use ordered=False for better insert performance
        try:
            self.collection.insert_many(documents, ordered=False)
        except Exception as e:
            return 0, e
        return len(documents), None

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        **kwargs,
    ) -> list[int]:
        """Search for similar vectors"""
        search_params = self.case_config.search_param()

        vector_search = {"queryVector": query, "index": "vector_index", "path": self.vector_field, "limit": k}

        # Add exact search parameter if specified
        if search_params["exact"]:
            vector_search["exact"] = True
        else:
            # Set numCandidates based on k value and data size
            # For 50K dataset, use higher multiplier for better recall
            num_candidates = min(10000, k * search_params["num_candidates_ratio"])
            vector_search["numCandidates"] = num_candidates

        # Add filter if specified
        if filters:
            log.info(f"Applying filter: {filters}")
            vector_search["filter"] = {
                "id": {"gte": filters["id"]},
            }
        pipeline = [
            {"$vectorSearch": vector_search},
            {
                "$project": {
                    "_id": 0,
                    self.id_field: 1,
                    "score": {"$meta": "vectorSearchScore"},  # Include similarity score
                }
            },
        ]

        results = list(self.collection.aggregate(pipeline))
        return [doc[self.id_field] for doc in results]

    def optimize(self, data_size: int | None = None) -> None:
        """MongoDB vector search indexes are self-optimizing"""
        log.info("optimize for search")
        self._create_index()
        self._wait_for_index_ready("vector_index")

    def ready_to_load(self) -> None:
        """MongoDB is always ready to load"""
