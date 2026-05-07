"""Wrapper around the AWS S3 Vectors service."""

import logging
from collections.abc import Iterable
from contextlib import contextmanager

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import S3VectorsIndexConfig

log = logging.getLogger(__name__)


class S3Vectors(VectorDB):
    """AWS S3 Vectors backend for VectorDBBench.

    Concurrency model:
    - thread_safe=True (inherited from VectorDB base class).
    - The ConcurrentInsertRunner and MultiProcessingSearchRunner drive
      concurrency at the worker level. All workers share the same
      self.client built in init() — boto3's low-level client is thread-safe.
    - The urllib3 connection pool size is governed by
      db_config["max_pool_connections"]; size it >= 2 * worker count.
    - Adaptive retry with botocore handles ThrottlingException; we do NOT
      add a custom retry layer because that would collide with botocore's
      adaptive token bucket.
    - PutVectors is capped at 500 vectors/call by AWS; insert_embeddings
      chunks the runner's batch into db_config["insert_batch_size"] slices.
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
        db_case_config: S3VectorsIndexConfig,
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        """Initialize wrapper around the s3-vectors client."""
        self.db_config = db_config
        self.case_config = db_case_config
        self.with_scalar_labels = with_scalar_labels

        self.insert_batch_size = self.db_config["insert_batch_size"]

        self._scalar_id_field = "id"
        self._scalar_label_field = "label"
        self._vector_field = "vector"

        self.region_name = self.db_config.get("region_name")
        self.access_key_id = self.db_config.get("access_key_id")
        self.secret_access_key = self.db_config.get("secret_access_key")
        self.bucket_name = self.db_config.get("bucket_name")
        self.index_name = self.db_config.get("index_name")
        self.endpoint_url = self.db_config.get("endpoint_url")

        self._botocore_config = Config(
            max_pool_connections=self.db_config["max_pool_connections"],
            retries={
                "mode": self.db_config["retry_mode"],
                "max_attempts": self.db_config["retry_max_attempts"],
            },
        )

        setup_client = boto3.client(
            service_name="s3vectors",
            region_name=self.region_name,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            endpoint_url=self.endpoint_url,
            config=self._botocore_config,
        )

        if drop_old:
            # delete old index if exists
            response = setup_client.list_indexes(vectorBucketName=self.bucket_name)
            index_names = [index["indexName"] for index in response["indexes"]]
            if self.index_name in index_names:
                log.info(f"drop old index: {self.index_name}")
                setup_client.delete_index(vectorBucketName=self.bucket_name, indexName=self.index_name)

            # create the index
            setup_client.create_index(
                vectorBucketName=self.bucket_name,
                indexName=self.index_name,
                dataType=self.case_config.data_type,
                dimension=dim,
                distanceMetric=self.case_config.parse_metric(),
            )

        setup_client.close()

    @contextmanager
    def init(self):
        """Yield with a long-lived boto3 client shared by all worker threads.

        boto3's low-level client is thread-safe; the connection pool size and
        retry behavior are set via self._botocore_config.
        """
        self.client = boto3.client(
            service_name="s3vectors",
            region_name=self.region_name,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            endpoint_url=self.endpoint_url,
            config=self._botocore_config,
        )

        yield
        self.client.close()

    def optimize(self, **kwargs):
        return

    def need_normalize_cosine(self) -> bool:
        """Wheather this database need to normalize dataset to support COSINE"""
        return False

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception | None]:
        """Insert embeddings into S3 Vectors via PutVectors.

        Chunks the input into self.insert_batch_size slices (<= 500, AWS hard
        limit). On error returns (count_so_far, exception); the runner decides
        whether to retry the remainder.
        """
        assert self.client is not None
        assert len(embeddings) == len(metadata)
        insert_count = 0
        try:
            for batch_start_offset in range(0, len(embeddings), self.insert_batch_size):
                batch_end_offset = min(batch_start_offset + self.insert_batch_size, len(embeddings))
                insert_data = [
                    {
                        "key": str(metadata[i]),
                        "data": {self.case_config.data_type: embeddings[i]},
                        "metadata": (
                            {self._scalar_label_field: labels_data[i], self._scalar_id_field: metadata[i]}
                            if self.with_scalar_labels
                            else {self._scalar_id_field: metadata[i]}
                        ),
                    }
                    for i in range(batch_start_offset, batch_end_offset)
                ]
                self.client.put_vectors(
                    vectorBucketName=self.bucket_name,
                    indexName=self.index_name,
                    vectors=insert_data,
                )
                insert_count += len(insert_data)
        except ClientError as e:
            log.warning(f"S3 Vectors put_vectors failed after {insert_count} inserts: {e}")
            return insert_count, e
        return insert_count, None

    def prepare_filter(self, filters: Filter):
        if filters.type == FilterOp.NonFilter:
            self.filter = None
        elif filters.type == FilterOp.NumGE:
            self.filter = {self._scalar_id_field: {"$gte": filters.int_value}}
        elif filters.type == FilterOp.StrEqual:
            self.filter = {self._scalar_label_field: filters.label_value}
        else:
            msg = f"Not support Filter for S3Vectors - {filters}"
            raise ValueError(msg)

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        timeout: int | None = None,
    ) -> list[int]:
        """Perform a search on a query embedding and return results."""
        assert self.client is not None

        # Perform the search.
        res = self.client.query_vectors(
            vectorBucketName=self.bucket_name,
            indexName=self.index_name,
            queryVector={"float32": query},
            topK=k,
            filter=self.filter,
            returnDistance=False,
            returnMetadata=False,
        )

        # Organize results.
        return [int(result["key"]) for result in res["vectors"]]
