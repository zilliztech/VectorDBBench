"""Wrapper around the Milvus vector database over VectorDB"""

import logging
from collections.abc import Iterable
from contextlib import contextmanager

import boto3

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import S3VectorsIndexConfig

log = logging.getLogger(__name__)


class S3Vectors(VectorDB):
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

        self.batch_size = 500

        self._scalar_id_field = "id"
        self._scalar_label_field = "label"
        self._vector_field = "vector"

        self.region_name = self.db_config.get("region_name")
        self.access_key_id = self.db_config.get("access_key_id")
        self.secret_access_key = self.db_config.get("secret_access_key")
        self.bucket_name = self.db_config.get("bucket_name")
        self.index_name = self.db_config.get("index_name")

        client = boto3.client(
            service_name="s3vectors",
            region_name=self.region_name,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
        )

        if drop_old:
            # delete old index if exists
            response = client.list_indexes(vectorBucketName=self.bucket_name)
            index_names = [index["indexName"] for index in response["indexes"]]
            if self.index_name in index_names:
                log.info(f"drop old index: {self.index_name}")
                client.delete_index(vectorBucketName=self.bucket_name, indexName=self.index_name)

            # create the index
            client.create_index(
                vectorBucketName=self.bucket_name,
                indexName=self.index_name,
                dataType=self.case_config.data_type,
                dimension=dim,
                distanceMetric=self.case_config.parse_metric(),
            )

        client.close()

    @contextmanager
    def init(self):
        """
        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding()
        """
        self.client = boto3.client(
            service_name="s3vectors",
            region_name=self.region_name,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
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
        embeddings: Iterable[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception]:
        """Insert embeddings into s3-vectors. should call self.init() first"""
        # use the first insert_embeddings to init collection
        assert self.client is not None
        assert len(embeddings) == len(metadata)
        insert_count = 0
        try:
            for batch_start_offset in range(0, len(embeddings), self.batch_size):
                batch_end_offset = min(batch_start_offset + self.batch_size, len(embeddings))
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
        except Exception as e:
            log.info(f"Failed to insert data: {e}")
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
