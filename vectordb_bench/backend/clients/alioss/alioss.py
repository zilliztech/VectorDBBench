"""Wrapper around the Aliyun OSS Vector service."""

import logging
from contextlib import contextmanager

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import AliOSSIndexConfig

log = logging.getLogger(__name__)


class AliOSS(VectorDB):
    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: AliOSSIndexConfig,
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        import alibabacloud_oss_v2 as oss
        import alibabacloud_oss_v2.vectors as oss_vectors

        self.db_config = db_config
        self.case_config = db_case_config
        self.with_scalar_labels = with_scalar_labels
        self.insert_batch_size = self.db_config["insert_batch_size"]
        self.filter = None

        self._scalar_id_field = "id"
        self._scalar_label_field = "label"

        self.access_key_id = self.db_config["access_key_id"]
        self.access_key_secret = self.db_config["access_key_secret"]
        self.region = self.db_config["region"]
        self.account_id = self.db_config["account_id"]
        self.bucket_name = self.db_config["bucket_name"]
        self.index_name = self.db_config["index_name"]

        cfg = oss.config.load_default()
        cfg.credentials_provider = oss.credentials.StaticCredentialsProvider(
            access_key_id=self.access_key_id,
            access_key_secret=self.access_key_secret,
        )
        cfg.region = self.region
        cfg.account_id = self.account_id
        self._oss_cfg = cfg

        if drop_old:
            setup_client = oss_vectors.Client(cfg)
            try:
                setup_client.delete_vector_index(
                    oss_vectors.models.DeleteVectorIndexRequest(
                        bucket=self.bucket_name,
                        index_name=self.index_name,
                    )
                )
                log.info(f"dropped old index: {self.index_name}")
            except Exception as e:
                err = str(e)
                if "NoSuchVectorIndex" not in err and "not exist" not in err.lower():
                    raise

            setup_client.put_vector_index(
                oss_vectors.models.PutVectorIndexRequest(
                    bucket=self.bucket_name,
                    index_name=self.index_name,
                    dimension=dim,
                    data_type=self.case_config.data_type,
                    distance_metric=self.case_config.parse_metric(),
                )
            )

    @contextmanager
    def init(self):
        import alibabacloud_oss_v2.vectors as oss_vectors

        self.client = oss_vectors.Client(self._oss_cfg)
        yield

    def optimize(self, **kwargs):
        return

    def need_normalize_cosine(self) -> bool:
        return False

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception | None]:
        import alibabacloud_oss_v2.vectors as oss_vectors

        assert self.client is not None
        assert len(embeddings) == len(metadata)
        insert_count = 0
        try:
            for batch_start in range(0, len(embeddings), self.insert_batch_size):
                batch_end = min(batch_start + self.insert_batch_size, len(embeddings))
                vectors = [
                    {
                        "key": str(metadata[i]),
                        "data": {self.case_config.data_type: embeddings[i]},
                        "metadata": (
                            {self._scalar_label_field: labels_data[i], self._scalar_id_field: metadata[i]}
                            if self.with_scalar_labels
                            else {self._scalar_id_field: metadata[i]}
                        ),
                    }
                    for i in range(batch_start, batch_end)
                ]
                self.client.put_vectors(
                    oss_vectors.models.PutVectorsRequest(
                        bucket=self.bucket_name,
                        index_name=self.index_name,
                        vectors=vectors,
                    )
                )
                insert_count += len(vectors)
        except Exception as e:
            log.warning(f"AliOSS put_vectors failed after {insert_count} inserts: {e}")
            return insert_count, e
        return insert_count, None

    def prepare_filter(self, filters: Filter):
        if filters.type == FilterOp.NonFilter:
            self.filter = None
        elif filters.type == FilterOp.NumGE:
            self.filter = {self._scalar_id_field: {"$gte": filters.int_value}}
        elif filters.type == FilterOp.StrEqual:
            self.filter = {self._scalar_label_field: {"$eq": filters.label_value}}
        else:
            raise ValueError(f"Not support Filter for AliOSS - {filters}")

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        timeout: int | None = None,
    ) -> list[int]:
        import alibabacloud_oss_v2.vectors as oss_vectors

        assert self.client is not None

        resp = self.client.query_vectors(
            oss_vectors.models.QueryVectorsRequest(
                bucket=self.bucket_name,
                index_name=self.index_name,
                query_vector={self.case_config.data_type: query},
                top_k=k,
                filter=self.filter,
                return_distance=False,
                return_metadata=False,
            )
        )
        return [int(v.key) for v in (resp.vectors or [])]
