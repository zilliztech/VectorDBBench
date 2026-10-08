from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path

import zvec
from zvec import (
    CollectionOption,
    CollectionSchema,
    DataType,
    Doc,
    FieldSchema,
    InvertIndexParam,
    LogLevel,
    OptimizeOption,
    QuantizeType,
    VectorQuery,
    VectorSchema,
)

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import MetricType, NonRetryableInsertError, PartialInsertError, VectorDB
from .config import ZvecConfig, ZvecDiskANNIndexConfig, ZvecHNSWIndexConfig, ZvecIndexConfig

log = logging.getLogger(__name__)

zvec.init(log_level=LogLevel.WARN)


class Zvec(VectorDB):
    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    def __init__(
        self,
        dim: int,
        db_config: ZvecConfig,
        db_case_config: ZvecIndexConfig,
        collection_name: str = "vector_bench_test",
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        self.name = "Zvec"
        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = collection_name
        self.dim = dim
        self.path = db_config["path"]
        self.expr = ""
        # avoid the search_param being called every time during the search process
        self.search_config = db_case_config.search_param()
        self._scalar_id_field = "id"
        self._scalar_label_field = "label"
        self.with_scalar_labels = with_scalar_labels

        log.info(f"Search config: {self.search_config}")

        fields = [
            FieldSchema(
                "id", DataType.INT64, nullable=False, index_param=InvertIndexParam(enable_range_optimization=True)
            ),
        ]
        if with_scalar_labels:
            fields.append(
                FieldSchema(
                    self._scalar_label_field,
                    DataType.STRING,
                    nullable=False,
                    index_param=InvertIndexParam(enable_range_optimization=False),
                )
            )
        self.schema = CollectionSchema(
            name=self.table_name,
            fields=fields,
            vectors=[
                VectorSchema(
                    "dense",
                    DataType.VECTOR_FP32,
                    dimension=dim,
                    index_param=Zvec._parse_index_param(db_case_config),
                ),
            ],
        )

        self.option = CollectionOption(read_only=False, enable_mmap=True)

        self.query_param = Zvec._parse_query_param(db_case_config)

        if isinstance(db_case_config, ZvecDiskANNIndexConfig):
            if dim <= 0 or db_case_config.pq_chunk_num > dim:
                message = "DiskANN requires a positive dimension and pq_chunk_num <= dimension"
                raise ValueError(message)
            self._prepare_diskann_collection(drop_old)
            return

        if drop_old:
            try:
                collection = zvec.open(self.path)
                collection.destroy()
            except Exception as e:
                log.warning(f"Failed to drop table {self.table_name}: {e}")

            collection = zvec.create_and_open(path=self.path, schema=self.schema, option=self.option)
        else:
            collection = zvec.open(self.path)

    def _prepare_diskann_collection(self, drop_old: bool) -> None:
        if Path(self.path).exists():
            option = CollectionOption(read_only=not drop_old, enable_mmap=True)
            collection = zvec.open(self.path, option=option)
            destroyed = False
            try:
                self._validate_diskann_collection(collection, require_ready=not drop_old, rebuilding=drop_old)
                if drop_old:
                    collection.destroy()
                    destroyed = True
            finally:
                if not destroyed:
                    collection.close()
        elif not drop_old:
            message = f"DiskANN collection does not exist at {self.path}; build it before search-only benchmarking"
            raise ValueError(message)
        if drop_old:
            collection = zvec.create_and_open(path=self.path, schema=self.schema, option=self.option)
            collection.close()
        log.info(
            "DiskANN selected: path=%s metric=%s build=%s search=%s stored_vectors=FP32",
            self.path,
            self._parse_metric(self.case_config.metric_type),
            self.case_config.index_param(),
            self.search_config,
        )

    def _validate_diskann_collection(
        self,
        collection: zvec.Collection,
        *,
        require_ready: bool,
        rebuilding: bool = False,
    ) -> None:
        field = collection.schema.vector("dense")
        if field is None or field.index_param.type != zvec.IndexType.DISKANN:
            message = "Refusing to use or delete a non-DiskANN collection; choose a separate DiskANN path"
            raise ValueError(message)
        if rebuilding:
            return
        expected = self._parse_index_param(self.case_config)
        actual = field.index_param
        if field.dimension != self.dim or any(
            getattr(actual, name) != getattr(expected, name)
            for name in ("metric_type", "quantize_type", "max_degree", "list_size", "pq_chunk_num")
        ):
            message = "Stored DiskANN schema differs from requested dimensions, metric or build parameters"
            raise ValueError(message)
        if require_ready:
            stats = collection.stats
            if stats.doc_count <= 0 or stats.index_completeness.get("dense") != 1.0:
                message = "DiskANN collection is empty or not fully indexed; complete load and optimize first"
                raise RuntimeError(message)

    @contextmanager
    def init(self):
        self.collection = zvec.open(self.path, self.option)
        try:
            if isinstance(self.case_config, ZvecDiskANNIndexConfig):
                self._validate_diskann_collection(self.collection, require_ready=self.option.read_only)
            yield
        finally:
            try:
                if isinstance(self.case_config, ZvecDiskANNIndexConfig):
                    self.collection.close()
            finally:
                self.collection = None

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception | None]:
        if isinstance(self.case_config, ZvecDiskANNIndexConfig) and (
            len(embeddings) != len(metadata)
            or (self.with_scalar_labels and (labels_data is None or len(labels_data) != len(metadata)))
        ):
            return 0, NonRetryableInsertError("Embedding, ID and label batch lengths must match")
        docs = []
        for i, id_ in enumerate(metadata):
            embedding = embeddings[i]
            fields = (
                {"id": id_} if not self.with_scalar_labels else {"id": id_, self._scalar_label_field: labels_data[i]}
            )
            docs.append(
                Doc(
                    id=f"{id_}",
                    fields=fields,
                    vectors={
                        "dense": embedding,
                    },
                )
            )
        try:
            statuses = self.collection.insert(docs)
            if isinstance(self.case_config, ZvecDiskANNIndexConfig):
                if len(statuses) != len(docs):
                    return 0, NonRetryableInsertError("zvec insert returned an incomplete status list")
                inserted = sum(status.ok() for status in statuses)
                if inserted != len(docs):
                    message = f"zvec DiskANN batch insert failed for {len(docs) - inserted}/{len(docs)} documents"
                    return inserted, PartialInsertError(message, inserted_count=inserted)
            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert data into Zvec table ({self.table_name}), error: {e}")
            return 0, e

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
    ) -> list[int]:
        if filters:
            results = []
        else:
            results = self.collection.query(
                output_fields=[],
                topk=k,
                filter=self.expr,
                vectors=VectorQuery(field_name="dense", vector=query, param=self.query_param),
            )

        return [int(result.id) for result in results]

    def optimize(self, data_size: int | None = None):
        if isinstance(self.case_config, ZvecDiskANNIndexConfig):
            self._validate_diskann_collection(self.collection, require_ready=False)
            if data_size is not None and self.collection.stats.doc_count != data_size:
                message = f"DiskANN document count {self.collection.stats.doc_count} differs from expected {data_size}"
                raise RuntimeError(message)
            log.info("Building DiskANN index via optimize: documents=%s", self.collection.stats.doc_count)
        self.collection.optimize(option=OptimizeOption())
        if isinstance(self.case_config, ZvecDiskANNIndexConfig):
            self._validate_diskann_collection(self.collection, require_ready=True)
            stats = self.collection.stats
            if data_size is not None and stats.doc_count != data_size:
                message = f"DiskANN document count after optimize is {stats.doc_count}, expected {data_size}"
                raise RuntimeError(message)
            log.info(
                "DiskANN optimize completed: documents=%s index_completeness=%s",
                stats.doc_count,
                stats.index_completeness["dense"],
            )

    def prepare_filter(self, filters: Filter):
        self.option = CollectionOption(read_only=True, enable_mmap=True)
        log.debug("set readonly: %s", self.option.read_only)

        if filters.type == FilterOp.NonFilter:
            self.expr = ""
        elif filters.type == FilterOp.NumGE:
            self.expr = f"{self._scalar_id_field} >= {filters.int_value}"
        elif filters.type == FilterOp.StrEqual:
            self.expr = f"{self._scalar_label_field} = '{filters.label_value}'"
        else:
            msg = f"Not support Filter for zvec - {filters}"
            raise ValueError(msg)

    @classmethod
    def _parse_metric(cls, metric_type: MetricType) -> zvec.MetricType:
        if not metric_type:
            return zvec.MetricType.IP
        d = {
            MetricType.COSINE: zvec.MetricType.COSINE,
            MetricType.L2: zvec.MetricType.L2,
            MetricType.IP: zvec.MetricType.IP,
        }
        return d[metric_type]

    @classmethod
    def _parse_index_param(cls, index_config: ZvecIndexConfig) -> zvec.HnswIndexParam | zvec.DiskAnnIndexParam:
        if isinstance(index_config, ZvecDiskANNIndexConfig):
            if not hasattr(zvec, "DiskAnnIndexParam"):
                message = "Installed zvec SDK does not expose DiskAnnIndexParam; install a DiskANN-enabled zvec build"
                raise ValueError(message)
            return zvec.DiskAnnIndexParam(
                metric_type=Zvec._parse_metric(index_config.metric_type),
                max_degree=index_config.max_degree,
                list_size=index_config.build_list_size,
                pq_chunk_num=index_config.pq_chunk_num,
                quantize_type=QuantizeType.UNDEFINED,
            )
        if isinstance(index_config, ZvecHNSWIndexConfig):
            return zvec.HnswIndexParam(
                metric_type=Zvec._parse_metric(index_config.metric_type),
                m=index_config.M,
                ef_construction=index_config.ef_construction,
                quantize_type=Zvec._parse_quantize_type(index_config.quantize_type),
            )
        message = f"Not support index type - {index_config}"
        raise ValueError(message)

    @classmethod
    def _parse_query_param(cls, index_config: ZvecIndexConfig) -> zvec.HnswQueryParam | zvec.DiskAnnQueryParam:
        if isinstance(index_config, ZvecDiskANNIndexConfig):
            if not hasattr(zvec, "DiskAnnQueryParam"):
                message = "Installed zvec SDK does not expose DiskAnnQueryParam; install a DiskANN-enabled zvec build"
                raise ValueError(message)
            return zvec.DiskAnnQueryParam(list_size=index_config.search_list_size)
        if isinstance(index_config, ZvecHNSWIndexConfig):
            return zvec.HnswQueryParam(
                ef=index_config.ef_search,
                is_using_refiner=index_config.is_using_refiner,
            )
        message = f"Not support index type - {index_config}"
        raise ValueError(message)

    @classmethod
    def _parse_quantize_type(cls, quantize_type: str) -> QuantizeType:
        if not quantize_type:
            return QuantizeType.UNDEFINED
        d = {
            "FP16": QuantizeType.FP16,
            "INT8": QuantizeType.INT8,
            "INT4": QuantizeType.INT4,
        }
        return d[quantize_type.upper()]
