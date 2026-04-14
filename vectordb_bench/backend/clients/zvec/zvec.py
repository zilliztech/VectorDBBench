import logging
from contextlib import contextmanager

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

from ..api import MetricType, VectorDB
from .config import ZvecConfig, ZvecHNSWIndexConfig, ZvecIndexConfig

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

        if drop_old:
            try:
                collection = zvec.open(self.path)
                collection.destroy()
            except Exception as e:
                log.warning(f"Failed to drop table {self.table_name}: {e}")

            collection = zvec.create_and_open(path=self.path, schema=self.schema, option=self.option)
        else:
            collection = zvec.open(self.path)

    @contextmanager
    def init(self):
        self.collection = zvec.open(self.path, self.option)
        yield
        self.collection = None

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception]:
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
            self.collection.insert(docs)
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
        self.collection.optimize(option=OptimizeOption())

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
    def _parse_index_param(cls, index_config: ZvecIndexConfig) -> zvec.HnswIndexParam:
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
    def _parse_query_param(cls, index_config: ZvecIndexConfig) -> zvec.HnswQueryParam:
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
