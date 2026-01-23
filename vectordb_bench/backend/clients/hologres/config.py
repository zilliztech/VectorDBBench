from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class HologresConfig(DBConfig):
    user_name: SecretStr = SecretStr("hologres")
    password: SecretStr
    host: str = "localhost"
    port: int = 5432
    db_name: str

    def to_dict(self) -> dict:
        user_str = self.user_name.get_secret_value()
        pwd_str = self.password.get_secret_value()
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.db_name,
            "user": user_str,
            "password": pwd_str,
        }


class HologresIndexConfig(BaseModel, DBCaseConfig):
    index: IndexType = IndexType.Hologres_HGraph
    metric_type: MetricType | None = None

    create_index_before_load: bool = False
    create_index_after_load: bool = True

    min_flush_proxima_row_count: int = 1000
    min_compaction_proxima_row_count: int = 1000
    max_total_size_to_merge_mb: int = 4096
    full_compact_max_file_size_mb: int = 4096

    base_quantization_type: str = "sq8_uniform"
    precise_quantization_type: str = "fp32"
    use_reorder: bool = True
    build_thread_count: int = 16
    max_degree: int = 64
    ef_construction: int = 400

    ef_search: int = 51

    def index_param(self) -> dict:
        return {
            "algorithm": self.algorithm(),
            "distance_method": self.distance_method(),
            "builder_params": self.builder_params(),
            "full_compact_max_file_size_mb": self.full_compact_max_file_size_mb,
        }

    def search_param(self) -> dict:
        return {
            "distance_function": self.distance_function(),
            "order_direction": self.order_direction(),
            "searcher_params": self.search_params(),
        }

    def algorithm(self) -> str:
        return self.index.value

    def is_proxima(self) -> bool:
        return self.index == IndexType.Hologres_Graph

    def distance_method(self) -> str:
        if self.metric_type == MetricType.L2:
            if self.index == IndexType.Hologres_Graph:
                return "SquaredEuclidean"
            return "Euclidean"
        if self.metric_type == MetricType.IP:
            return "InnerProduct"
        if self.metric_type == MetricType.COSINE:
            if self.index == IndexType.Hologres_Graph:
                return "InnerProduct"
            return "Cosine"
        return "Euclidean"

    def distance_function(self) -> str:
        if self.metric_type == MetricType.L2:
            if self.index == IndexType.Hologres_Graph:
                return "approx_squared_euclidean_distance"
            return "approx_euclidean_distance"
        if self.metric_type == MetricType.IP:
            return "approx_inner_product_distance"
        if self.metric_type == MetricType.COSINE:
            if self.index == IndexType.Hologres_Graph:
                return "approx_inner_product_distance"
            return "approx_cosine_distance"
        return "approx_euclidean_distance"

    def order_direction(self) -> str:
        if self.metric_type == MetricType.L2:
            return "ASC"
        if self.metric_type in {MetricType.IP, MetricType.COSINE}:
            return "DESC"
        return "ASC"

    def builder_params(self) -> dict:
        if self.use_reorder:
            self.base_quantization_type = "sq8_uniform"
        else:
            self.base_quantization_type = "fp32"

        return {
            "max_total_size_to_merge_mb": self.max_total_size_to_merge_mb,
            "build_thread_count": self.build_thread_count,
            "base_quantization_type": self.base_quantization_type,
            "max_degree": self.max_degree,
            "ef_construction": self.ef_construction,
            "precise_quantization_type": self.precise_quantization_type,
            "use_reorder": self.use_reorder,
            "precise_io_type": "reader_io",
        }

    def searcher_params(self) -> dict:
        return {
            "ef_search": self.ef_search,
        }
