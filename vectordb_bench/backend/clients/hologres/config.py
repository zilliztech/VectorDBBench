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
    full_compact_max_file_size_mb: int = 16384

    # Base quantization type for HGraph index.
    # Available values: "rabitq", "sq8_uniform", "fp32"
    # When use_reorder=False, this is ignored and "fp32" is used (no reorder requires full precision).
    # "rabitq" requires use_reorder=True; rabitq_use_fht is automatically enabled when rabitq is selected.
    quantization_method: str = "rabitq"
    precise_quantization_type: str = "fp32"
    # Storage medium for the precise (high-precision) index. Only effective when use_reorder=True.
    # "block_memory_io": both base and precise indexes in memory.
    # "reader_io": base index in memory, precise index on disk.
    precise_io_type: str = "block_memory_io"
    use_reorder: bool = True
    build_thread_count: int = 16
    max_degree: int = 64
    ef_construction: int = 400

    # When True, embeds the primary key ("id") in the HGraph index via extra_columns,
    # avoiding a base-table lookup during search.
    use_extra_column_id: bool = True

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
            "searcher_params": self.searcher_params(),
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
        base_quantization_type = self.quantization_method if self.use_reorder else "fp32"

        params = {
            "max_total_size_to_merge_mb": self.max_total_size_to_merge_mb,
            "build_thread_count": self.build_thread_count,
            "base_quantization_type": base_quantization_type,
            "max_degree": self.max_degree,
            "ef_construction": self.ef_construction,
            "precise_quantization_type": self.precise_quantization_type,
            "use_reorder": self.use_reorder,
        }

        if self.use_reorder:
            params["precise_io_type"] = self.precise_io_type

        if base_quantization_type == "rabitq":
            params["rabitq_use_fht"] = True

        if self.use_extra_column_id:
            params["extra_columns"] = "id"

        return params

    def searcher_params(self) -> dict:
        return {
            "ef_search": self.ef_search,
        }
