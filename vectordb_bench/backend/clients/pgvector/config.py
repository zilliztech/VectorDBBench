from pydantic import BaseModel, SecretStr
from ..api import DBConfig, DBCaseConfig, IndexType, MetricType

POSTGRE_URL_PLACEHOLDER = "postgresql://%s:%s@%s/%s"

class PgVectorConfig(DBConfig):
    user_name: SecretStr = "postgres"
    password: SecretStr
    host: str = "localhost"
    port: int = 5432
    db_name: str

    def to_dict(self) -> dict:
        user_str = self.user_name.get_secret_value()
        pwd_str = self.password.get_secret_value()
        return {
            "host" : self.host,
            "port" : self.port,
            "dbname" : self.db_name,
            "user" : user_str,
            "password" : pwd_str
        }

class PgVectorIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    index: IndexType
    lists: int | None = 1000
    probes: int | None = 10

    def parse_metric(self) -> str: 
        if self.metric_type == MetricType.L2:
            return "vector_l2_ops"
        elif self.metric_type == MetricType.IP:
            return "vector_ip_ops"
        return "vector_cosine_ops"
    
    def parse_metric_fun_op(self) -> str:
        if self.metric_type == MetricType.L2:
            return "<->"
        elif self.metric_type == MetricType.IP:
            return "<#>"
        return "<=>"
    
    def parse_metric_fun_str(self) -> str: 
        if self.metric_type == MetricType.L2:
            return "l2_distance"
        elif self.metric_type == MetricType.IP:
            return "max_inner_product"
        return "cosine_distance"

    

class HNSWConfig(PgVectorIndexConfig):
    m: int
    ef_construction: int
    ef: int | None = None
    index: IndexType = IndexType.HNSW

    def index_param(self) -> dict:
        return {
            "m" : self.m,
            "ef_construction" : self.ef_construction,
            "metric" : self.parse_metric()
        }

    def search_param(self) -> dict:
        return {
            "ef" : self.ef,
            "metric_fun" : self.parse_metric_fun_str(),
            "metric_fun_op" : self.parse_metric_fun_op(),
        }


class IVFFlatConfig(PgVectorIndexConfig):
    lists: int
    probes: int | None = None
    index: IndexType = IndexType.IVFFlat

    def index_param(self) -> dict:
        return {
            "lists" : self.lists,
            "metric" : self.parse_metric()
        }

    def search_param(self) -> dict:
        return {
            "probes" : self.probes,
            "metric_fun" : self.parse_metric_fun_str(),
            "metric_fun_op" : self.parse_metric_fun_op(),
        }

_pgvector_case_config = {
    IndexType.HNSW: HNSWConfig,
    IndexType.IVFFlat: IVFFlatConfig,
}