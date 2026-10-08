from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, MetricType


class DynamoDBConfig(DBConfig):
    """Connection config for Amazon DynamoDB vector search.

    Credentials are optional: when access_key_id / secret_access_key are left
    empty the standard boto3 credential chain is used (environment variables,
    shared config/credentials files, or an attached IAM role).
    """

    region_name: str = "us-east-1"
    access_key_id: SecretStr | None = None
    secret_access_key: SecretStr | None = None
    session_token: SecretStr | None = None
    table_name: str = "vdbbench_vectors"
    index_name: str = "vdbbench-index"

    def to_dict(self) -> dict:
        return {
            "region_name": self.region_name,
            "access_key_id": (self.access_key_id.get_secret_value() if self.access_key_id else None),
            "secret_access_key": (self.secret_access_key.get_secret_value() if self.secret_access_key else None),
            "session_token": (self.session_token.get_secret_value() if self.session_token else None),
            "table_name": self.table_name,
            "index_name": self.index_name,
        }


class DynamoDBIndexConfig(DBCaseConfig, BaseModel):
    """Case config for a DynamoDB vector index.

    DynamoDB builds and manages the vector index server-side, so there are no
    client-tunable ANN build/search params. What is configurable is the
    SearchSchema:

    - metric_type:    the distance function (set at index creation).
    - num_partitions: when > 1, the client defines a SearchSchema HASH key
      (vector index partition key) and spreads vectors UNIFORMLY across this
      many partition values (``id % num_partitions``). This is a balanced
      baseline: every partition holds an equal share, so there are no hot or
      empty partitions. Real-world partition keys (Category, tenant, ...) are
      usually skewed, so these numbers represent the best-case, evenly-balanced
      scenario rather than skewed production behaviour. Each SearchVectors call
      is scoped to ONE partition value (chosen at random per query), the
      realistic single-partition access pattern the feature is built for. This
      searches only ~1/num_partitions of the data, so recall measured against
      VectorDBBench's whole-dataset ground truth is expected to be about
      1/num_partitions; QPS/latency reflect the scoped search. Use 1 (default)
      for a whole-index recall benchmark with no partition key.

    A vector index partition key must be a low-to-medium cardinality attribute
    and, once defined, its value is REQUIRED in the SearchConditionExpression
    of every SearchVectors call (per the DynamoDB vector search docs).
    """

    metric_type: MetricType | None = None
    num_partitions: int = 1

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.COSINE:
            return "COSINE"
        if self.metric_type == MetricType.L2:
            return "EUCLIDEAN"
        if self.metric_type == MetricType.IP:
            return "DOT_PRODUCT"
        msg = f"Unsupported metric type for DynamoDB: {self.metric_type}"
        raise ValueError(msg)

    def use_partition_key(self) -> bool:
        return self.num_partitions > 1

    def index_param(self) -> dict:
        return {"num_partitions": self.num_partitions}

    def search_param(self) -> dict:
        return {}
