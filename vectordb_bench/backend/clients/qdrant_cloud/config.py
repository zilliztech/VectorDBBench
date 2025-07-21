from typing import TypeVar

from pydantic import BaseModel, SecretStr, validator

from ..api import DBCaseConfig, DBConfig, MetricType

# define type "SearchParams"
SearchParams = TypeVar("SearchParams")


# Allowing `api_key` to be left empty, to ensure compatibility with the open-source Qdrant.
class QdrantConfig(DBConfig):
    url: SecretStr
    api_key: SecretStr | None = None

    def to_dict(self) -> dict:
        api_key_value = self.api_key.get_secret_value() if self.api_key else None
        if api_key_value:
            return {
                "url": self.url.get_secret_value(),
                "api_key": api_key_value,
                "prefer_grpc": True,
            }
        return {
            "url": self.url.get_secret_value(),
        }

    @validator("*")
    def not_empty_field(cls, v: any, field: any):
        if field.name in ["api_key"]:
            return v
        return super().not_empty_field(v, field)


class QdrantIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    m: int = 16
    payload_m: int = 16  # only for label_filter cases
    create_payload_int_index: bool = False
    create_payload_keyword_index: bool = False
    is_tenant: bool = False
    use_scalar_quant: bool = False
    sq_quantile: float = 0.99
    default_segment_number: int = 0

    use_rescore: bool = False
    oversampling: float = 1.0
    indexed_only: bool = False
    hnsw_ef: int | None = 100
    exact: bool = False

    with_payload: bool = False

    def __eq__(self, obj: any):
        return (
            self.m == obj.m
            and self.payload_m == obj.payload_m
            and self.create_payload_int_index == obj.create_payload_int_index
            and self.create_payload_keyword_index == obj.create_payload_keyword_index
            and self.is_tenant == obj.is_tenant
            and self.use_scalar_quant == obj.use_scalar_quant
            and self.sq_quantile == obj.sq_quantile
            and self.default_segment_number == obj.default_segment_number
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.m,
                self.payload_m,
                self.create_payload_int_index,
                self.create_payload_keyword_index,
                self.is_tenant,
                self.use_scalar_quant,
                self.sq_quantile,
                self.default_segment_number,
            )
        )

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "Euclid"

        if self.metric_type == MetricType.IP:
            return "Dot"

        return "Cosine"

    def index_param(self) -> dict:
        return {"distance": self.parse_metric()}

    def search_param(self) -> SearchParams:
        # Import while in use
        from qdrant_client.http.models import QuantizationSearchParams, SearchParams

        quantization = (
            QuantizationSearchParams(
                ignore=False,
                rescore=True,
                oversampling=self.oversampling,
            )
            if self.use_rescore
            else None
        )
        return SearchParams(
            hnsw_ef=self.hnsw_ef,
            exact=self.exact,
            indexed_only=self.indexed_only,
            quantization=quantization,
        )
