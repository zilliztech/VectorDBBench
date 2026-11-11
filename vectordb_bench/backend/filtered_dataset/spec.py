"""Specification of a generated filtered-search fixture.

A spec fully determines the output: the same spec and seed produce identical files with the
same numpy and pyarrow versions on the same platform.
"""

from enum import StrEnum

from pydantic import Field, field_validator, model_validator

from ...base import BaseModel
from ..clients.api import MetricType

FORMAT_VERSION = 1
SUPPORTED_METRICS = (MetricType.L2, MetricType.COSINE, MetricType.IP)
# Keeps the brute-force ground truth and the in-memory arrays bounded on a laptop: at the element cap
# (e.g. 390k x 128) generation peaks around 1.3 GB, and the cap stays below the int32 list offsets
# Arrow uses for the vector column.
MAX_TRAIN_VECTORS = 2_000_000
MAX_TEST_VECTORS = 100_000
MAX_TRAIN_ELEMENTS = 50_000_000
MAX_INT_DOMAIN = 10_000_000


class Distribution(StrEnum):
    UNIFORM = "uniform"
    ZIPF = "zipf"


class CaseType(StrEnum):
    LABEL_EQUAL = "label_equal"
    INT_RANGE = "int_range"


class LabelSpec(BaseModel):
    """One categorical value per row, written to the `labels` column."""

    # The default Zipf spread (about 23% down to 0.15%) gives each default target its own label value.
    num_values: int = Field(default=100, ge=1)
    distribution: Distribution = Distribution.ZIPF
    zipf_s: float = Field(default=1.1, gt=0)
    # 0 assigns labels independently of the vectors; 1 makes them a function of vector position.
    correlation: float = Field(default=0.0, ge=0.0, le=1.0)


class IntSpec(BaseModel):
    """One integer attribute per row in [low, high], written to the `int_attr` column."""

    low: int = 0
    high: int = 9_999
    distribution: Distribution = Distribution.UNIFORM
    zipf_s: float = Field(default=1.1, gt=0)
    correlation: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_bounds(self) -> "IntSpec":
        if self.high < self.low:
            msg = f"int_attr high ({self.high}) must be >= low ({self.low})"
            raise ValueError(msg)
        if self.high - self.low + 1 > MAX_INT_DOMAIN:
            msg = f"int_attr spans {self.high - self.low + 1} values; at most {MAX_INT_DOMAIN} are supported"
            raise ValueError(msg)
        return self


class FilterCaseSpec(BaseModel):
    """A filter whose matching fraction should be close to `target_selectivity`.

    Selectivity here is the fraction of rows that MATCH the filter (VectorDBBench's
    `filter_rate` is the fraction filtered out, i.e. 1 - selectivity).
    """

    type: CaseType
    target_selectivity: float = Field(gt=0.0, le=1.0)
    # int_range only: where the range starts, as a quantile of int_attr.
    start_quantile: float = Field(default=0.0, ge=0.0, lt=1.0)


class FilteredDatasetSpec(BaseModel):
    name: str = "filtered_synthetic"
    seed: int = 42
    num_train: int = Field(default=100_000, ge=1, le=MAX_TRAIN_VECTORS)
    num_test: int = Field(default=1_000, ge=1, le=MAX_TEST_VECTORS)
    dim: int = Field(default=128, ge=1, le=4096)
    metric_type: MetricType = MetricType.L2
    num_clusters: int = Field(default=32, ge=1)
    cluster_std: float = Field(default=0.25, gt=0)
    k: int = Field(default=100, ge=1)
    labels: LabelSpec = LabelSpec()
    int_attr: IntSpec = IntSpec()
    cases: list[FilterCaseSpec] = Field(
        default_factory=lambda: [
            FilterCaseSpec(type=CaseType.LABEL_EQUAL, target_selectivity=s) for s in (0.01, 0.05, 0.2)
        ]
        + [FilterCaseSpec(type=CaseType.INT_RANGE, target_selectivity=s) for s in (0.01, 0.1, 0.5)]
    )

    @field_validator("metric_type")
    @classmethod
    def check_metric(cls, v: MetricType) -> MetricType:
        if v not in SUPPORTED_METRICS:
            msg = f"metric_type must be one of {[m.value for m in SUPPORTED_METRICS]}, got {v}"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def check_sizes(self) -> "FilteredDatasetSpec":
        if self.num_train * self.dim > MAX_TRAIN_ELEMENTS:
            msg = (
                f"num_train * dim is {self.num_train * self.dim}; at most {MAX_TRAIN_ELEMENTS} elements are "
                "supported, so reduce num_train or dim"
            )
            raise ValueError(msg)
        if self.labels.num_values > self.num_train:
            msg = f"labels.num_values ({self.labels.num_values}) cannot exceed num_train ({self.num_train})"
            raise ValueError(msg)
        return self
