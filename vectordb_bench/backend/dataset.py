"""
Usage:
    >>> from xxx.dataset import Dataset
    >>> Dataset.Cohere.get(100_000)
"""

import fnmatch
import glob
import json
import logging
import math
import pathlib
import tempfile
import types
import typing
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, NamedTuple

import h5py
import ir_datasets
import numpy as np
import pandas as pd
import polars as pl
from pyarrow.parquet import ParquetFile
from pydantic import Field as PydanticField
from pydantic import PrivateAttr, field_validator

from vectordb_bench import config
from vectordb_bench.base import BaseModel

from . import utils
from .clients import MetricType
from .data_source import DatasetReader, DatasetSource
from .filter import Filter, FilterOp, NewIntFilter, non_filter

log = logging.getLogger(__name__)
DEFAULT_INSERT_BATCH_SIZE = config.DEFAULT_INSERT_BATCH_SIZE


class SizeLabel(NamedTuple):
    size: int
    label: str
    file_count: int


class BaseDataset(BaseModel):
    name: str
    size: int
    dim: int
    metric_type: MetricType
    use_shuffled: bool
    with_gt: bool = False
    _size_label: ClassVar[dict[int, SizeLabel]]
    is_custom: bool = False
    with_remote_resource: bool = True
    # for label filter cases
    with_scalar_labels: bool = False
    # if True, scalar_labels will be retrieved from a separate parquet file;
    #   otherwise, they will be obtained from train.parquet.
    scalar_labels_file_separated: bool = True
    scalar_labels_file: str = "scalar_labels.parquet"
    scalar_label_percentages: list[float] = []
    scalar_int_rates: list[float] = []
    train_id_field: str = "id"
    train_vector_field: str = "emb"
    test_file: str = "test.parquet"
    test_id_field: str = "id"
    test_vector_field: str = "emb"
    gt_id_field: str = "id"
    gt_neighbors_field: str = "neighbors_id"
    source: DatasetSource | None = None
    source_dataset: str | None = None
    source_revision: str | None = None
    dataset_metadata: dict[str, Any] | None = None

    @field_validator("size")
    @classmethod
    def verify_size(cls, v: int):
        if v not in cls._size_label:
            msg = f"Size {v} not supported for the dataset, expected: {cls._size_label.keys()}"
            raise ValueError(msg)
        return v

    @property
    def label(self) -> str:
        return self._size_label.get(self.size).label

    @property
    def full_name(self) -> str:
        return f"{self.name.capitalize()} ({self.label.capitalize()})"

    @property
    def dir_name(self) -> str:
        return f"{self.name}_{self.label}_{utils.numerize(self.size)}".lower()

    @property
    def file_count(self) -> int:
        return self._size_label.get(self.size).file_count

    @property
    def train_files(self) -> list[str]:
        return utils.compose_train_files(self.file_count, self.use_shuffled)


class CustomDataset(BaseDataset):
    dir: str
    file_num: int
    is_custom: bool = True
    with_remote_resource: bool = False
    train_file: str = "train"
    train_id_field: str = "id"
    train_vector_field: str = "emb"
    test_file: str = "test.parquet"
    gt_file: str = "neighbors.parquet"
    test_vector_field: str = "emb"
    gt_neighbors_field: str = "neighbors_id"
    with_scalar_labels: bool = True
    scalar_labels_file_separated: bool = True
    scalar_labels_file: str = "scalar_labels.parquet"
    label_percentages: list[float] = []

    @field_validator("size")
    @classmethod
    def verify_size(cls, v: int):
        return v

    @property
    def label(self) -> str:
        return "Custom"

    @property
    def dir_name(self) -> str:
        return self.dir

    @property
    def file_count(self) -> int:
        return self.file_num

    @property
    def train_files(self) -> list[str]:
        if ("," not in self.train_file) and self.file_num > 1:
            return utils.compose_train_files(self.file_num, self.use_shuffled)
        train_file = self.train_file
        prefix = f"{train_file}"
        train_files = []
        prefix_s = [item.strip() for item in prefix.split(",") if item.strip()]
        for i in range(len(prefix_s)):
            sub_file = f"{prefix_s[i]}.parquet"
            train_files.append(sub_file)
        return train_files


class ParquetDataset(BaseDataset):
    """Artifact roles and schema for a Parquet vector dataset."""

    train_selectors: tuple[str, ...]
    query_selectors: tuple[str, ...]
    gt_selector: str
    ground_truth_width: int
    query_count: int | None = None
    family: str | None = None
    point_type: str | None = None

    @field_validator("size")
    @classmethod
    def verify_size(cls, value: int) -> int:
        if value <= 0:
            msg = f"Dataset size must be positive, got {value}"
            raise ValueError(msg)
        return value

    @property
    def label(self) -> str:
        return self.family or "Parquet"

    @property
    def full_name(self) -> str:
        return self.name

    @property
    def dir_name(self) -> str:
        return self.name

    @property
    def train_files(self) -> list[str]:
        return list(self.train_selectors)


class LAION(BaseDataset):
    name: str = "LAION"
    dim: int = 768
    metric_type: MetricType = MetricType.L2
    use_shuffled: bool = False
    with_gt: bool = True
    with_scalar_labels: bool = True
    scalar_label_percentages: list[float] = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    _size_label: ClassVar[dict[int, SizeLabel]] = {
        100_000_000: SizeLabel(100_000_000, "LARGE", 100),
    }


@dataclass(frozen=True)
class SearchDatasetFiles:
    test_file: str | tuple[str, ...]
    gt_file: str
    width: int | None = None
    query_count: int | None = None

    @property
    def test_files(self) -> tuple[str, ...]:
        if isinstance(self.test_file, str):
            return (self.test_file,)
        return self.test_file


LAION_SEARCH_DATASET_FILES = (
    (1_000, SearchDatasetFiles("test.parquet", "neighbors.parquet", width=1_000, query_count=1_000)),
    (
        100_000,
        SearchDatasetFiles(
            "test_nq200.parquet",
            "neighbors_top100k_nq200.parquet",
            width=100_000,
            query_count=200,
        ),
    ),
    (
        1_000_000,
        SearchDatasetFiles(
            "test_nq200.parquet",
            "neighbors_top1m_nq200.parquet",
            width=1_000_000,
            query_count=200,
        ),
    ),
)

# Published widths are capped by the population left after applying each ID threshold.
LAION_INT_FILTER_SEARCH_WIDTHS: dict[float, tuple[int, ...]] = {
    0.5: (100_000, 1_000_000),
    0.6: (100_000, 1_000_000),
    0.7: (100_000, 1_000_000),
    0.8: (100_000, 1_000_000),
    0.9: (100_000, 1_000_000),
    0.95: (100_000, 1_000_000),
    0.98: (100_000, 1_000_000),
    0.99: (100_000, 1_000_000),
    0.995: (100_000, 500_000),
    0.998: (100_000, 200_000),
    0.999: (100_000,),
}


@dataclass(frozen=True)
class ParquetGroundTruth:
    path: pathlib.Path
    neighbors_field: str
    row_count: int
    width: int

    @classmethod
    def from_file(
        cls,
        path: pathlib.Path,
        *,
        id_field: str,
        neighbors_field: str,
        expected_query_ids: typing.Sequence[Any],
        minimum_width: int,
        expected_width: int | None = None,
    ) -> "ParquetGroundTruth":
        if not path.exists():
            msg = f"No such file: {path}"
            raise FileNotFoundError(msg)

        parquet_file = ParquetFile(path, memory_map=True, pre_buffer=False)
        schema_names = parquet_file.schema_arrow.names
        missing_fields = [field for field in (id_field, neighbors_field) if field not in schema_names]
        if missing_fields:
            msg = f"Ground truth file {path} is missing fields: {missing_fields}"
            raise ValueError(msg)

        query_ids = parquet_file.read(columns=[id_field]).column(0).to_pylist()
        if query_ids != list(expected_query_ids):
            msg = f"Ground truth query IDs in {path} do not match the selected query file"
            raise ValueError(msg)

        row_count = parquet_file.metadata.num_rows
        minimum_observed_width = None
        observed_rows = 0
        for batch in parquet_file.iter_batches(batch_size=1, columns=[neighbors_field]):
            for row in batch.column(0):
                if not row.is_valid:
                    msg = f"Ground truth file {path} contains a null neighbors row"
                    raise ValueError(msg)
                width = len(row.values)
                observed_rows += 1
                minimum_observed_width = width if minimum_observed_width is None else min(minimum_observed_width, width)
                if expected_width is not None and width != expected_width:
                    msg = f"Ground truth width {width} in {path} does not match expected width {expected_width}"
                    raise ValueError(msg)
                if width < minimum_width:
                    msg = f"Ground truth width {width} in {path} is smaller than requested K={minimum_width}"
                    raise ValueError(msg)

        if observed_rows != row_count or minimum_observed_width is None:
            msg = f"Ground truth row count in {path} is invalid: expected {row_count}, read {observed_rows}"
            raise ValueError(msg)

        return cls(
            path=path,
            neighbors_field=neighbors_field,
            row_count=row_count,
            width=minimum_observed_width,
        )

    def __len__(self) -> int:
        return self.row_count

    def iter_rows(self) -> Iterator[Any]:
        parquet_file = ParquetFile(self.path, memory_map=True, pre_buffer=False)
        for batch in parquet_file.iter_batches(batch_size=1, columns=[self.neighbors_field]):
            for row in batch.column(0):
                yield row.values.to_numpy(zero_copy_only=False)  # noqa: PD011


class GIST(BaseDataset):
    name: str = "GIST"
    dim: int = 960
    metric_type: MetricType = MetricType.L2
    use_shuffled: bool = False
    _size_label: ClassVar[dict[int, SizeLabel]] = {
        100_000: SizeLabel(100_000, "SMALL", 1),
        1_000_000: SizeLabel(1_000_000, "MEDIUM", 1),
    }


class Cohere(BaseDataset):
    name: str = "Cohere"
    dim: int = 768
    metric_type: MetricType = MetricType.COSINE
    use_shuffled: bool = config.USE_SHUFFLED_DATA
    with_gt: bool = True
    _size_label: ClassVar[dict[int, SizeLabel]] = {
        100_000: SizeLabel(100_000, "SMALL", 1),
        1_000_000: SizeLabel(1_000_000, "MEDIUM", 1),
        10_000_000: SizeLabel(10_000_000, "LARGE", 10),
    }
    with_scalar_labels: bool = True
    scalar_label_percentages: list[float] = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    scalar_int_rates: list[float] = [
        0.001,
        0.002,
        0.005,
        0.01,
        0.02,
        0.05,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.95,
        0.98,
        0.99,
        0.995,
        0.998,
        0.999,
    ]


class Bioasq(BaseDataset):
    name: str = "Bioasq"
    dim: int = 1024
    metric_type: MetricType = MetricType.COSINE
    use_shuffled: bool = config.USE_SHUFFLED_DATA
    with_gt: bool = True
    _size_label: ClassVar[dict[int, SizeLabel]] = {
        1_000_000: SizeLabel(1_000_000, "MEDIUM", 1),
        10_000_000: SizeLabel(10_000_000, "LARGE", 10),
    }
    with_scalar_labels: bool = True
    scalar_label_percentages: list[float] = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    scalar_int_rates: list[float] = [
        0.001,
        0.002,
        0.005,
        0.01,
        0.02,
        0.05,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.95,
        0.98,
        0.99,
        0.995,
        0.998,
        0.999,
    ]


class Glove(BaseDataset):
    name: str = "Glove"
    dim: int = 200
    metric_type: MetricType = MetricType.COSINE
    use_shuffled: bool = False
    _size_label: ClassVar[dict[int, SizeLabel]] = {1_000_000: SizeLabel(1_000_000, "MEDIUM", 1)}


class SIFT(BaseDataset):
    name: str = "SIFT"
    dim: int = 128
    metric_type: MetricType = MetricType.L2
    use_shuffled: bool = False
    _size_label: ClassVar[dict[int, SizeLabel]] = {
        500_000: SizeLabel(
            500_000,
            "SMALL",
            1,
        ),
        5_000_000: SizeLabel(5_000_000, "MEDIUM", 1),
        #  50_000_000: SizeLabel(50_000_000, "LARGE", 50),
    }


class OpenAI(BaseDataset):
    name: str = "OpenAI"
    dim: int = 1536
    metric_type: MetricType = MetricType.COSINE
    use_shuffled: bool = config.USE_SHUFFLED_DATA
    with_gt: bool = True
    _size_label: ClassVar[dict[int, SizeLabel]] = {
        50_000: SizeLabel(50_000, "SMALL", 1),
        500_000: SizeLabel(500_000, "MEDIUM", 1),
        5_000_000: SizeLabel(5_000_000, "LARGE", 10),
    }
    with_scalar_labels: bool = True
    scalar_label_percentages: list[float] = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    scalar_int_rates: list[float] = [
        0.001,
        0.002,
        0.005,
        0.01,
        0.02,
        0.05,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.95,
        0.98,
        0.99,
        0.995,
        0.998,
        0.999,
    ]


class DatasetManager(BaseModel, ABC):
    """Common in-memory contract consumed by vector benchmark runners."""

    data: BaseDataset
    load_timeout: float | int = config.LOAD_TIMEOUT_DEFAULT
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_DEFAULT
    test_data: list[list[float]] | None = None
    gt_data: ParquetGroundTruth | list[list[int]] | None = None
    search_files: SearchDatasetFiles | None = None
    scalar_labels: pl.DataFrame | None = None
    train_files: list[str] = []
    reader: DatasetReader | None = None
    resolved_files: dict[str, pathlib.Path] = {}
    result_metadata: dict[str, Any] | None = None

    def __eq__(self, obj: any):
        if isinstance(obj, DatasetManager):
            return self.data.name == obj.data.name and self.data.label == obj.data.label
        return False

    def __hash__(self) -> int:
        return hash((self.data.name, self.data.label))

    def set_reader(self, reader: DatasetReader):
        self.reader = reader

    @property
    def preferred_source(self) -> DatasetSource | None:
        return self.data.source

    @property
    def data_dir(self) -> pathlib.Path:
        """data local directory: config.DATASET_LOCAL_DIR/{dataset_name}/{dataset_dirname}

        Examples:
            >>> sift_s = Dataset.SIFT.manager(500_000)
            >>> sift_s.relative_path
            '/tmp/vectordb_bench/dataset/sift/sift_small_500k/'
        """
        return pathlib.Path(
            config.DATASET_LOCAL_DIR,
            self.data.name.lower(),
            self.data.dir_name,
        )

    def __iter__(self):
        return self.iter_batches(DEFAULT_INSERT_BATCH_SIZE)

    @abstractmethod
    def iter_batches(self, batch_size: int):
        """Return insertion batches without materializing the whole corpus."""

    @abstractmethod
    def prepare(
        self,
        source: DatasetSource = DatasetSource.S3,
        filters: Filter = non_filter,
        with_train_files: bool = True,
        with_scalar_labels: bool = False,
        k: int | None = None,
    ) -> bool:
        """Resolve the source and load query and ground-truth data."""

    def max_search_k(self, filters: Filter = non_filter) -> int | None:
        return None

    @abstractmethod
    def resolve_search_files(self, *, k: int, filters: Filter = non_filter) -> SearchDatasetFiles:
        """Validate search inputs and describe the query and ground-truth data."""

    def _local_path(self, file_name: str) -> pathlib.Path:
        return self.resolved_files.get(file_name, pathlib.Path(self.data_dir, file_name))


class ParquetDatasetManager(DatasetManager):
    """Prepare and stream datasets stored as Parquet files."""

    def iter_batches(self, batch_size: int):
        return ParquetDatasetIterator(self, batch_size=batch_size)

    # TODO passing use_shuffle from outside
    def prepare(
        self,
        source: DatasetSource = DatasetSource.S3,
        filters: Filter = non_filter,
        with_train_files: bool = True,
        with_scalar_labels: bool = False,
        k: int | None = None,
    ) -> bool:
        """Download the dataset from DatasetSource
         url = f"{source}/{self.data.dir_name}"

        Args:
            source(DatasetSource): S3 or AliyunOSS, default as S3
            filters(Filter): combined with dataset's with_gt to
              compose the correct ground_truth file
            k(int | None): requested search depth used to select and validate ground truth

        Returns:
            bool: whether the dataset is successfully prepared

        """
        requested_k = config.K_DEFAULT if k is None else k
        train_selectors = self.data.train_files if with_train_files else []
        gt_file = None
        test_selectors = ()
        if self.data.with_gt:
            self.search_files = self.resolve_search_files(k=requested_k, filters=filters)
            gt_file, test_selectors = self.search_files.gt_file, self.search_files.test_files

        actual_source = self.preferred_source or source
        if self.data.with_remote_resource:
            download_files = [*train_selectors, *test_selectors]
            if gt_file is not None:
                download_files.append(gt_file)
            if self.data.with_scalar_labels and self.data.scalar_labels_file_separated:
                download_files.append(self.data.scalar_labels_file)
            download_files = list(dict.fromkeys(download_files))
            self.resolved_files = actual_source.reader().read(
                dataset=self.data.source_dataset or self.data.dir_name.lower(),
                files=download_files,
                local_ds_root=self.data_dir,
                revision=self.data.source_revision,
            )

        self.train_files = self._resolve_selectors(train_selectors)
        test_files = self._resolve_selectors(test_selectors)
        resolved_gt_files = self._resolve_selectors((gt_file,)) if gt_file is not None else []
        if len(resolved_gt_files) > 1:
            msg = f"Ground truth selector {gt_file!r} resolved to multiple files"
            raise ValueError(msg)
        resolved_gt_file = resolved_gt_files[0] if resolved_gt_files else None
        needs_scalar_labels = filters.type == FilterOp.StrEqual or with_scalar_labels

        # read scalar_labels_file if separated
        if needs_scalar_labels and self.data.with_scalar_labels and self.data.scalar_labels_file_separated:
            self.scalar_labels = self._read_file(self.data.scalar_labels_file)

        if resolved_gt_file is not None and test_files:
            query_frames = [self._read_file(file) for file in test_files]
            test_frame = query_frames[0] if len(query_frames) == 1 else pl.concat(query_frames, how="vertical")
            if self.search_files.query_count is not None and len(test_frame) != self.search_files.query_count:
                msg = (
                    f"Query row count {len(test_frame)} in {test_files} does not match "
                    f"expected count {self.search_files.query_count}"
                )
                raise ValueError(msg)
            query_ids = test_frame[self.data.test_id_field].to_list()
            self.test_data = test_frame[self.data.test_vector_field].to_list()
            self.gt_data = ParquetGroundTruth.from_file(
                self._local_path(resolved_gt_file),
                id_field=self.data.gt_id_field,
                neighbors_field=self.data.gt_neighbors_field,
                expected_query_ids=query_ids,
                minimum_width=requested_k,
                expected_width=self.search_files.width,
            )

        if isinstance(self.data, ParquetDataset):
            self.result_metadata = self._result_metadata(
                actual_source,
                train_files=self.train_files,
                query_files=test_files,
                ground_truth_file=resolved_gt_file,
            )
        log.debug(f"{self.data.name}: available train files {self.train_files}")

        return True

    def max_search_k(self, filters: Filter = non_filter) -> int | None:
        if isinstance(self.data, ParquetDataset):
            self._validate_registered_filters(filters)
            return self.data.ground_truth_width
        if not isinstance(self.data, LAION):
            return None
        if isinstance(filters, NewIntFilter):
            widths = LAION_INT_FILTER_SEARCH_WIDTHS.get(filters.filter_rate)
            return widths[-1] if widths is not None else None
        if filters.type == FilterOp.NonFilter:
            return LAION_SEARCH_DATASET_FILES[-1][0]
        return LAION_SEARCH_DATASET_FILES[0][0]

    def resolve_search_files(self, *, k: int, filters: Filter = non_filter) -> SearchDatasetFiles:
        if k <= 0:
            msg = f"{self.data.name} search K must be positive, got {k}"
            raise ValueError(msg)

        if isinstance(self.data, ParquetDataset):
            self._validate_registered_filters(filters)
            if k > self.data.ground_truth_width:
                msg = f"{self.data.name} supports K from 1 to {self.data.ground_truth_width}, got {k}"
                raise ValueError(msg)
            return SearchDatasetFiles(
                self.data.query_selectors,
                self.data.gt_selector,
                width=self.data.ground_truth_width,
                query_count=self.data.query_count,
            )

        if isinstance(self.data, LAION):
            max_k = LAION_SEARCH_DATASET_FILES[-1][0]
            if k > max_k:
                msg = f"LAION supports K up to {max_k:,}, got {k:,}"
                raise ValueError(msg)

            if isinstance(filters, NewIntFilter):
                widths = LAION_INT_FILTER_SEARCH_WIDTHS.get(filters.filter_rate)
                if widths is None:
                    supported_rates = ", ".join(f"{rate * 100:g}%" for rate in LAION_INT_FILTER_SEARCH_WIDTHS)
                    msg = f"LAION supported filter rates are: {supported_rates}; got {filters.filter_rate * 100:g}%"
                    raise ValueError(msg)
                if k <= LAION_SEARCH_DATASET_FILES[0][0]:
                    return SearchDatasetFiles(self.data.test_file, filters.groundtruth_file)
                for width in widths:
                    if k <= width:
                        width_suffix = f"{width // 1_000_000}m" if width >= 1_000_000 else f"{width // 1_000}k"
                        return SearchDatasetFiles(
                            "test_nq200.parquet",
                            f"neighbors_{filters.int_rate}_top{width_suffix}_nq200.parquet",
                            width=width,
                            query_count=200,
                        )
                msg = (
                    f"LAION integer filter {filters.filter_rate * 100:g}% supports K up to "
                    f"{widths[-1]:,}, got {k:,}"
                )
                raise ValueError(msg)

            if filters.type != FilterOp.NonFilter:
                if k > LAION_SEARCH_DATASET_FILES[0][0]:
                    msg = "LAION large-TopK ground truth is published only for integer filters"
                    raise ValueError(msg)
                return SearchDatasetFiles(self.data.test_file, filters.groundtruth_file)
            for upper_bound, files in LAION_SEARCH_DATASET_FILES:
                if k <= upper_bound:
                    return files

        return SearchDatasetFiles(self.data.test_file, filters.groundtruth_file)

    @staticmethod
    def _validate_registered_filters(filters: Filter) -> None:
        if filters.type != FilterOp.NonFilter:
            msg = "Parquet dataset does not contain scalar fields or filtered ground truth"
            raise ValueError(msg)

    def _resolve_selectors(self, selectors: typing.Iterable[str]) -> list[str]:
        resolved = []
        for selector in selectors:
            matches = sorted(name for name in self.resolved_files if fnmatch.fnmatchcase(name, selector))
            if not matches and not glob.has_magic(selector) and self._local_path(selector).exists():
                matches = [selector]
            if not matches:
                msg = f"No dataset files match selector {selector!r}"
                raise FileNotFoundError(msg)
            for name in matches:
                if name not in resolved:
                    resolved.append(name)
        return resolved

    def _result_metadata(
        self,
        source: DatasetSource,
        *,
        train_files: list[str],
        query_files: list[str],
        ground_truth_file: str | None,
    ) -> dict[str, Any]:
        metadata = dict(self.data.dataset_metadata or {})
        metadata.update(
            {
                "name": self.data.name,
                "family": getattr(self.data, "family", None),
                "source": source.value,
                "repository": self.data.source_dataset,
                "revision": self.data.source_revision,
                "metric_type": self.data.metric_type.value,
                "point_type": getattr(self.data, "point_type", None),
                "storage_format": "parquet",
                "train_files": train_files,
                "query_files": query_files,
                "ground_truth_file": ground_truth_file,
            }
        )
        return metadata

    def _read_file(self, file_name: str) -> pl.DataFrame:
        """read one file from disk into memory"""
        log.info(f"Read the entire file into memory: {file_name}")
        p = self._local_path(file_name)
        if not p.exists():
            log.warning(f"No such file: {p}")
            return pl.DataFrame()

        return pl.read_parquet(p)


class ParquetDatasetIterator:
    def __init__(self, dataset: ParquetDatasetManager, batch_size: int = DEFAULT_INSERT_BATCH_SIZE):
        if batch_size <= 0:
            msg = f"insert batch size must be greater than 0, got {batch_size}"
            raise ValueError(msg)
        self._ds = dataset
        self._batch_size = batch_size
        self._idx = 0  # file number
        self._cur = None
        self._sub_idx = [0 for i in range(len(self._ds.train_files))]  # iter num for each file

    def __getstate__(self):
        """Custom pickle support to handle unpicklable generator."""
        state = self.__dict__.copy()
        # Remove the unpicklable generator from ParquetFile.iter_batches()
        state["_cur"] = None
        return state

    def __setstate__(self, state: Any):
        """Restore state after unpickling."""
        self.__dict__.update(state)

    def __iter__(self):
        return self

    def _get_iter(self, file_name: str):
        p = self._ds._local_path(file_name)
        log.info(f"Get iterator for {p.name}")
        if not p.exists():
            msg = f"No such file: {p}"
            log.warning(msg)
            raise IndexError(msg)
        return ParquetFile(p, memory_map=True, pre_buffer=True).iter_batches(self._batch_size)

    def __next__(self) -> pd.DataFrame:
        """return the data in the next file of the training list"""
        if self._idx < len(self._ds.train_files):
            if self._cur is None:
                file_name = self._ds.train_files[self._idx]
                self._cur = self._get_iter(file_name)

            try:
                return next(self._cur).to_pandas()
            except StopIteration:
                if self._idx == len(self._ds.train_files) - 1:
                    raise StopIteration from None

                self._idx += 1
                file_name = self._ds.train_files[self._idx]
                self._cur = self._get_iter(file_name)
                return next(self._cur).to_pandas()
        raise StopIteration


# Backwards-compatible name for callers that imported the Parquet iterator directly.
DataSetIterator = ParquetDatasetIterator


class Hdf5Dataset(BaseDataset):
    """Schema and source details for an HDF5 vector dataset."""

    file_name: str
    train_key: str = "train"
    test_key: str = "test"
    neighbors_key: str = "neighbors"
    distances_key: str | None = "distances"
    dimension_attr: str | None = "dimension"
    distance_attr: str | None = "distance"
    point_type_attr: str | None = "point_type"
    source_distance: str | None = None
    point_type: str | None = None
    ground_truth_width: int = 100
    family: str | None = None
    distribution: str | None = None
    modality: str | None = None

    @field_validator("size")
    @classmethod
    def verify_size(cls, value: int) -> int:
        if value <= 0:
            msg = f"Dataset size must be positive, got {value}"
            raise ValueError(msg)
        return value

    @property
    def label(self) -> str:
        return self.family or "HDF5"

    @property
    def full_name(self) -> str:
        return self.name

    @property
    def dir_name(self) -> str:
        return self.name

    @property
    def train_files(self) -> list[str]:
        return [self.file_name]


class Hdf5DatasetIterator:
    """Keep one HDF5 file open for the lifetime of an insertion iterator."""

    def __init__(self, dataset: Hdf5Dataset, path: pathlib.Path, batch_size: int):
        if batch_size <= 0:
            msg = f"insert batch size must be greater than 0, got {batch_size}"
            raise ValueError(msg)
        self._dataset = dataset
        self._batch_size = batch_size
        self._offset = 0
        self._file: h5py.File | None = None
        self._train: h5py.Dataset | None = None
        self._file = h5py.File(path, "r")
        self._train = self._file[dataset.train_key]

    def __iter__(self):
        return self

    def __next__(self) -> pd.DataFrame:
        if self._train is None or self._offset >= len(self._train):
            self.close()
            raise StopIteration

        end = min(self._offset + self._batch_size, len(self._train))
        vectors = np.ascontiguousarray(self._train[self._offset : end])
        batch = pd.DataFrame(
            {
                self._dataset.train_id_field: np.arange(self._offset, end, dtype=np.int64),
                self._dataset.train_vector_field: list(vectors),
            }
        )
        self._offset = end
        return batch

    def close(self) -> None:
        self._train = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        self.close()


class Hdf5DatasetManager(DatasetManager):
    """Prepare query data and stream corpus vectors directly from HDF5."""

    data: Hdf5Dataset
    source_path: pathlib.Path | None = None

    def iter_batches(self, batch_size: int):
        if not self.train_files:
            return iter(())
        if self.source_path is None:
            raise RuntimeError("HDF5 dataset is not prepared")
        return Hdf5DatasetIterator(self.data, self.source_path, batch_size)

    def max_search_k(self, filters: Filter = non_filter) -> int | None:
        self._validate_unfiltered(filters)
        return self.data.ground_truth_width

    def resolve_search_files(self, *, k: int, filters: Filter = non_filter) -> SearchDatasetFiles:
        self._validate_unfiltered(filters)
        if not 1 <= k <= self.data.ground_truth_width:
            msg = f"{self.data.name} supports K from 1 to {self.data.ground_truth_width}, got {k}"
            raise ValueError(msg)
        return SearchDatasetFiles(
            self.data.file_name,
            self.data.file_name,
            width=self.data.ground_truth_width,
        )

    def prepare(
        self,
        source: DatasetSource = DatasetSource.S3,
        filters: Filter = non_filter,
        with_train_files: bool = True,
        with_scalar_labels: bool = False,
        k: int | None = None,
    ) -> bool:
        if with_scalar_labels:
            msg = f"{self.data.name} does not provide scalar labels"
            raise ValueError(msg)
        requested_k = config.K_DEFAULT if k is None else k
        self.search_files = self.resolve_search_files(k=requested_k, filters=filters)
        actual_source = self.preferred_source or source
        if self.data.with_remote_resource:
            self.resolved_files = actual_source.reader().read(
                self.data.source_dataset or self.data.dir_name,
                [self.data.file_name],
                self.data_dir,
                revision=self.data.source_revision,
            )
        self.source_path = self._local_path(self.data.file_name)

        with h5py.File(self.source_path, "r") as hdf5:
            self._validate_source(hdf5)
            queries = np.ascontiguousarray(hdf5[self.data.test_key][:])
            neighbors = np.ascontiguousarray(hdf5[self.data.neighbors_key][:])

        if neighbors.size and (neighbors.min() < 0 or neighbors.max() >= self.data.size):
            msg = f"Neighbor ID is outside [0, {self.data.size}) for {self.data.name}"
            raise ValueError(msg)

        self.test_data = queries.tolist()
        self.gt_data = neighbors.tolist()
        self.train_files = self.data.train_files if with_train_files else []
        self.result_metadata = self._result_metadata(actual_source)
        return True

    @staticmethod
    def _validate_unfiltered(filters: Filter) -> None:
        if filters.type != FilterOp.NonFilter:
            msg = "HDF5 dataset does not contain scalar fields or filtered ground truth"
            raise ValueError(msg)

    def _validate_source(self, source: h5py.File) -> None:
        required_arrays = {self.data.train_key, self.data.test_key, self.data.neighbors_key}
        if self.data.distances_key is not None:
            required_arrays.add(self.data.distances_key)
        missing_arrays = required_arrays - set(source)
        if missing_arrays:
            msg = f"Invalid HDF5 {self.data.file_name}: missing arrays={sorted(missing_arrays)}"
            raise ValueError(msg)

        self._validate_attr(source, self.data.dimension_attr, self.data.dim)
        self._validate_attr(source, self.data.distance_attr, self.data.source_distance)
        self._validate_attr(source, self.data.point_type_attr, self.data.point_type)

        train = source[self.data.train_key]
        test = source[self.data.test_key]
        neighbors = source[self.data.neighbors_key]
        if train.shape != (self.data.size, self.data.dim):
            msg = f"Unexpected train shape for {self.data.name}: {train.shape}"
            raise ValueError(msg)
        if test.ndim != 2 or test.shape[1] != self.data.dim:
            msg = f"Unexpected test shape for {self.data.name}: {test.shape}"
            raise ValueError(msg)
        expected_gt_shape = (test.shape[0], self.data.ground_truth_width)
        if neighbors.shape != expected_gt_shape:
            msg = f"Unexpected ground-truth shape for {self.data.name}: {neighbors.shape}"
            raise ValueError(msg)
        if self.data.distances_key is not None and source[self.data.distances_key].shape != expected_gt_shape:
            msg = f"Unexpected distance shape for {self.data.name}: {source[self.data.distances_key].shape}"
            raise ValueError(msg)
        if train.dtype != test.dtype or not np.issubdtype(train.dtype, np.floating):
            msg = f"Unsupported vector dtype for {self.data.name}: {train.dtype}/{test.dtype}"
            raise ValueError(msg)
        if not np.issubdtype(neighbors.dtype, np.integer):
            msg = f"Neighbors must be integers, got {neighbors.dtype}"
            raise ValueError(msg)

    def _validate_attr(self, source: h5py.File, name: str | None, expected: object) -> None:
        if name is None or expected is None:
            return
        if name not in source.attrs:
            msg = f"Invalid HDF5 {self.data.file_name}: missing attribute {name}"
            raise ValueError(msg)
        actual = source.attrs[name]
        if isinstance(actual, bytes):
            actual = actual.decode("utf-8")
        if isinstance(expected, str):
            actual = str(actual).lower()
            expected = expected.lower()
        if actual != expected:
            msg = f"HDF5 attribute {name}={actual!r} does not match {expected!r} for {self.data.name}"
            raise ValueError(msg)

    def _result_metadata(self, source: DatasetSource) -> dict[str, Any]:
        metadata = dict(self.data.dataset_metadata or {})
        metadata.update(
            {
                "name": self.data.name,
                "source": source.value,
                "repository": self.data.source_dataset or "",
                "filename": self.data.file_name,
                "revision": self.data.source_revision or "",
                "source_distance": self.data.source_distance or "",
                "metric_type": self.data.metric_type.value,
                "point_type": self.data.point_type or "",
            }
        )
        return metadata


class Dataset(Enum):
    """
    Value is Dataset classes, DO NOT use it
    Example:
        >>> all_dataset = [ds.name for ds in Dataset]
        >>> Dataset.COHERE.manager(100_000)
        >>> Dataset.COHERE.get(100_000)
    """

    LAION = LAION
    GIST = GIST
    COHERE = Cohere
    BIOASQ = Bioasq
    GLOVE = Glove
    SIFT = SIFT
    OPENAI = OpenAI

    def get(self, size: int) -> BaseDataset:
        return self.value(size=size)

    def manager(
        self,
        size: int,
        *,
        load_timeout: float = config.LOAD_TIMEOUT_DEFAULT,
        optimize_timeout: float | None = config.OPTIMIZE_TIMEOUT_DEFAULT,
    ) -> DatasetManager:
        return ParquetDatasetManager(
            data=self.get(size),
            load_timeout=load_timeout,
            optimize_timeout=optimize_timeout,
        )


class DatasetWithSizeType(Enum):
    CohereSmall = "Small Cohere (768dim, 100K)"
    CohereMedium = "Medium Cohere (768dim, 1M)"
    CohereLarge = "Large Cohere (768dim, 10M)"
    LAIONLarge = "Large LAION (768dim, 100M)"
    BioasqMedium = "Medium Bioasq (1024dim, 1M)"
    BioasqLarge = "Large Bioasq (1024dim, 10M)"
    OpenAISmall = "Small OpenAI (1536dim, 50K)"
    OpenAIMedium = "Medium OpenAI (1536dim, 500K)"
    OpenAILarge = "Large OpenAI (1536dim, 5M)"

    def get_manager(self) -> DatasetManager:
        if self not in DatasetWithSizeMap:
            msg = f"wrong ScalarDatasetWithSizeType: {self.name}"
            raise ValueError(msg)
        return DatasetWithSizeMap.get(self)

    def get_load_timeout(self) -> float:
        if self is DatasetWithSizeType.LAIONLarge:
            return config.LOAD_TIMEOUT_768D_100M
        if "small" in self.value.lower():
            return config.LOAD_TIMEOUT_768D_100K
        if "medium" in self.value.lower():
            return config.LOAD_TIMEOUT_768D_1M
        if "large" in self.value.lower():
            return config.LOAD_TIMEOUT_768D_10M
        msg = f"No load_timeout for {self.value}"
        raise KeyError(msg)

    def get_optimize_timeout(self) -> float:
        if self is DatasetWithSizeType.LAIONLarge:
            return config.OPTIMIZE_TIMEOUT_768D_100M
        if "small" in self.value.lower():
            return config.OPTIMIZE_TIMEOUT_768D_100K
        if "medium" in self.value.lower():
            return config.OPTIMIZE_TIMEOUT_768D_1M
        if "large" in self.value.lower():
            return config.OPTIMIZE_TIMEOUT_768D_10M
        return config.OPTIMIZE_TIMEOUT_DEFAULT


DatasetWithSizeMap = {
    DatasetWithSizeType.CohereSmall: Dataset.COHERE.manager(100_000),
    DatasetWithSizeType.CohereMedium: Dataset.COHERE.manager(
        1_000_000,
        load_timeout=config.LOAD_TIMEOUT_768D_1M,
        optimize_timeout=config.OPTIMIZE_TIMEOUT_768D_1M,
    ),
    DatasetWithSizeType.CohereLarge: Dataset.COHERE.manager(
        10_000_000,
        load_timeout=config.LOAD_TIMEOUT_768D_10M,
        optimize_timeout=config.OPTIMIZE_TIMEOUT_768D_10M,
    ),
    DatasetWithSizeType.LAIONLarge: Dataset.LAION.manager(
        100_000_000,
        load_timeout=config.LOAD_TIMEOUT_768D_100M,
        optimize_timeout=config.OPTIMIZE_TIMEOUT_768D_100M,
    ),
    DatasetWithSizeType.BioasqMedium: Dataset.BIOASQ.manager(
        1_000_000,
        load_timeout=config.LOAD_TIMEOUT_1024D_1M,
        optimize_timeout=config.OPTIMIZE_TIMEOUT_1024D_1M,
    ),
    DatasetWithSizeType.BioasqLarge: Dataset.BIOASQ.manager(
        10_000_000,
        load_timeout=config.LOAD_TIMEOUT_1024D_10M,
        optimize_timeout=config.OPTIMIZE_TIMEOUT_1024D_10M,
    ),
    DatasetWithSizeType.OpenAISmall: Dataset.OPENAI.manager(
        50_000,
        load_timeout=3600,
    ),
    DatasetWithSizeType.OpenAIMedium: Dataset.OPENAI.manager(
        500_000,
        load_timeout=config.LOAD_TIMEOUT_1536D_500K,
        optimize_timeout=config.OPTIMIZE_TIMEOUT_1536D_500K,
    ),
    DatasetWithSizeType.OpenAILarge: Dataset.OPENAI.manager(
        5_000_000,
        load_timeout=config.LOAD_TIMEOUT_1536D_5M,
        optimize_timeout=config.OPTIMIZE_TIMEOUT_1536D_5M,
    ),
}


def _hdf5_manager(
    name: str,
    distribution: str,
    modality: str,
    size: int,
    dimension: int,
    source_distance: str,
    *,
    load_timeout: float = config.LOAD_TIMEOUT_DEFAULT,
    optimize_timeout: float | None = config.OPTIMIZE_TIMEOUT_DEFAULT,
) -> Hdf5DatasetManager:
    metric_type = (
        MetricType.L2
        if source_distance == "euclidean"
        else MetricType.IP if source_distance == "ip" else MetricType.COSINE
    )
    return Hdf5DatasetManager(
        load_timeout=load_timeout,
        optimize_timeout=optimize_timeout,
        data=Hdf5Dataset(
            name=name,
            size=size,
            dim=dimension,
            metric_type=metric_type,
            use_shuffled=False,
            with_gt=True,
            with_remote_resource=True,
            file_name=f"{name}.hdf5",
            source=DatasetSource.HuggingFace,
            source_dataset="vector-index-bench/vibe",
            source_revision="07b387891a221b7b073b83d2f752b76462e5fa03",
            source_distance=source_distance,
            point_type="float",
            family="VIBE",
            distribution=distribution,
            modality=modality,
            dataset_metadata={"distribution": distribution},
        ),
    )


def _parquet_manager(
    name: str,
    size: int,
    repository: str,
    revision: str,
    train_selectors: tuple[str, ...],
    query_selectors: tuple[str, ...],
    gt_selector: str,
    *,
    load_timeout: float = config.LOAD_TIMEOUT_DEFAULT,
    optimize_timeout: float | None = config.OPTIMIZE_TIMEOUT_DEFAULT,
) -> ParquetDatasetManager:
    return ParquetDatasetManager(
        load_timeout=load_timeout,
        optimize_timeout=optimize_timeout,
        data=ParquetDataset(
            name=name,
            size=size,
            dim=4096,
            metric_type=MetricType.IP,
            use_shuffled=False,
            with_gt=True,
            source=DatasetSource.HuggingFace,
            source_dataset=repository,
            source_revision=revision,
            train_selectors=train_selectors,
            query_selectors=query_selectors,
            gt_selector=gt_selector,
            gt_neighbors_field="neighbors",
            ground_truth_width=100,
            query_count=10_000,
            family="VDBBench",
            point_type="float32",
            dataset_metadata={
                "normalization": "l2",
                "model": "Qwen3-VL-Embedding-8B",
            },
        ),
    )


_PARQUET_DATASETS = (
    _parquet_manager(
        "multimodal-embedding-1m",
        1_000_000,
        "VDBBench/multimodal-embedding-1M",
        "4a13d5b19c13121c5201f5d4cd8877c082ef6a0c",
        ("train.parquet",),
        ("test.parquet",),
        "neighbors.parquet",
    ),
    _parquet_manager(
        "multimodal-embedding-10m",
        10_000_000,
        "VDBBench/multimodal-embedding-10M",
        "4275de9e83dccfafa044eafad67fb4e8a3a5f6e0",
        ("data/train-*.parquet",),
        ("data/test-*.parquet",),
        "data/neighbors.parquet",
        load_timeout=config.LOAD_TIMEOUT_768D_10M,
        optimize_timeout=config.OPTIMIZE_TIMEOUT_768D_10M,
    ),
    _parquet_manager(
        "multimodal-embedding-100m",
        100_000_000,
        "VDBBench/multimodal-embedding-100M",
        "560b5909ed6b03441b0b536485c350a08eee06c5",
        ("train/shard-*/*.parquet",),
        ("test/*.parquet",),
        "neighbors/neighbors.parquet",
        load_timeout=config.LOAD_TIMEOUT_768D_100M,
        optimize_timeout=config.OPTIMIZE_TIMEOUT_768D_100M,
    ),
)


_HDF5_DATASETS = (
    _hdf5_manager("agnews-mxbai-1024-euclidean", "id", "Text", 769_382, 1024, "euclidean"),
    _hdf5_manager("arxiv-nomic-768-normalized", "id", "Text", 1_344_643, 768, "normalized"),
    _hdf5_manager(
        "dpr-jina-768-normalized",
        "id",
        "Text",
        20_969_760,
        768,
        "normalized",
        load_timeout=config.LOAD_TIMEOUT_768D_10M,
        optimize_timeout=config.OPTIMIZE_TIMEOUT_768D_10M,
    ),
    _hdf5_manager("glove-200-cosine", "id", "Word", 1_192_514, 200, "cosine"),
    _hdf5_manager("gooaq-distilroberta-768-normalized", "id", "Text", 1_475_024, 768, "normalized"),
    _hdf5_manager("imagenet-clip-512-normalized", "id", "Image", 1_281_167, 512, "normalized"),
    _hdf5_manager("inaturalist-resnet-2048-cosine", "id", "Image", 499_000, 2048, "cosine"),
    _hdf5_manager("landmark-dino-768-cosine", "id", "Image", 760_757, 768, "cosine"),
    _hdf5_manager("landmark-nomic-768-normalized", "id", "Image", 760_757, 768, "normalized"),
    _hdf5_manager(
        "msmarco-qwen-1024-normalized",
        "id",
        "Text",
        8_840_823,
        1024,
        "normalized",
        load_timeout=config.LOAD_TIMEOUT_1024D_10M,
        optimize_timeout=config.OPTIMIZE_TIMEOUT_1024D_10M,
    ),
    _hdf5_manager("yahoo-minilm-384-normalized", "id", "Text", 677_305, 384, "normalized"),
    _hdf5_manager(
        "hotpotqa-harrier-640-normalized",
        "ood",
        "Text",
        5_233_329,
        640,
        "normalized",
        load_timeout=config.LOAD_TIMEOUT_768D_10M,
        optimize_timeout=config.OPTIMIZE_TIMEOUT_768D_10M,
    ),
    _hdf5_manager("imagenet-align-640-normalized", "ood", "Text-to-Image", 1_281_167, 640, "normalized"),
    _hdf5_manager("laion-clip-512-normalized", "ood", "Text-to-Image", 1_000_448, 512, "normalized"),
    _hdf5_manager("yandex-200-cosine", "ood", "Text-to-Image", 1_000_000, 200, "cosine"),
    _hdf5_manager("cqadupstack-lemur-2048-ip", "ood", "Multi-vector encoding", 457_149, 2048, "ip"),
    _hdf5_manager("cqadupstack-muvera-5120-ip", "ood", "Multi-vector encoding", 457_149, 5120, "ip"),
    _hdf5_manager("yi-128-ip", "ood", "Attention", 187_843, 128, "ip"),
    _hdf5_manager("llama-128-ip", "ood", "Attention", 256_921, 128, "ip"),
    _hdf5_manager("ccnews-nomic-768-normalized", "id", "Text", 495_328, 768, "normalized"),
    _hdf5_manager("celeba-resnet-2048-cosine", "id", "Image", 201_599, 2048, "cosine"),
    _hdf5_manager("coco-nomic-768-normalized", "ood", "Text-to-Image", 282_360, 768, "normalized"),
    _hdf5_manager("codesearchnet-jina-768-cosine", "id", "Code", 1_374_067, 768, "cosine"),
    _hdf5_manager("simplewiki-openai-3072-normalized", "id", "Text", 260_372, 3072, "normalized"),
)

REGISTERED_DATASETS: dict[str, DatasetManager] = {
    **{dataset_type.value: manager for dataset_type, manager in DatasetWithSizeMap.items()},
    **{manager.data.name: manager for manager in _HDF5_DATASETS},
    **{manager.data.name: manager for manager in _PARQUET_DATASETS},
}


def get_dataset_manager(name: str) -> DatasetManager:
    try:
        return REGISTERED_DATASETS[name].model_copy(deep=True)
    except KeyError as exc:
        supported = ", ".join(REGISTERED_DATASETS)
        msg = f"Unknown dataset {name!r}; supported datasets: {supported}"
        raise ValueError(msg) from exc


def get_registered_datasets(*, family: str | None = None) -> list[DatasetManager]:
    return [
        manager.model_copy(deep=True)
        for manager in REGISTERED_DATASETS.values()
        if (family is None or getattr(manager.data, "family", None) == family)
    ]


# FTS Dataset Translator Pattern
@dataclass
class FtsQuery:
    """Internal representation of an FTS query."""

    query_id: str
    text: str


@dataclass
class FtsDocument:
    """Internal representation of an FTS document."""

    doc_id: str
    text: str
    filter_id: int | None = None


_FTS_FILTER_GOLDEN_RATIO_64 = 0x9E3779B97F4A7C15
_FTS_FILTER_OFFSET_SEED = 0xD1B54A32D192ED03


@dataclass(frozen=True)
class FtsFilterIdPermutation:
    """Deterministic bijection that scatters FTS filter IDs across corpus order."""

    size: int
    multiplier: int
    offset: int

    @property
    def algorithm(self) -> str:
        return "affine_permutation_v1"

    @classmethod
    def for_size(cls, size: int) -> "FtsFilterIdPermutation":
        if size <= 0:
            msg = f"FTS filter ID permutation size must be positive, got {size}"
            raise ValueError(msg)
        if size == 1:
            return cls(size=1, multiplier=1, offset=0)

        multiplier = max(1, (size * _FTS_FILTER_GOLDEN_RATIO_64) >> 64)
        while math.gcd(multiplier, size) != 1:
            multiplier += 1
            if multiplier >= size:
                multiplier = 1

        return cls(
            size=size,
            multiplier=multiplier,
            offset=_FTS_FILTER_OFFSET_SEED % size,
        )

    def map(self, ordinal: int) -> int:
        if ordinal < 0 or ordinal >= self.size:
            msg = f"FTS filter ID ordinal must be in [0, {self.size}), got {ordinal}"
            raise ValueError(msg)
        return (self.multiplier * ordinal + self.offset) % self.size


class FtsDatasetTranslator(ABC):
    """Abstract base class for converting ir_datasets schema to internal format.

    This translator pattern allows easy extension to support new datasets
    (BEIR, TREC, etc.) without modifying core code.
    """

    @property
    @abstractmethod
    def ir_datasets_name(self) -> str:
        """Return the ir_datasets dataset name.

        Example: 'msmarco-passage/dev/small'
        """

    @abstractmethod
    def translate_query(self, ir_query: typing.Any) -> FtsQuery:
        """Convert ir_datasets query to internal FtsQuery format."""

    @abstractmethod
    def translate_document(self, ir_doc: typing.Any) -> FtsDocument:
        """Convert ir_datasets document to internal FtsDocument format."""

    def load(self) -> typing.Any:
        """Load ir_datasets dataset."""
        return ir_datasets.load(self.ir_datasets_name)

    def iter_queries(self, dataset: typing.Any) -> Iterator[FtsQuery]:
        """Iterate over queries in the dataset."""
        for q in dataset.queries_iter():
            yield self.translate_query(q)

    def iter_documents(self, dataset: typing.Any) -> Iterator[FtsDocument]:
        """Iterate over documents in the dataset."""
        for doc in dataset.docs_iter():
            yield self.translate_document(doc)

    def load_ground_truth(self, dataset: typing.Any) -> dict[str, dict[str, int]]:
        """Load positive semantic qrels keyed by query id.

        ir_datasets qrels may contain non-positive judgments. Those are not
        relevant documents for recall/MRR/NDCG, so they are ignored here.
        """
        qrels: dict[str, dict[str, int]] = {}
        for qrel in dataset.qrels_iter():
            relevance = int(getattr(qrel, "relevance", 0))
            if relevance <= 0:
                continue
            query_id = str(qrel.query_id)
            doc_id = str(qrel.doc_id)
            qrels.setdefault(query_id, {})[doc_id] = max(
                relevance,
                qrels.get(query_id, {}).get(doc_id, 0),
            )
        return qrels


class MSMarcoTranslator(FtsDatasetTranslator):
    """Translator for MS MARCO passage retrieval dataset."""

    @property
    def ir_datasets_name(self) -> str:
        return "msmarco-passage/dev/small"

    def translate_query(self, ir_query: typing.Any) -> FtsQuery:
        return FtsQuery(query_id=str(ir_query.query_id), text=ir_query.text)

    def translate_document(self, ir_doc: typing.Any) -> FtsDocument:
        clean_text = ir_doc.text.replace("\t", " ").replace("\n", " ")
        return FtsDocument(doc_id=str(ir_doc.doc_id), text=clean_text)


class HotpotQATranslator(FtsDatasetTranslator):
    """Translator for BEIR HotpotQA."""

    @property
    def ir_datasets_name(self) -> str:
        return "beir/hotpotqa/test"

    def translate_query(self, ir_query: typing.Any) -> FtsQuery:
        return FtsQuery(query_id=str(ir_query.query_id), text=ir_query.text)

    def translate_document(self, ir_doc: typing.Any) -> FtsDocument:
        title = getattr(ir_doc, "title", "") or ""
        text = getattr(ir_doc, "text", "") or ""
        clean_text = f"{title} {text}".replace("\t", " ").replace("\n", " ").strip()
        return FtsDocument(doc_id=str(ir_doc.doc_id), text=clean_text)


class FtsBaseDataset(BaseModel):
    """Base class for FTS datasets - completely independent from BaseDataset.

    FTS datasets are text-based and use TSV files instead of parquet files.
    They don't have vector dimensions; native full-text search uses BM25.

    """

    name: str
    size: int
    metric_type: MetricType = MetricType.BM25
    with_gt: bool = True
    with_remote_resource: bool = False
    gt_neighbors_field: str = "neighbors_id"

    _size_label: ClassVar[dict[int, SizeLabel]]

    @field_validator("size")
    @classmethod
    def verify_size(cls, v: int):
        if v not in cls._size_label:
            msg = f"Size {v} not supported for the FTS dataset, expected: {cls._size_label.keys()}"
            raise ValueError(msg)
        return v

    @property
    def label(self) -> str:
        """Get size label (SMALL, MEDIUM, LARGE, etc.)"""
        return self._size_label.get(self.size).label

    @property
    def full_name(self) -> str:
        return f"{self.name} FTS ({self.label})"

    @property
    def dir_name(self) -> str:
        return f"{self.name}_{self.label}_{utils.numerize(self.size)}".lower()


class MSMarcoFts(FtsBaseDataset):
    name: str = "MS MARCO"
    with_gt: bool = True
    with_remote_resource: bool = False

    _size_label: ClassVar[dict[int, SizeLabel]] = {
        100_000: SizeLabel(100_000, "SMALL", 1),
        1_000_000: SizeLabel(1_000_000, "MEDIUM", 1),
        8_841_823: SizeLabel(8_841_823, "LARGE", 1),
    }

    @property
    def dir_name(self) -> str:
        return f"msmarco_{self.label}_{utils.numerize(self.size)}".lower()


class HotpotQAFts(FtsBaseDataset):
    name: str = "HotpotQA"
    with_gt: bool = True
    with_remote_resource: bool = False

    _size_label: ClassVar[dict[int, SizeLabel]] = {
        100_000: SizeLabel(100_000, "SMALL", 1),
        1_000_000: SizeLabel(1_000_000, "MEDIUM", 1),
        5_233_329: SizeLabel(5_233_329, "LARGE", 1),
    }


class FtsDatasetManager(BaseModel):
    """Manager for FTS datasets - independent from DatasetManager.

    Handles FTS dataset preparation using Translator pattern for extensibility.

    Similar to DatasetManager, but for text-based FTS datasets:
    - queries_data: loaded queries (similar to test_data in vectors)
    - gt_data: loaded ground truth (similar to gt_data in vectors)
    - recall_queries_data: recall-valid queries after optional FTS filter
    - recall_gt_data: recall-valid ground truth after optional FTS filter
    - translator: dataset-specific translator for schema conversion
    - _ir_dataset: ir_datasets dataset object for direct access
    """

    data: FtsBaseDataset
    _translator: typing.Any = PrivateAttr()

    queries_data: list[FtsQuery] | None = None
    gt_data: list[dict[str, int]] | None = None
    recall_queries_data: list[FtsQuery] | None = None
    recall_gt_data: list[dict[str, int]] | None = None
    recall_skipped: bool = False
    recall_skip_reason: str | None = None
    qrels_data: dict[str, dict[str, int]] = PydanticField(default_factory=dict)
    required_doc_ids: set[str] = PydanticField(default_factory=set)
    selected_doc_ids: set[str] | None = None
    qrel_filter_ids: dict[str, int] = PydanticField(default_factory=dict)
    filter_stats: dict[str, int | float | str] = PydanticField(default_factory=dict)
    _ir_dataset: typing.Any = PrivateAttr(default=None)
    _prepared_documents_dir: typing.Any = PrivateAttr(default=None)
    _prepared_documents_path: pathlib.Path | None = PrivateAttr(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize translator based on dataset name
        if isinstance(self.data, MSMarcoFts):
            self._translator = MSMarcoTranslator()
        elif isinstance(self.data, HotpotQAFts):
            self._translator = HotpotQATranslator()
        else:
            msg = f"No translator available for dataset: {self.data.name}"
            raise TypeError(msg)

    def __eq__(self, obj: any):
        if isinstance(obj, FtsDatasetManager):
            return self.data.name == obj.data.name and self.data.size == obj.data.size
        return False

    def __hash__(self) -> int:
        return hash((self.data.name, self.data.size))

    @property
    def preferred_source(self) -> DatasetSource:
        return DatasetSource.IR_DATASETS

    @property
    def data_dir(self) -> pathlib.Path:
        """Get local data directory for this FTS dataset, following vector dataset structure"""
        return pathlib.Path(
            config.DATASET_LOCAL_DIR,
            self.data.name.lower(),
            self.data.dir_name,
        )

    def _validate_cap(self, required_doc_ids: set[str], target_size: int) -> None:
        if len(required_doc_ids) > target_size:
            msg = (
                f"{self.data.full_name} size={target_size} is too small for semantic qrels; "
                f"requires {len(required_doc_ids)} qrel documents"
            )
            raise ValueError(msg)

    def _build_selected_doc_ids(self) -> set[str]:
        """Select the capped corpus while preserving every positive qrel doc."""
        if self._ir_dataset is None:
            msg = "ir_datasets dataset not loaded. Call prepare() first."
            raise RuntimeError(msg)

        required_doc_ids = set(self.required_doc_ids)
        self._validate_cap(required_doc_ids=required_doc_ids, target_size=self.data.size)

        selected_doc_ids = set(required_doc_ids)
        found_required_doc_ids: set[str] = set()
        for doc in self._translator.iter_documents(self._ir_dataset):
            doc_id = str(doc.doc_id)
            if doc_id in required_doc_ids:
                found_required_doc_ids.add(doc_id)

            if doc_id not in selected_doc_ids and len(selected_doc_ids) < self.data.size:
                selected_doc_ids.add(doc_id)

            if len(selected_doc_ids) >= self.data.size and found_required_doc_ids == required_doc_ids:
                break

        missing_doc_ids = required_doc_ids - found_required_doc_ids
        if missing_doc_ids:
            preview = ", ".join(sorted(missing_doc_ids)[:10])
            msg = (
                f"{self.data.full_name} semantic qrel docs missing from corpus: {preview}"
                f"{'...' if len(missing_doc_ids) > 10 else ''}"
            )
            raise ValueError(msg)

        return selected_doc_ids

    def _prepare_qrel_preserving_documents(self) -> None:
        """Materialize capped documents before timed insertion."""
        if self._prepared_documents_dir is not None:
            self._prepared_documents_dir.cleanup()
        self._prepared_documents_dir = None
        self._prepared_documents_path = None

        if self.data.size == max(self.data._size_label):
            return

        required_doc_ids = set(self.required_doc_ids)
        self._validate_cap(required_doc_ids=required_doc_ids, target_size=self.data.size)
        filler_limit = self.data.size - len(required_doc_ids)
        filler_count = 0
        selected_doc_ids: set[str] = set()
        found_required_doc_ids: set[str] = set()
        prepared_dir = tempfile.TemporaryDirectory(prefix="vdbbench_fts_qrel_v1_")
        prepared_path = pathlib.Path(prepared_dir.name, f"{self.data.dir_name}.jsonl")

        try:
            with prepared_path.open("w", encoding="utf-8") as output:
                for doc in self._translator.iter_documents(self._ir_dataset):
                    doc_id = str(doc.doc_id)
                    if doc_id in selected_doc_ids:
                        continue
                    if doc_id in required_doc_ids:
                        found_required_doc_ids.add(doc_id)
                    elif filler_count < filler_limit:
                        filler_count += 1
                    else:
                        continue
                    selected_doc_ids.add(doc_id)
                    output.write(json.dumps([doc_id, doc.text], ensure_ascii=False) + "\n")
                    if len(selected_doc_ids) == self.data.size and found_required_doc_ids == required_doc_ids:
                        break

            missing_doc_ids = required_doc_ids - found_required_doc_ids
            if missing_doc_ids:
                preview = ", ".join(sorted(missing_doc_ids)[:10])
                msg = (
                    f"{self.data.full_name} semantic qrel docs missing from corpus: {preview}"
                    f"{'...' if len(missing_doc_ids) > 10 else ''}"
                )
                raise ValueError(msg)  # noqa: TRY301
            if len(selected_doc_ids) != self.data.size:
                msg = f"{self.data.full_name} prepared {len(selected_doc_ids)} documents, expected {self.data.size}"
                raise ValueError(msg)  # noqa: TRY301
        except Exception:
            prepared_dir.cleanup()
            raise

        self._prepared_documents_dir = prepared_dir
        self._prepared_documents_path = prepared_path

    def _iter_prepared_documents(self) -> Iterator[FtsDocument]:
        if self._prepared_documents_path is None:
            yield from self._translator.iter_documents(self._ir_dataset)
            return
        with self._prepared_documents_path.open(encoding="utf-8") as prepared:
            for line in prepared:
                doc_id, text = json.loads(line)
                yield FtsDocument(doc_id=doc_id, text=text)

    def _iter_selected_documents_with_filter_ids(self, include_filter_ids: bool = False) -> Iterator[FtsDocument]:
        """Yield selected documents with the exact filter IDs used for insertion and qrels."""
        if self._ir_dataset is None:
            msg = "ir_datasets dataset not loaded. Call prepare() first."
            raise RuntimeError(msg)

        permutation = FtsFilterIdPermutation.for_size(self.data.size) if include_filter_ids else None
        documents = iter(self._iter_prepared_documents())
        emitted_count = 0
        while emitted_count < self.data.size:
            try:
                doc = next(documents)
                doc.doc_id = str(doc.doc_id)
                if self.selected_doc_ids is not None and doc.doc_id not in self.selected_doc_ids:
                    continue
                if permutation is not None:
                    doc.filter_id = permutation.map(emitted_count)
            except StopIteration:
                break
            except Exception as e:
                log.debug(f"Skipping malformed document: {e}")
                continue

            emitted_count += 1
            yield doc

    def _build_qrel_filter_ids(self) -> dict[str, int]:
        """Map qrel doc IDs to their deterministic permuted FTS filter ID."""
        if self._ir_dataset is None:
            msg = "ir_datasets dataset not loaded. Call prepare() first."
            raise RuntimeError(msg)

        qrel_doc_ids = set(self.required_doc_ids)
        qrel_filter_ids: dict[str, int] = {}
        for doc in self._iter_selected_documents_with_filter_ids(include_filter_ids=True):
            doc_id = doc.doc_id
            if doc_id in qrel_doc_ids:
                qrel_filter_ids[doc_id] = doc.filter_id

        missing_doc_ids = qrel_doc_ids - set(qrel_filter_ids)
        if missing_doc_ids:
            preview = ", ".join(sorted(missing_doc_ids)[:10])
            msg = (
                f"{self.data.full_name} semantic qrel docs missing filter_id assignment: {preview}"
                f"{'...' if len(missing_doc_ids) > 10 else ''}"
            )
            raise ValueError(msg)
        return qrel_filter_ids

    def _apply_integer_filter_to_qrels(
        self,
        queries: list[FtsQuery],
        ground_truth: list[dict[str, int]],
        filters: Filter,
    ) -> tuple[list[FtsQuery], list[dict[str, int]]]:
        filter_field = getattr(filters, "int_field", "filter_id")
        if filter_field != "filter_id":
            msg = f"FTS integer filters require int_field='filter_id', got {filter_field!r}"
            raise ValueError(msg)

        filter_value = int(filters.int_value)
        if filter_value < 0 or filter_value > self.data.size:
            msg = f"FTS filter_id threshold must be in [0, {self.data.size}], got {filter_value}"
            raise ValueError(msg)

        self.qrel_filter_ids = self._build_qrel_filter_ids()
        filtered_queries: list[FtsQuery] = []
        filtered_gt: list[dict[str, int]] = []
        for query, qrels in zip(queries, ground_truth, strict=True):
            filtered_qrels = {
                doc_id: rel for doc_id, rel in qrels.items() if self.qrel_filter_ids.get(doc_id, -1) >= filter_value
            }
            if not filtered_qrels:
                continue
            filtered_queries.append(query)
            filtered_gt.append(filtered_qrels)

        matched_doc_count = self.data.size - filter_value
        filtered_relevant_doc_ids = {doc_id for qrels in filtered_gt for doc_id in qrels}
        permutation = FtsFilterIdPermutation.for_size(self.data.size)
        self.filter_stats = {
            "filter_type": filters.type.value,
            "filter_field": filter_field,
            "filter_value": filter_value,
            "filter_rate": filters.filter_rate,
            "filter_id_distribution": permutation.algorithm,
            "filter_id_multiplier": permutation.multiplier,
            "filter_id_offset": permutation.offset,
            "matched_doc_count": matched_doc_count,
            "matched_doc_ratio": round(matched_doc_count / self.data.size, 6),
            "original_query_count": len(queries),
            "filtered_query_count": len(filtered_queries),
            "filtered_query_ratio": round(len(filtered_queries) / len(queries), 6),
            "original_relevant_doc_count": len(self.required_doc_ids),
            "filtered_relevant_doc_count": len(filtered_relevant_doc_ids),
        }
        log.info(
            "Applied FTS integer filter %s >= %s: queries %s/%s, relevant docs %s/%s",
            filter_field,
            filter_value,
            len(filtered_queries),
            len(queries),
            len(filtered_relevant_doc_ids),
            len(self.required_doc_ids),
        )
        if not filtered_queries:
            self.recall_skipped = True
            self.recall_skip_reason = "no_positive_qrels_after_filter"
        return filtered_queries, filtered_gt

    def _apply_filters_to_qrels(
        self,
        queries: list[FtsQuery],
        ground_truth: list[dict[str, int]],
        filters: Filter | None,
    ) -> tuple[list[FtsQuery], list[dict[str, int]]]:
        self.filter_stats = {}
        self.qrel_filter_ids = {}
        self.recall_skipped = False
        self.recall_skip_reason = None
        if filters is None or filters.type == FilterOp.NonFilter:
            return queries, ground_truth
        if filters.type == FilterOp.NumGE:
            return self._apply_integer_filter_to_qrels(queries, ground_truth, filters)
        msg = f"FTS dataset filtering does not support filter type {filters.type}"
        raise ValueError(msg)

    def prepare(
        self,
        source: DatasetSource | None = None,
        filters: Filter | None = None,
    ) -> bool:
        """Prepare FTS dataset for testing using Translator pattern.

        Directly uses ir_datasets API without generating TSV files:
        1. Downloads dataset using ir_datasets (if needed)
        2. Loads dataset object using translator
        3. Loads queries and semantic qrels from ir_datasets

        Args:
            source: Data source to download from (should be IR_DATASETS for FTS)
            filters: Optional filters. FTS supports natural semantic GT
                filtering for integer filter_id cases.

        Returns:
            bool: True if preparation successful, False otherwise
        """
        log.info(f"Preparing FTS dataset: {self.data.full_name}")

        try:
            # Download dataset if needed (ir_datasets handles caching)
            if source is not None:
                reader = source.reader()
                if reader is not None:
                    dataset_name = self._translator.ir_datasets_name
                    # reader.read() will download the dataset if needed
                    reader.read(dataset_name, [], self.data_dir)

            # Load dataset using translator
            self._ir_dataset = self._translator.load()
            log.info(f"Successfully loaded ir_datasets dataset: {self._translator.ir_datasets_name}")

            # Load queries from ir_datasets and semantic ground truth by query id.
            if self.data.with_gt:
                all_queries = list(self._translator.iter_queries(self._ir_dataset))
                log.info(f"Loaded {len(all_queries)} queries into memory")

                self.qrels_data = self._translator.load_ground_truth(self._ir_dataset)
                self.queries_data = []
                self.gt_data = []
                for query in all_queries:
                    qrels = self.qrels_data.get(query.query_id)
                    if not qrels:
                        continue
                    self.queries_data.append(
                        FtsQuery(
                            query_id=query.query_id,
                            text=query.text,
                        )
                    )
                    self.gt_data.append(qrels)

                if not self.queries_data:
                    msg = f"{self.data.full_name} has no queries with positive semantic qrels"
                    raise ValueError(msg)  # noqa: TRY301

                self.required_doc_ids = {doc_id for qrels in self.gt_data for doc_id in qrels}
                self.selected_doc_ids = None
                self._prepare_qrel_preserving_documents()
                self.recall_queries_data, self.recall_gt_data = self._apply_filters_to_qrels(
                    self.queries_data,
                    self.gt_data,
                    filters,
                )
                log.info(
                    "Loaded semantic qrels for %s queries; recall uses %s queries; "
                    "selected %s corpus docs including %s qrel docs",
                    len(self.gt_data),
                    len(self.recall_gt_data),
                    len(self.selected_doc_ids) if self.selected_doc_ids is not None else self.data.size,
                    len(self.required_doc_ids),
                )
            else:
                self.selected_doc_ids = None
                self.qrel_filter_ids = {}
                self.filter_stats = {}
                self.recall_queries_data = None
                self.recall_gt_data = None
                self.recall_skipped = False
                self.recall_skip_reason = None

        except (TypeError, ValueError):
            log.exception("Invalid FTS dataset configuration")
            raise
        except Exception:
            log.exception("Failed to prepare FTS dataset")
            return False
        else:
            log.debug(f"{self.data.name}: FTS dataset prepared")
            log.info(f"FTS dataset preparation completed: {self.data.full_name}")
            return True

    def iter_batches(self, batch_size: int = DEFAULT_INSERT_BATCH_SIZE):
        """Return an iterator for streaming FTS document batches."""
        return FtsDocumentIterator(self, batch_size=batch_size)

    def __iter__(self):
        """Return iterator for streaming document batches.

        Similar to DatasetManager.__iter__() which returns DataSetIterator.
        This enables batch-by-batch processing of documents without loading
        all documents into memory at once.

        Example:
            >>> manager = FtsDataset.MSMARCO.manager(100_000)
            >>> for batch in manager:
            >>>     print(f"Processing {len(batch)} documents")
        """
        return self.iter_batches()


class FtsDocumentIterator:
    """Iterator for streaming FTS document batches using Translator pattern.

    Similar to DataSetIterator for vector datasets, but reads directly from ir_datasets
    using translator. Yields batches of FtsDocument objects for memory-efficient
    processing of large datasets.
    """

    def __init__(self, dataset: FtsDatasetManager, batch_size: int = DEFAULT_INSERT_BATCH_SIZE):
        if batch_size <= 0:
            msg = f"insert batch size must be greater than 0, got {batch_size}"
            raise ValueError(msg)
        self._ds = dataset
        self._batch_size = batch_size
        self._finished = False
        self._docs_iter = None

    def __iter__(self):
        return self

    def __next__(self) -> list[FtsDocument]:
        """Return the next batch of documents.

        Returns:
            list[FtsDocument]: List of FtsDocument objects

        Raises:
            StopIteration: When all documents have been read
        """
        if self._finished:
            raise StopIteration

        if self._docs_iter is None:
            self._docs_iter = self._ds._iter_selected_documents_with_filter_ids(
                include_filter_ids=bool(self._ds.filter_stats),
            )

        batch = []
        while len(batch) < self._batch_size:
            try:
                batch.append(next(self._docs_iter))
            except StopIteration:
                self._finished = True
                if batch:
                    return batch
                raise
        return batch

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit context manager."""

    def __del__(self):
        """Cleanup when iterator is destroyed."""


class FtsDataset(Enum):
    MSMARCO = MSMarcoFts
    HOTPOTQA = HotpotQAFts

    def get(self, size: int) -> FtsBaseDataset:
        return self.value(size=size)

    def manager(self, size: int) -> FtsDatasetManager:
        return FtsDatasetManager(data=self.get(size))


class FtsDatasetWithSizeType(Enum):
    MSMarcoSmall = "MS MARCO Small (100K documents)"
    MSMarcoMedium = "MS MARCO Medium (1M documents)"
    MSMarcoLarge = "MS MARCO Large (8.8M documents)"
    HotpotQASmall = "HotpotQA Small (100K documents)"
    HotpotQAMedium = "HotpotQA Medium (1M documents)"
    HotpotQALarge = "HotpotQA Large (5.2M documents)"

    def get_manager(self) -> FtsDatasetManager:
        return {
            FtsDatasetWithSizeType.MSMarcoSmall: FtsDataset.MSMARCO.manager(100_000),
            FtsDatasetWithSizeType.MSMarcoMedium: FtsDataset.MSMARCO.manager(1_000_000),
            FtsDatasetWithSizeType.MSMarcoLarge: FtsDataset.MSMARCO.manager(8_841_823),
            FtsDatasetWithSizeType.HotpotQASmall: FtsDataset.HOTPOTQA.manager(100_000),
            FtsDatasetWithSizeType.HotpotQAMedium: FtsDataset.HOTPOTQA.manager(1_000_000),
            FtsDatasetWithSizeType.HotpotQALarge: FtsDataset.HOTPOTQA.manager(5_233_329),
        }[self]

    def get_load_timeout(self) -> float:
        if self in {FtsDatasetWithSizeType.MSMarcoSmall, FtsDatasetWithSizeType.HotpotQASmall}:
            return config.LOAD_TIMEOUT_768D_100K
        return config.LOAD_TIMEOUT_DEFAULT

    def get_optimize_timeout(self) -> float:
        if self in {FtsDatasetWithSizeType.MSMarcoSmall, FtsDatasetWithSizeType.HotpotQASmall}:
            return config.OPTIMIZE_TIMEOUT_768D_100K
        return config.OPTIMIZE_TIMEOUT_DEFAULT

    @property
    def is_advanced(self) -> bool:
        return self in {FtsDatasetWithSizeType.MSMarcoLarge, FtsDatasetWithSizeType.HotpotQALarge}
