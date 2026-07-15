"""
Usage:
    >>> from xxx.dataset import Dataset
    >>> Dataset.Cohere.get(100_000)
"""

import json
import logging
import pathlib
import struct
import types
import typing
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, NamedTuple

import ir_datasets
import pandas as pd
import polars as pl
import s3fs
from pyarrow.parquet import ParquetFile
from pydantic import Field as PydanticField
from pydantic import PrivateAttr, field_validator
from tqdm import tqdm

from vectordb_bench import config
from vectordb_bench.base import BaseModel

from . import utils
from .clients import MetricType
from .data_source import DatasetReader, DatasetSource
from .filter import Filter, FilterOp, non_filter

log = logging.getLogger(__name__)


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


class SIFTBinary(BaseDataset):
    name: str = "SIFTBinary"
    dim: int = 128
    metric_type: MetricType = MetricType.HAMMING
    use_shuffled: bool = False
    with_gt: bool = True
    remote_path: str = "assets.zilliz.com/nightly/v1/siftbinary_1m"
    test_file: str = "query.bin"
    gt_file: str = "truth.ibin"
    _size_label: ClassVar[dict[int, SizeLabel]] = {
        1_000_000: SizeLabel(1_000_000, "1M", 1),
    }

    @property
    def dir_name(self) -> str:
        return "siftbinary_1m"

    @property
    def train_files(self) -> list[str]:
        return ["base.bin"]


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


class DatasetManager(BaseModel):
    """Download dataset if not in the local directory. Provide data for cases.

    DatasetManager is iterable, each iteration will return the next batch of data in pandas.DataFrame

    Examples:
        >>> cohere = Dataset.COHERE.manager(100_000)
        >>> for data in cohere:
        >>>    print(data.columns)
    """

    data: BaseDataset
    test_data: list[list[float]] | None = None
    gt_data: list[list[int]] | None = None
    scalar_labels: pl.DataFrame | None = None
    train_files: list[str] = []
    reader: DatasetReader | None = None

    def __eq__(self, obj: any):
        if isinstance(obj, DatasetManager):
            return self.data.name == obj.data.name and self.data.label == obj.data.label
        return False

    def __hash__(self) -> int:
        return hash((self.data.name, self.data.label))

    def set_reader(self, reader: DatasetReader):
        self.reader = reader

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
        return DataSetIterator(self)

    def iter_batches(self, batch_size: int):
        return DataSetIterator(self, batch_size=batch_size)

    # TODO passing use_shuffle from outside
    def prepare(
        self,
        source: DatasetSource = DatasetSource.S3,
        filters: Filter = non_filter,
        with_train_files: bool = True,
        with_scalar_labels: bool = False,
    ) -> bool:
        """Download the dataset from DatasetSource
         url = f"{source}/{self.data.dir_name}"

        Args:
            source(DatasetSource): S3 or AliyunOSS, default as S3
            filters(Filter): combined with dataset's with_gt to
              compose the correct ground_truth file

        Returns:
            bool: whether the dataset is successfully prepared

        """
        self.train_files = self.data.train_files if with_train_files else []
        gt_file, test_file = None, None
        if self.data.with_gt:
            gt_file, test_file = filters.groundtruth_file, self.data.test_file

        if self.data.with_remote_resource:
            download_files = [file for file in self.train_files]
            download_files.extend([gt_file, test_file])
            if self.data.with_scalar_labels and self.data.scalar_labels_file_separated:
                download_files.append(self.data.scalar_labels_file)
            download_files = [file for file in download_files if file is not None]
            source.reader().read(
                dataset=self.data.dir_name.lower(),
                files=download_files,
                local_ds_root=self.data_dir,
            )

        needs_scalar_labels = filters.type == FilterOp.StrEqual or with_scalar_labels

        # read scalar_labels_file if separated
        if needs_scalar_labels and self.data.with_scalar_labels and self.data.scalar_labels_file_separated:
            self.scalar_labels = self._read_file(self.data.scalar_labels_file)

        if gt_file is not None and test_file is not None:
            self.test_data = self._read_file(test_file)[self.data.test_vector_field].to_list()
            self.gt_data = self._read_file(gt_file)[self.data.gt_neighbors_field].to_list()

        log.debug(f"{self.data.name}: available train files {self.train_files}")

        return True

    def _read_file(self, file_name: str) -> pl.DataFrame:
        """read one file from disk into memory"""
        log.info(f"Read the entire file into memory: {file_name}")
        p = pathlib.Path(self.data_dir, file_name)
        if not p.exists():
            log.warning(f"No such file: {p}")
            return pl.DataFrame()

        return pl.read_parquet(p)


class DataSetIterator:
    def __init__(self, dataset: DatasetManager, batch_size: int = config.NUM_PER_BATCH):
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
        p = pathlib.Path(self._ds.data_dir, file_name)
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


class BinaryDatasetManager(DatasetManager):
    """Dataset manager for vecTool-style packed binary vector files."""

    data: SIFTBinary

    def __iter__(self):
        return BinaryDataSetIterator(self)

    def iter_batches(self, batch_size: int):
        return BinaryDataSetIterator(self, batch_size=batch_size)

    def prepare(
        self,
        source: DatasetSource = DatasetSource.S3,
        filters: Filter = non_filter,
        with_train_files: bool = True,
        with_scalar_labels: bool = False,
    ) -> bool:
        if source != DatasetSource.S3:
            msg = f"{self.data.name} is currently hosted under assets.zilliz.com/nightly and supports S3 only"
            raise ValueError(msg)

        self.train_files = self.data.train_files if with_train_files else []
        download_files = [*self.train_files, self.data.test_file, self.data.gt_file, "manifest.json"]
        self._download_files(download_files)

        self.test_data = self._read_binary_vectors(self.data.test_file)
        self.gt_data = self._read_truth_ids(self.data.gt_file)
        log.debug(f"{self.data.name}: available train files {self.train_files}")
        return True

    def _download_files(self, files: list[str]) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        fs = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": "us-west-2"})
        downloads = []
        for file in files:
            remote_file = pathlib.PurePosixPath(self.data.remote_path, file)
            local_file = self.data_dir.joinpath(file)
            if not local_file.exists():
                downloads.append((remote_file, local_file))
                continue

            remote_size = fs.info(remote_file.as_posix()).get("size")
            if remote_size != local_file.stat().st_size:
                downloads.append((remote_file, local_file))

        if not downloads:
            return

        log.info(f"Start downloading binary dataset files, total count: {len(downloads)}")
        for remote_file, local_file in tqdm(downloads):
            fs.download(remote_file.as_posix(), local_file.as_posix())

    def _read_binary_header(self, file_name: str) -> tuple[int, int]:
        path = self.data_dir.joinpath(file_name)
        with path.open("rb") as fp:
            header = fp.read(8)
        if len(header) != 8:
            msg = f"Invalid binary vector file header: {path}"
            raise ValueError(msg)
        rows, dim = struct.unpack("<II", header)
        if dim != self.data.dim:
            msg = f"Unexpected {file_name} dim={dim}; expected {self.data.dim}"
            raise ValueError(msg)
        return rows, dim

    def _read_binary_vectors(self, file_name: str) -> list[str]:
        path = self.data_dir.joinpath(file_name)
        rows, dim = self._read_binary_header(file_name)
        bytes_per_vector = dim // 8
        with path.open("rb") as fp:
            fp.seek(8)
            raw = fp.read()
        expected = rows * bytes_per_vector
        if len(raw) != expected:
            msg = f"Unexpected {file_name} payload size={len(raw)}; expected {expected}"
            raise ValueError(msg)
        return [raw[i : i + bytes_per_vector].hex() for i in range(0, len(raw), bytes_per_vector)]

    def _read_truth_ids(self, file_name: str) -> list[list[int]]:
        path = self.data_dir.joinpath(file_name)
        with path.open("rb") as fp:
            header = fp.read(8)
            raw = fp.read()
        if len(header) != 8:
            msg = f"Invalid truth file header: {path}"
            raise ValueError(msg)
        nq, topk = struct.unpack("<II", header)
        expected = nq * topk * 4
        if len(raw) != expected:
            msg = f"Unexpected {file_name} payload size={len(raw)}; expected {expected}"
            raise ValueError(msg)
        ids = [value[0] for value in struct.iter_unpack("<i", raw)]
        return [ids[i : i + topk] for i in range(0, len(ids), topk)]


class BinaryDataSetIterator:
    def __init__(self, dataset: BinaryDatasetManager, batch_size: int = config.NUM_PER_BATCH):
        self._ds = dataset
        self._batch_size = batch_size
        self._idx = 0
        self._rows, self._dim = self._ds._read_binary_header("base.bin")
        self._bytes_per_vector = self._dim // 8
        self._fp = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fp"] = None
        return state

    def __setstate__(self, state: Any):
        self.__dict__.update(state)

    def __iter__(self):
        return self

    def __next__(self) -> pd.DataFrame:
        if self._idx >= self._rows:
            if self._fp is not None:
                self._fp.close()
                self._fp = None
            raise StopIteration

        if self._fp is None:
            path = self._ds.data_dir.joinpath("base.bin")
            self._fp = path.open("rb")
            self._fp.seek(8 + self._idx * self._bytes_per_vector)

        batch_rows = min(self._batch_size, self._rows - self._idx)
        raw = self._fp.read(batch_rows * self._bytes_per_vector)
        if len(raw) != batch_rows * self._bytes_per_vector:
            msg = f"Unexpected EOF while reading {self._ds.data.name} at row {self._idx}"
            raise ValueError(msg)

        start = self._idx
        self._idx += batch_rows
        vectors = [raw[i : i + self._bytes_per_vector].hex() for i in range(0, len(raw), self._bytes_per_vector)]
        return pd.DataFrame(
            {
                self._ds.data.train_id_field: range(start, start + batch_rows),
                self._ds.data.train_vector_field: vectors,
            }
        )


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
    SIFT_BINARY = SIFTBinary
    OPENAI = OpenAI

    def get(self, size: int) -> BaseDataset:
        return self.value(size=size)

    def manager(self, size: int) -> DatasetManager:
        data = self.get(size)
        if isinstance(data, SIFTBinary):
            return BinaryDatasetManager(data=data)
        return DatasetManager(data=data)


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
    SIFTBinary1M = "Medium SIFT Binary (128bit, 1M)"

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
    DatasetWithSizeType.CohereMedium: Dataset.COHERE.manager(1_000_000),
    DatasetWithSizeType.CohereLarge: Dataset.COHERE.manager(10_000_000),
    DatasetWithSizeType.LAIONLarge: Dataset.LAION.manager(100_000_000),
    DatasetWithSizeType.BioasqMedium: Dataset.BIOASQ.manager(1_000_000),
    DatasetWithSizeType.BioasqLarge: Dataset.BIOASQ.manager(10_000_000),
    DatasetWithSizeType.OpenAISmall: Dataset.OPENAI.manager(50_000),
    DatasetWithSizeType.OpenAIMedium: Dataset.OPENAI.manager(500_000),
    DatasetWithSizeType.OpenAILarge: Dataset.OPENAI.manager(5_000_000),
    DatasetWithSizeType.SIFTBinary1M: Dataset.SIFT_BINARY.manager(1_000_000),
}


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


FTS_GT_FILE = "neighbors.parquet"
FTS_BUILD_MANIFEST_FILE = "build_manifest.json"
FTS_MATH_GT_FILES = (FTS_GT_FILE, FTS_BUILD_MANIFEST_FILE, "manifest.json")


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
    - translator: dataset-specific translator for schema conversion
    - _ir_dataset: ir_datasets dataset object for direct access
    """

    data: FtsBaseDataset
    _translator: typing.Any = PrivateAttr()

    queries_data: list[FtsQuery] | None = None
    gt_data: list[list[str]] | None = None
    bm25_params: dict[str, float] = PydanticField(default_factory=dict)
    analyzer_params: dict[str, typing.Any] = PydanticField(default_factory=dict)
    _ir_dataset: typing.Any = PrivateAttr(default=None)

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
    def data_dir(self) -> pathlib.Path:
        """Get local data directory for this FTS dataset, following vector dataset structure"""
        return pathlib.Path(
            config.DATASET_LOCAL_DIR,
            self.data.name.lower(),
            self.data.dir_name,
        )

    def _download_math_gt_files(self) -> None:
        DatasetSource.S3.reader().read(
            dataset=self.data.dir_name.lower(),
            files=list(FTS_MATH_GT_FILES),
            local_ds_root=self.data_dir,
        )

    def _load_math_gt_data(self) -> list[list[str]]:
        p = pathlib.Path(self.data_dir, FTS_GT_FILE)
        if not p.exists():
            msg = f"No such file: {p}"
            raise FileNotFoundError(msg)
        gt_rows = pl.read_parquet(p)[self.data.gt_neighbors_field].to_list()
        # FTS math GT stores dense document row IDs, not original ir_datasets doc IDs.
        # FtsDocumentIterator assigns these same row IDs during insertion.
        return [[str(doc_id) for doc_id in row if str(doc_id) != "-1"] for row in gt_rows]

    def _load_build_manifest(self) -> dict[str, typing.Any]:
        p = pathlib.Path(self.data_dir, FTS_BUILD_MANIFEST_FILE)
        if not p.exists():
            msg = f"No such file: {p}"
            raise FileNotFoundError(msg)
        manifest = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            msg = f"Invalid FTS build manifest: {p}"
            raise TypeError(msg)
        return manifest

    def _validate_build_manifest(self, manifest: dict[str, typing.Any]) -> None:
        source_ir_dataset = manifest.get("source_ir_dataset")
        if source_ir_dataset is not None and source_ir_dataset != self._translator.ir_datasets_name:
            msg = (
                f"{self.data.full_name} manifest source_ir_dataset={source_ir_dataset!r} "
                f"does not match {self._translator.ir_datasets_name!r}"
            )
            raise ValueError(msg)

        for field_name in ("doc_limit", "indexed_doc_count"):
            value = manifest.get(field_name)
            if value is None:
                continue
            if int(value) != self.data.size:
                msg = f"{self.data.full_name} manifest {field_name}={value} does not match size={self.data.size}"
                raise ValueError(msg)

        query_count = manifest.get("query_count")
        if query_count is not None and self.queries_data is not None and int(query_count) != len(self.queries_data):
            msg = (
                f"{self.data.full_name} manifest query_count={query_count} "
                f"does not match loaded query count={len(self.queries_data)}"
            )
            raise ValueError(msg)

    def _load_manifest_params(self) -> None:
        manifest = self._load_build_manifest()
        self._validate_build_manifest(manifest)
        bm25 = manifest.get("bm25") or {}
        analyzer = manifest.get("analyzer") or {}
        self.bm25_params = {
            key: float(bm25[key]) for key in ("k1", "b", "avgdl") if key in bm25 and bm25[key] is not None
        }
        self.analyzer_params = analyzer if isinstance(analyzer, dict) else {}

    def prepare(
        self,
        source: DatasetSource | None = None,
        filters: Filter | None = None,
    ) -> bool:
        """Prepare FTS dataset for testing using Translator pattern.

        Directly uses ir_datasets API without generating TSV files:
        1. Downloads dataset using ir_datasets (if needed)
        2. Loads dataset object using translator
        3. Loads queries from ir_datasets and mathematical ground truth from S3

        Args:
            source: Data source to download from (should be IR_DATASETS for FTS)
            filters: Optional filters (not used for FTS)

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

            # Force ir_datasets lazy document cache work before timed insert.
            for idx, _ in enumerate(self._translator.iter_documents(self._ir_dataset), start=1):
                if idx >= self.data.size:
                    break

            # Load queries from ir_datasets and mathematical ground truth artifacts by row order.
            if self.data.with_gt:
                # Load queries using translator
                self.queries_data = list(self._translator.iter_queries(self._ir_dataset))
                log.info(f"Loaded {len(self.queries_data)} queries into memory")

                self._download_math_gt_files()
                self._load_manifest_params()
                self.gt_data = self._load_math_gt_data()
                if len(self.queries_data) != len(self.gt_data):
                    msg = (
                        f"{self.data.full_name} query count {len(self.queries_data)} "
                        f"does not match ground truth row count {len(self.gt_data)}"
                    )
                    raise ValueError(msg)  # noqa: TRY301
                log.info(f"Loaded mathematical ground truth for {len(self.gt_data)} queries into memory")

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

    def iter_batches(self, batch_size: int = config.NUM_PER_BATCH):
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

    def __init__(self, dataset: FtsDatasetManager, batch_size: int = config.NUM_PER_BATCH):
        self._ds = dataset
        self._batch_size = batch_size
        self._finished = False
        self._doc_count = 0  # Track total documents processed
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

        # Initialize iterator on first call
        if self._docs_iter is None:
            if self._ds._ir_dataset is None:
                error_msg = "ir_datasets dataset not loaded. Call prepare() first."
                log.error(error_msg)
                raise RuntimeError(error_msg)

            log.info("Starting to iterate documents using translator")
            self._docs_iter = self._ds._translator.iter_documents(self._ds._ir_dataset)

        # Read batch with proper error handling
        try:
            batch = []
            for _ in range(self._batch_size):
                if self._doc_count >= self._ds.data.size:
                    self._finished = True
                    if batch:
                        return batch
                    raise StopIteration  # noqa: TRY301
                try:
                    doc = next(self._docs_iter)
                    doc.doc_id = str(self._doc_count)
                    batch.append(doc)
                    self._doc_count += 1
                except StopIteration:
                    self._finished = True
                    if batch:
                        return batch
                    raise
                except Exception as e:
                    log.debug(f"Skipping malformed document: {e}")
                    continue

        except StopIteration:
            self._finished = True
            raise
        except Exception:
            log.exception("Error reading documents from translator")
            raise
        else:
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
