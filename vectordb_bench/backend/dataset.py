"""
Usage:
    >>> from xxx.dataset import Dataset
    >>> Dataset.Cohere.get(100_000)
"""

import logging
import pathlib
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
from pyarrow.parquet import ParquetFile
from pydantic import Field as PydanticField
from pydantic import PrivateAttr, field_validator

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

    def manager(self, size: int) -> DatasetManager:
        return DatasetManager(data=self.get(size))


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
    DatasetWithSizeType.CohereMedium: Dataset.COHERE.manager(1_000_000),
    DatasetWithSizeType.CohereLarge: Dataset.COHERE.manager(10_000_000),
    DatasetWithSizeType.LAIONLarge: Dataset.LAION.manager(100_000_000),
    DatasetWithSizeType.BioasqMedium: Dataset.BIOASQ.manager(1_000_000),
    DatasetWithSizeType.BioasqLarge: Dataset.BIOASQ.manager(10_000_000),
    DatasetWithSizeType.OpenAISmall: Dataset.OPENAI.manager(50_000),
    DatasetWithSizeType.OpenAIMedium: Dataset.OPENAI.manager(500_000),
    DatasetWithSizeType.OpenAILarge: Dataset.OPENAI.manager(5_000_000),
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
    - translator: dataset-specific translator for schema conversion
    - _ir_dataset: ir_datasets dataset object for direct access
    """

    data: FtsBaseDataset
    _translator: typing.Any = PrivateAttr()

    queries_data: list[FtsQuery] | None = None
    gt_data: list[dict[str, int]] | None = None
    qrels_data: dict[str, dict[str, int]] = PydanticField(default_factory=dict)
    required_doc_ids: set[str] = PydanticField(default_factory=set)
    selected_doc_ids: set[str] | None = None
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
                self.selected_doc_ids = self._build_selected_doc_ids()
                log.info(
                    "Loaded semantic qrels for %s queries; selected %s corpus docs including %s qrel docs",
                    len(self.gt_data),
                    len(self.selected_doc_ids),
                    len(self.required_doc_ids),
                )
            else:
                self.selected_doc_ids = None

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
        self._doc_count = 0  # Track total documents emitted
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
            while len(batch) < self._batch_size:
                if self._doc_count >= self._ds.data.size:
                    self._finished = True
                    if batch:
                        return batch
                    raise StopIteration  # noqa: TRY301
                try:
                    doc = next(self._docs_iter)
                    doc.doc_id = str(doc.doc_id)
                    if self._ds.selected_doc_ids is not None and doc.doc_id not in self._ds.selected_doc_ids:
                        continue
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
