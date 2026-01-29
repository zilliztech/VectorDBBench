"""
Usage:
    >>> from xxx.dataset import Dataset
    >>> Dataset.Cohere.get(100_000)
"""

import logging
import pathlib
import types
import typing
from enum import Enum

import pandas as pd
import polars as pl
from pyarrow.parquet import ParquetFile
from pydantic import PrivateAttr, validator

from vectordb_bench import config
from vectordb_bench.base import BaseModel

from . import utils
from .clients import MetricType
from .data_source import DatasetReader, DatasetSource
from .filter import Filter, FilterOp, non_filter

log = logging.getLogger(__name__)


class SizeLabel(typing.NamedTuple):
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
    _size_label: dict[int, SizeLabel] = PrivateAttr()
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

    @validator("size", allow_reuse=True)
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

    @validator("size", allow_reuse=True)
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
    _size_label: dict = {
        100_000_000: SizeLabel(100_000_000, "LARGE", 100),
    }


class GIST(BaseDataset):
    name: str = "GIST"
    dim: int = 960
    metric_type: MetricType = MetricType.L2
    use_shuffled: bool = False
    _size_label: dict = {
        100_000: SizeLabel(100_000, "SMALL", 1),
        1_000_000: SizeLabel(1_000_000, "MEDIUM", 1),
    }


class Cohere(BaseDataset):
    name: str = "Cohere"
    dim: int = 768
    metric_type: MetricType = MetricType.COSINE
    use_shuffled: bool = config.USE_SHUFFLED_DATA
    with_gt: bool = True
    _size_label: dict = {
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
    _size_label: dict = {
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
    _size_label: dict = {1_000_000: SizeLabel(1_000_000, "MEDIUM", 1)}


class SIFT(BaseDataset):
    name: str = "SIFT"
    dim: int = 128
    metric_type: MetricType = MetricType.L2
    use_shuffled: bool = False
    _size_label: dict = {
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
    _size_label: dict = {
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

    # TODO passing use_shuffle from outside
    def prepare(
        self,
        source: DatasetSource = DatasetSource.S3,
        filters: Filter = non_filter,
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
        self.train_files = self.data.train_files
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

        # read scalar_labels_file if separated
        if (
            filters.type == FilterOp.StrEqual
            and self.data.with_scalar_labels
            and self.data.scalar_labels_file_separated
        ):
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
    def __init__(self, dataset: DatasetManager):
        self._ds = dataset
        self._idx = 0  # file number
        self._cur = None
        self._sub_idx = [0 for i in range(len(self._ds.train_files))]  # iter num for each file

    def __iter__(self):
        return self

    def _get_iter(self, file_name: str):
        p = pathlib.Path(self._ds.data_dir, file_name)
        log.info(f"Get iterator for {p.name}")
        if not p.exists():
            msg = f"No such file: {p}"
            log.warning(msg)
            raise IndexError(msg)
        return ParquetFile(p, memory_map=True, pre_buffer=True).iter_batches(config.NUM_PER_BATCH)

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
        if "small" in self.value.lower():
            return config.LOAD_TIMEOUT_768D_100K
        if "medium" in self.value.lower():
            return config.LOAD_TIMEOUT_768D_1M
        if "large" in self.value.lower():
            return config.LOAD_TIMEOUT_768D_10M
        msg = f"No load_timeout for {self.value}"
        raise KeyError(msg)

    def get_optimize_timeout(self) -> float:
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
    DatasetWithSizeType.BioasqMedium: Dataset.BIOASQ.manager(1_000_000),
    DatasetWithSizeType.BioasqLarge: Dataset.BIOASQ.manager(10_000_000),
    DatasetWithSizeType.OpenAISmall: Dataset.OPENAI.manager(50_000),
    DatasetWithSizeType.OpenAIMedium: Dataset.OPENAI.manager(500_000),
    DatasetWithSizeType.OpenAILarge: Dataset.OPENAI.manager(5_000_000),
}


class FtsBaseDataset(BaseModel):
    """Base class for FTS datasets - completely independent from BaseDataset.

    FTS datasets are text-based and use TSV files instead of parquet files.
    They don't have vector dimensions or metric types.

    """

    name: str
    size: int
    with_gt: bool = True
    with_remote_resource: bool = False

    _size_label: dict[int, SizeLabel]

    @property
    def collection_file(self) -> str:
        """Get collection file name based on dataset size."""
        filenames = self.get_filenames()
        return filenames["collection"]

    @property
    def queries_file(self) -> str:
        """Get queries file name based on dataset size."""
        filenames = self.get_filenames()
        return filenames["queries"]

    @property
    def qrels_file(self) -> str:
        """Get qrels file name based on dataset size."""
        filenames = self.get_filenames()
        return filenames["qrels"]

    @property
    def scoreddocs_file(self) -> str:
        """Get scoreddocs file name based on dataset size."""
        filenames = self.get_filenames()
        return filenames["scoreddocs"]

    @validator("size", allow_reuse=True)
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
    """MS MARCO Passage Retrieval dataset for FTS testing.

    MS MARCO is a large-scale dataset for information retrieval.

    Standard evaluation metrics:
    - MRR@10 (Mean Reciprocal Rank)
    - Recall@K
    - NDCG@K (Normalized Discounted Cumulative Gain)
    """

    name: str = "MSMarco"
    with_gt: bool = True
    with_remote_resource: bool = False

    _size_label: dict = {
        100_000: SizeLabel(100_000, "SMALL", 1),
    }

    def get_filenames(self) -> dict[str, str]:
        # Only support 100K dataset now, which maps to msmarco-passage/dev/small
        # Return filenames relative to data_dir (following vector dataset structure)
        return {
            "collection": "docs.small.tsv",
            "queries": "queries.small.tsv",
            "qrels": "qrels.small.tsv",
            "scoreddocs": "scoreddocs.small.tsv",
        }


class FtsDatasetManager(BaseModel):
    """Manager for FTS datasets - independent from DatasetManager.

    Handles FTS dataset preparation, file management, and provides
    access to TSV data files.

    Similar to DatasetManager, but for text-based FTS datasets:
    - queries_data: loaded queries (similar to test_data in vectors)
    - qrels_data: loaded ground truth (similar to gt_data in vectors)
    """

    data: FtsBaseDataset

    queries_data: list[dict] | None = None
    qrels_data: dict[int, list[int]] | None = None

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

    def _read_queries(self, file_path: pathlib.Path) -> list[dict]:
        """Read queries from TSV file into memory.

        Similar to DatasetManager._read_file() for parquet files.
        Args:
            file_path: Path to queries.dev.tsv file
        Returns:
            List of queries with query_id and query_text
        """
        log.info(f"Read the entire queries file into memory: {file_path}")

        if not file_path.exists():
            log.warning(f"No such file: {file_path}")
            return []

        queries = []
        try:
            with file_path.open(encoding="utf-8") as f:
                for i, line in enumerate(f):
                    try:
                        parts = line.strip().split("\t")
                        if len(parts) >= 2:
                            query_id = int(parts[0])
                            query_text = parts[1]
                            queries.append({"query_id": query_id, "query_text": query_text})
                    except Exception as e:
                        log.debug(f"Skipping malformed line {i}: {e}")
                        continue
        except Exception:
            log.exception("Error reading queries file")
            raise

        return queries

    def _read_qrels(self, file_path: pathlib.Path) -> dict[int, list[int]]:
        """Read ground truth from TSV file into memory.

        Similar to DatasetManager._read_file() for parquet files.

        Args:
            file_path: Path to qrels.dev.tsv file

        Returns:
            Dictionary mapping query_id to list of relevant doc_ids
        """
        log.info(f"Read the entire qrels file into memory: {file_path}")

        if not file_path.exists():
            log.warning(f"No such file: {file_path}")
            return {}

        ground_truth = {}
        try:
            with file_path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        parts = line.strip().split("\t")
                        if len(parts) >= 4:
                            query_id = int(parts[0])
                            doc_id = int(parts[2])
                            relevance = float(parts[3])

                            # Only keep relevant documents (relevance > 0)
                            if relevance > 0:
                                if query_id not in ground_truth:
                                    ground_truth[query_id] = []
                                ground_truth[query_id].append(doc_id)
                    except Exception as e:
                        log.debug(f"Skipping malformed line: {e}")
                        continue
        except Exception:
            log.exception("Error reading qrels file")
            raise

        return ground_truth

    def _read_scoreddocs(self, file_path: pathlib.Path) -> dict[int, list[int]]:
        """Read ground truth from scoreddocs TSV file into memory.

        Scoreddocs format: query_id\t doc_id\t score
        We treat all documents in scoreddocs as relevant (they are retrieval results).

        Args:
            file_path: Path to scoreddocs.small.tsv file

        Returns:
            Dictionary mapping query_id to list of relevant doc_ids
        """
        log.info(f"Read the entire scoreddocs file into memory: {file_path}")

        if not file_path.exists():
            log.warning(f"No such file: {file_path}")
            return {}

        ground_truth = {}
        try:
            with file_path.open(encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    # Skip header line
                    if line_num == 0:
                        continue

                    try:
                        parts = line.strip().split("\t")
                        if len(parts) >= 3:
                            query_id = int(parts[0])
                            doc_id = int(parts[1])

                            # Treat all documents in scoreddocs as relevant retrieval results
                            if query_id not in ground_truth:
                                ground_truth[query_id] = []
                            ground_truth[query_id].append(doc_id)
                    except Exception as e:
                        log.debug(f"Skipping malformed line at {line_num}: {e}")
                        continue
        except Exception:
            log.exception("Error reading scoreddocs file")
            raise

        return ground_truth

    def prepare(
        self,
        source: DatasetSource | None = None,
        filters: Filter | None = None,
    ) -> bool:
        """Prepare FTS dataset for testing.

        Similar to DatasetManager.prepare(), this method:
        1. Downloads dataset from remote if files don't exist (using data source)
        2. Checks if required TSV files exist in the dataset directory
        3. Loads queries and qrels data into memory (like test_data and gt_data)

        Args:
            source: Data source to download from (S3, AliyunOSS, IR_DATASETS)
            filters: Optional filters (not used for FTS)

        Returns:
            bool: True if preparation successful, False otherwise
        """
        log.info(f"Preparing FTS dataset: {self.data.full_name}")
        log.info(f"Dataset directory: {self.data_dir}")

        # Check if directory exists
        if not self.data_dir.exists():
            log.info(f"Creating FTS dataset directory: {self.data_dir}")
            self.data_dir.mkdir(parents=True, exist_ok=True)

        # Check if files exist
        file_status = self._check_files()

        # If files are missing and we have a data source, try to download
        if not all(file_status.values()) and source is not None:
            missing_files = [name for name, exists in file_status.items() if not exists]
            log.info(f"Missing FTS files: {missing_files}, attempting to download from {source.value}")

            try:
                reader = source.reader()
                if reader is None:
                    log.error(f"No reader available for data source: {source}")
                    return False

                dataset_name = self._get_ir_datasets_name()
                expected_files = ["queries.small.tsv", "scoreddocs.small.tsv", "docs.small.tsv"]

                reader.read(dataset_name, expected_files, self.data_dir)

                # Re-check files after download
                file_status = self._check_files()

            except Exception:
                log.exception("Failed to download FTS dataset")
                return False

        # Final check
        if not all(file_status.values()):
            missing_files = [name for name, exists in file_status.items() if not exists]
            log.error(f"FTS dataset preparation failed. Missing files: {missing_files}")
            return False

        if self.data.with_gt:
            paths = self.get_file_paths()

            # Load queries (similar to test_data in vectors)
            self.queries_data = self._read_queries(paths["queries"])
            log.info(f"Loaded {len(self.queries_data)} queries into memory")

            # Load scoreddocs as ground truth (similar to gt_data in vectors)
            self.qrels_data = self._read_scoreddocs(paths["scoreddocs"])
            log.info(f"Loaded ground truth for {len(self.qrels_data)} queries into memory")

        log.debug(f"{self.data.name}: FTS dataset prepared")
        log.info(f"FTS dataset preparation completed: {self.data.full_name}")
        return True

    def _get_ir_datasets_name(self) -> str:
        """Convert internal dataset representation to ir_datasets name"""
        return "msmarco-passage/dev/small"

    def _check_files(self) -> dict[str, bool]:
        """Check if required files exist."""
        file_status = {}

        collection_path = self.data_dir / self.data.collection_file
        file_status["collection"] = collection_path.exists()
        log.debug(f"Collection file: {collection_path} - {'✓' if file_status['collection'] else '✗'}")

        queries_path = self.data_dir / self.data.queries_file
        file_status["queries"] = queries_path.exists()
        log.debug(f"Queries file: {queries_path} - {'✓' if file_status['queries'] else '✗'}")

        if self.data.with_gt:
            scoreddocs_path = self.data_dir / self.data.scoreddocs_file
            file_status["scoreddocs"] = scoreddocs_path.exists()
            log.debug(f"Scoreddocs file: {scoreddocs_path} - {'✓' if file_status['scoreddocs'] else '✗'}")

        return file_status

    def get_file_paths(self) -> dict[str, pathlib.Path]:
        """Get absolute paths to all dataset files.

        Returns:
            dict: Dictionary with keys 'collection', 'queries', 'scoreddocs'
        """
        paths = {
            "collection": self.data_dir / self.data.collection_file,
            "queries": self.data_dir / self.data.queries_file,
        }
        if self.data.with_gt:
            # Use scoreddocs instead of qrels as ground truth
            paths["scoreddocs"] = self.data_dir / self.data.scoreddocs_file
        return paths

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
        return FtsDocumentIterator(self)


class FtsDocumentIterator:
    """Iterator for streaming FTS document batches from TSV file.

    Similar to DataSetIterator for vector datasets, but reads from TSV instead of Parquet.
    Yields batches of documents to enable memory-efficient processing of large datasets.
    """

    def __init__(self, dataset: FtsDatasetManager):
        self._ds = dataset
        self._file = None
        self._batch_size = config.NUM_PER_BATCH
        self._finished = False
        self._doc_count = 0  # Track total documents processed

    def __iter__(self):
        return self

    def __next__(self) -> list[dict]:
        """Return the next batch of documents.

        Returns:
            list[dict]: List of documents, each with 'doc_id' and 'text'

        Raises:
            StopIteration: When all documents have been read
        """
        if self._finished:
            raise StopIteration

        # Open file on first iteration
        if self._file is None:
            collection_path = self._ds.data_dir / self._ds.data.collection_file
            if not collection_path.exists():
                error_msg = f"Collection file not found: {collection_path}"
                log.error(error_msg)
                raise FileNotFoundError(error_msg)

            log.info(f"Opening collection file for streaming: {collection_path}")
            self._file = collection_path.open(encoding="utf-8")

        # Read batch with proper error handling to prevent resource leaks
        try:
            batch = []
            for _ in range(self._batch_size):
                line = self._file.readline()

                if not line:
                    self._finished = True
                    return batch

                try:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        doc_id = int(parts[0])
                        text = parts[1]
                        batch.append({"doc_id": doc_id, "text": text})
                        self._doc_count += 1
                except Exception as e:
                    log.debug(f"Skipping malformed line: {e}")
                    continue

        except Exception:
            self._close_file()
            raise
        else:
            return batch
        finally:
            if self._finished:
                self._close_file()

    def _close_file(self):
        """Close the file handle if it's open."""
        if self._file and not self._file.closed:
            self._file.close()
            self._file = None

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit context manager and ensure file is closed."""
        self._close_file()

    def __del__(self):
        """Close file handle when iterator is destroyed."""
        self._close_file()


class FtsDataset(Enum):
    """Registry of available FTS datasets - independent from Dataset enum.

    Example:
        >>> dataset = FtsDataset.MSMARCO.get(100_000)
        >>> manager = FtsDataset.MSMARCO.manager(100_000)
        >>> manager.prepare()
    """

    MSMARCO = MSMarcoFts

    def get(self, size: int) -> FtsBaseDataset:
        """Get FTS dataset configuration for specified size."""
        return self.value(size=size)

    def manager(self, size: int) -> FtsDatasetManager:
        """Get FTS dataset manager for specified size."""
        return FtsDatasetManager(data=self.get(size))


class FtsDatasetWithSizeType(Enum):
    """FTS dataset configurations with specific sizes."""

    MSMarcoSmall = "MS MARCO Small (100K documents)"

    def get_manager(self) -> FtsDatasetManager:
        """Get the FTS dataset manager for this size configuration."""
        if self == FtsDatasetWithSizeType.MSMarcoSmall:
            return FtsDataset.MSMARCO.manager(100_000)
        error_msg = f"Unknown FTS dataset size: {self.name}"
        raise ValueError(error_msg)

    def get_load_timeout(self) -> float:
        """Get load timeout for this dataset size."""
        if self == FtsDatasetWithSizeType.MSMarcoSmall:
            return config.LOAD_TIMEOUT_768D_100K
        return config.LOAD_TIMEOUT_DEFAULT

    def get_optimize_timeout(self) -> float:
        """Get optimize timeout for this dataset size."""
        if self == FtsDatasetWithSizeType.MSMarcoSmall:
            return config.OPTIMIZE_TIMEOUT_768D_100K
        return config.OPTIMIZE_TIMEOUT_DEFAULT
