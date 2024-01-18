"""
Usage:
    >>> from xxx.dataset import Dataset
    >>> Dataset.Cohere.get(100_000)
"""

from collections import namedtuple
import logging
import pathlib
from enum import Enum
import pandas as pd
from pydantic import validator, PrivateAttr
import polars as pl
from pyarrow.parquet import ParquetFile

from ..base import BaseModel
from .. import config
from ..backend.clients import MetricType
from . import utils
from .data_source import DatasetSource, DatasetReader

log = logging.getLogger(__name__)


SizeLabel = namedtuple('SizeLabel', ['size', 'label', 'files'])


class BaseDataset(BaseModel):
    name: str
    size: int
    dim: int
    metric_type: MetricType
    use_shuffled: bool
    _size_label: dict[int, SizeLabel] = PrivateAttr()

    @validator("size")
    def verify_size(cls, v):
        if v not in cls._size_label:
            raise ValueError(f"Size {v} not supported for the dataset, expected: {cls._size_label.keys()}")
        return v

    @property
    def label(self) -> str:
        return self._size_label.get(self.size).label

    @property
    def dir_name(self) -> str:
        return f"{self.name}_{self.label}_{utils.numerize(self.size)}".lower()

    @property
    def files(self) -> str:
        return self._size_label.get(self.size).files


def get_files(train_count: int, use_shuffled: bool, with_gt: bool = True) -> list[str]:
    prefix = "shuffle_train" if use_shuffled else "train"
    middle = f"of-{train_count}"
    surfix = "parquet"

    train_files = []
    if train_count > 1:
        just_size = len(str(train_count))
        for i in range(train_count):
            sub_file = f"{prefix}-{str(i).rjust(just_size, '0')}-{middle}.{surfix}"
            train_files.append(sub_file)
    else:
        train_files.append(f"{prefix}.{surfix}")

    files = ['test.parquet']
    if with_gt:
        files.extend([
            'neighbors.parquet',
            'neighbors_tail_1p.parquet',
            'neighbors_head_1p.parquet',
        ])

    files.extend(train_files)
    return files


class LAION(BaseDataset):
    name: str = "LAION"
    dim: int = 768
    metric_type: MetricType = MetricType.L2
    use_shuffled: bool = False
    _size_label: dict = {
        100_000_000: SizeLabel(100_000_000, "LARGE", get_files(100, False)),
    }


class GIST(BaseDataset):
    name: str = "GIST"
    dim: int = 960
    metric_type: MetricType = MetricType.L2
    use_shuffled: bool = False
    _size_label: dict = {
        100_000: SizeLabel(100_000, "SMALL", get_files(1, False, False)),
        1_000_000: SizeLabel(1_000_000, "MEDIUM", get_files(1, False, False)),
    }


class Cohere(BaseDataset):
    name: str = "Cohere"
    dim: int = 768
    metric_type: MetricType = MetricType.COSINE
    use_shuffled: bool = config.USE_SHUFFLED_DATA
    _size_label: dict = {
        100_000: SizeLabel(100_000, "SMALL", get_files(1, config.USE_SHUFFLED_DATA)),
        1_000_000: SizeLabel(1_000_000, "MEDIUM", get_files(1, config.USE_SHUFFLED_DATA)),
        10_000_000: SizeLabel(10_000_000, "LARGE", get_files(10, config.USE_SHUFFLED_DATA)),
    }


class Glove(BaseDataset):
    name: str = "Glove"
    dim: int = 200
    metric_type: MetricType = MetricType.COSINE
    use_shuffled: bool = False
    _size_label: dict = {1_000_000: SizeLabel(1_000_000, "MEDIUM", get_files(1, False, False))}


class SIFT(BaseDataset):
    name: str = "SIFT"
    dim: int = 128
    metric_type: MetricType = MetricType.L2
    use_shuffled: bool = False
    _size_label: dict = {
        500_000: SizeLabel(500_000, "SMALL", get_files(1, False, False)),
        5_000_000: SizeLabel(5_000_000, "MEDIUM", get_files(1, False, False)),
        #  50_000_000: SizeLabel(50_000_000, "LARGE", get_files(50, False, False)),
    }


class OpenAI(BaseDataset):
    name: str = "OpenAI"
    dim: int = 1536
    metric_type: MetricType = MetricType.COSINE
    use_shuffled: bool = config.USE_SHUFFLED_DATA
    _size_label: dict = {
        50_000: SizeLabel(50_000, "SMALL", get_files(1, config.USE_SHUFFLED_DATA)),
        500_000: SizeLabel(500_000, "MEDIUM", get_files(1, config.USE_SHUFFLED_DATA)),
        5_000_000: SizeLabel(5_000_000, "LARGE", get_files(10, config.USE_SHUFFLED_DATA)),
    }


class DatasetManager(BaseModel):
    """Download dataset if not in the local directory. Provide data for cases.

    DatasetManager is iterable, each iteration will return the next batch of data in pandas.DataFrame

    Examples:
        >>> cohere = Dataset.COHERE.manager(100_000)
        >>> for data in cohere:
        >>>    print(data.columns)
    """
    data:   BaseDataset
    test_data: pd.DataFrame | None = None
    train_files : list[str] = []
    reader: DatasetReader | None = None

    def __eq__(self, obj):
        if isinstance(obj, DatasetManager):
            return self.data.name == obj.data.name and self.data.label == obj.data.label
        return False

    def set_reader(self, reader: DatasetReader):
        self.reader = reader

    @property
    def data_dir(self) -> pathlib.Path:
        """ data local directory: config.DATASET_LOCAL_DIR/{dataset_name}/{dataset_dirname}

        Examples:
            >>> sift_s = Dataset.SIFT.manager(500_000)
            >>> sift_s.relative_path
            '/tmp/vectordb_bench/dataset/sift/sift_small_500k/'
        """
        return pathlib.Path(config.DATASET_LOCAL_DIR, self.data.name.lower(), self.data.dir_name.lower())

    def __iter__(self):
        return DataSetIterator(self)

    def prepare(self, source: DatasetSource=DatasetSource.S3, check: bool=True) -> bool:
        """Download the dataset from DatasetSource
         url = f"{source}/{self.data.dir_name}"

         download files from url to self.data_dir, there'll be 4 types of files in the data_dir
             - train*.parquet: for training
             - test.parquet: for testing
             - neighbors.parquet: ground_truth of the test.parquet
             - neighbors_head_1p.parquet: ground_truth of the test.parquet after filtering 1% data
             - neighbors_99p.parquet: ground_truth of the test.parquet after filtering 99% data

        Args:
            source(DatasetSource): S3 or AliyunOSS, default as S3
            check(bool): Whether to do etags check

        Returns:
            bool: whether the dataset is successfully prepared

        """
        source.reader().read(
            dataset=self.data.dir_name.lower(),
            files=self.data.files,
            local_ds_root=self.data_dir,
        )

        prefix = "shuffle_train" if self.data.use_shuffled else "train"
        self.train_files = sorted([f.name for f in self.data_dir.glob(f'{prefix}*.parquet')])
        log.debug(f"{self.data.name}: available train files {self.train_files}")
        self.test_data = self._read_file("test.parquet")
        return True

    def get_ground_truth(self, filters: int | float | None = None) -> pd.DataFrame:

        file_name = ""
        if filters is None:
            file_name = "neighbors.parquet"
        elif filters == 0.01:
            file_name = "neighbors_head_1p.parquet"
        elif filters == 0.99:
            file_name = "neighbors_tail_1p.parquet"
        else:
            raise ValueError(f"Filters not supported: {filters}")
        return self._read_file(file_name)

    def _read_file(self, file_name: str) -> pd.DataFrame:
        """read one file from disk into memory"""
        log.info(f"Read the entire file into memory: {file_name}")
        p = pathlib.Path(self.data_dir, file_name)
        if not p.exists():
            log.warning(f"No such file: {p}")
            return pd.DataFrame()

        return pl.read_parquet(p)


class DataSetIterator:
    def __init__(self, dataset: DatasetManager):
        self._ds = dataset
        self._idx = 0  # file number
        self._cur = None
        self._sub_idx = [0 for i in range(len(self._ds.train_files))] # iter num for each file

    def _get_iter(self, file_name: str):
        p = pathlib.Path(self._ds.data_dir, file_name)
        log.info(f"Get iterator for {p.name}")
        if not p.exists():
            raise IndexError(f"No such file {p}")
            log.warning(f"No such file: {p}")
        return ParquetFile(p).iter_batches(config.NUM_PER_BATCH)

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
    GLOVE = Glove
    SIFT = SIFT
    OPENAI = OpenAI

    def get(self, size: int) -> BaseDataset:
        return self.value(size=size)

    def manager(self, size: int) -> DatasetManager:
        return DatasetManager(data=self.get(size))
