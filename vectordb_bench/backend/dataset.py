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


SizeLabel = namedtuple('SizeLabel', ['size', 'label', 'file_count'])


class BaseDataset(BaseModel):
    name: str
    size: int
    dim: int
    metric_type: MetricType
    use_shuffled: bool
    with_gt: bool = False
    _size_label: dict[int, SizeLabel] = PrivateAttr()
    isCustom: bool = False

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
    def file_count(self) -> int:
        return self._size_label.get(self.size).file_count

class CustomDataset(BaseDataset):
    dir: str
    file_num: int
    isCustom: bool = True

    @validator("size")
    def verify_size(cls, v):
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
    with_gt: bool = True,
    _size_label: dict = {
        100_000: SizeLabel(100_000, "SMALL", 1),
        1_000_000: SizeLabel(1_000_000, "MEDIUM", 1),
        10_000_000: SizeLabel(10_000_000, "LARGE", 10),
    }


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
        500_000: SizeLabel(500_000, "SMALL", 1,),
        5_000_000: SizeLabel(5_000_000, "MEDIUM", 1),
        #  50_000_000: SizeLabel(50_000_000, "LARGE", 50),
    }


class OpenAI(BaseDataset):
    name: str = "OpenAI"
    dim: int = 1536
    metric_type: MetricType = MetricType.COSINE
    use_shuffled: bool = config.USE_SHUFFLED_DATA
    with_gt: bool = True,
    _size_label: dict = {
        50_000: SizeLabel(50_000, "SMALL", 1),
        500_000: SizeLabel(500_000, "MEDIUM", 1),
        5_000_000: SizeLabel(5_000_000, "LARGE", 10),
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
    gt_data: pd.DataFrame | None = None
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

    # TODO passing use_shuffle from outside
    def prepare(self,
        source: DatasetSource=DatasetSource.S3,
        filters: int | float | str | None = None,
    ) -> bool:
        """Download the dataset from DatasetSource
         url = f"{source}/{self.data.dir_name}"

        Args:
            source(DatasetSource): S3 or AliyunOSS, default as S3
            filters(Optional[int | float | str]): combined with dataset's with_gt to
              compose the correct ground_truth file

        Returns:
            bool: whether the dataset is successfully prepared

        """
        file_count, use_shuffled = self.data.file_count, self.data.use_shuffled

        train_files = utils.compose_train_files(file_count, use_shuffled)
        all_files = train_files

        gt_file, test_file = None, None
        if self.data.with_gt:
            gt_file, test_file = utils.compose_gt_file(filters), "test.parquet"
            all_files.extend([gt_file, test_file])

        if not self.data.isCustom:
            source.reader().read(
                dataset=self.data.dir_name.lower(),
                files=all_files,
                local_ds_root=self.data_dir,
            )

        if gt_file is not None and test_file is not None:
            self.test_data = self._read_file(test_file)
            self.gt_data = self._read_file(gt_file)

        prefix = "shuffle_train" if use_shuffled else "train"
        self.train_files = sorted([f.name for f in self.data_dir.glob(f'{prefix}*.parquet')])
        log.debug(f"{self.data.name}: available train files {self.train_files}")

        return True

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

    def __iter__(self):
        return self

    def _get_iter(self, file_name: str):
        p = pathlib.Path(self._ds.data_dir, file_name)
        log.info(f"Get iterator for {p.name}")
        if not p.exists():
            raise IndexError(f"No such file {p}")
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
    GLOVE = Glove
    SIFT = SIFT
    OPENAI = OpenAI

    def get(self, size: int) -> BaseDataset:
        return self.value(size=size)

    def manager(self, size: int) -> DatasetManager:
        return DatasetManager(data=self.get(size))
