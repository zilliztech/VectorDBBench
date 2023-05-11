"""
Usage:
    >>> from xxx import dataset as ds
    >>> gist_s = ds.get(ds.Name.GIST, ds.Label.SMALL)
    >>> gist_s.model_dump()
    dataset: {'data': {'name': 'GIST', 'dim': 128, 'metric_type': 'L2', 'label': 'SMALL', 'size': 50000000}, 'data_dir': 'xxx'}
"""

import os
import logging
from enum import Enum, auto

import pandas as pd
from pydantic import BaseModel, computed_field, ConfigDict
from pydantic.dataclasses import dataclass

from .. import DATASET_LOCAL_DIR
from . import utils

log = logging.getLogger(__name__)

@dataclass
class GIST:
    name: str = "GIST"
    dim: int = 960
    metric_type: str = "L2"

    @computed_field
    @property
    def dir_name(self) -> str:
        return f"{self.name}_{self.label}_{utils.numerize(self.size)}".lower()

@dataclass
class Cohere:
    name: str = "Cohere"
    dim: int = 768
    metric_type: str = "Consine"

    @computed_field
    @property
    def dir_name(self) -> str:
        return f"{self.name}_{self.label}_{utils.numerize(self.size)}".lower()

@dataclass
class Glove:
    name: str = "Glove"
    dim: int = 200
    metric_type: str = "Consine"

    @computed_field
    @property
    def dir_name(self) -> str:
        return f"{self.name}_{self.label}_{utils.numerize(self.size)}".lower()

@dataclass
class SIFT:
    name: str = "SIFT"
    dim: int = 128
    metric_type: str = "L2"

    @computed_field
    @property
    def dir_name(self) -> str:
        return f"{self.name}_{self.label}_{utils.numerize(self.size)}".lower()

@dataclass
class GIST_S(GIST):
    label: str = "SMALL"
    size: int  = 100_000

@dataclass
class GIST_M(GIST):
    label: str = "MEDIUM"
    size: int  = 1_000_000

@dataclass
class Cohere_S(Cohere):
    label: str = "SMALL"
    size: int  = 100_000

@dataclass
class Cohere_M(Cohere):
    label: str = "MEDIUM"
    size: int = 1_000_000

@dataclass
class Cohere_L(Cohere):
    label : str = "LARGE"
    size  : int = 10_000_000

@dataclass
class Glove_S(Glove):
    label: str = "SMALL"
    size : int = 100_000

@dataclass
class Glove_M(Glove):
    label: str = "MEDIUM"
    size : int = 1_000_000

@dataclass
class SIFT_S(SIFT):
    label: str = "SMALL"
    size : int = 500_000

@dataclass
class SIFT_M(SIFT):
    label: str = "MEDIUM"
    size : int = 5_000_000

@dataclass
class SIFT_L(SIFT):
    label: str = "LARGE"
    size : int = 50_000_000


class DataSet(BaseModel):
    """Download dataset if not int the local directory. Provide data for cases.

    DataSet is iterable, each iteration will return the next batch of data in pandas.DataFrame

    Examples:
        >>> cohere_s = DataSet(data=Cohere_S)
        >>> for data in cohere_s:
        >>>    print(data.columns)
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data:   GIST | Cohere | Glove | SIFT
    train_files : list[str] = []

    @computed_field
    @property
    def data_dir(self) -> str:
        """ data local directory: DATASET_LOCAL_DIR/{dataset_name}/{dataset_dirname}

        Examples:
            >>> sift_s = DataSet(data=SIFT_L())
            >>> sift_s.relative_path
            '/tmp/vector_db_bench/dataset/sift/sift_small_500k/'
        """
        relative_path = os.path.join(self.data.name, self.data.dir_name).lower()
        return os.path.join(DATASET_LOCAL_DIR, relative_path)

    def __iter__(self):
        return DataSetIterator(self)

    def prepare(self) -> bool:
        """Download the dataset from S3
         url = f"{DEFAULT_DATASET_URL}/{self.data.dir_name}"

         download files from url to self.data_dir, there'll be 3 types of files in the data_dir
             - train*.parquet: for training
             - test.parquet: for testing
             - neighbors.parquet: ground_truth of the test.parquet
        """
        if not os.path.exists(self.data_dir):
            log.info(f"{self.data.name}: local file path not exist, creating it: {self.data_dir}")
            os.makedirs(self.data_dir)

        if len(os.listdir()) == 0:
            log.info(f"{self.data.name}: no data in the local file path, downloading it from s3")
            # TODO download

        # TODO: check md5 of the dir?
        # if not correct: re-download from url
        # url = f"{DEFAULT_DATASET_URL}/{self.data.dir_name}"

        self.train_files = sorted([f for f in os.listdir(self.data_dir) if "train" in f])

        log.debug(f"{self.data.name}: available train files {self.train_files}")
        return True

    def test(self) -> pd.DataFrame:
        """test data"""
        return self._read_file("test.parquet")

    def ground_truth(self) -> pd.DataFrame:
        """ground truth of the test data"""
        return self._read_file("neighbors.parquet")

    def _read_file(self, file_name: str) -> pd.DataFrame:
        """read one file from disk into memory"""
        path = os.path.join(self.data_dir, file_name)
        return pd.read_parquet(path)


class DataSetIterator:
    def __init__(self, dataset: DataSet):
        self._ds = dataset
        self._idx = 0

    def __next__(self) -> pd.DataFrame:
        """return the data in the next file of the training list"""
        if self._idx < len(self._ds.train_files):
            file_name = self._ds.train_files[self._idx]
            df_data = self._ds._read_file(file_name)
            self._idx +=1
            return df_data
        raise StopIteration


class Name(Enum):
    GIST = auto()
    Cohere = auto()
    Glove = auto()
    SIFT = auto()


class Label(Enum):
    SMALL = auto()
    MEDIUM = auto()
    LARGE = auto()

_global_ds_mapping = {
    Name.GIST: {
        Label.SMALL: DataSet(data=GIST_S()),
        Label.MEDIUM: DataSet(data=GIST_M()),
    },
    Name.Cohere: {
        Label.SMALL: DataSet(data=Cohere_S()),
        Label.MEDIUM: DataSet(data=Cohere_M()),
        Label.LARGE: DataSet(data=Cohere_L()),
    },
    Name.Glove:{
        Label.SMALL: DataSet(data=Glove_S()),
        Label.MEDIUM: DataSet(data=Glove_M()),
    },
    Name.SIFT: {
        Label.SMALL: DataSet(data=SIFT_S()),
        Label.MEDIUM: DataSet(data=SIFT_M()),
        Label.LARGE: DataSet(data=SIFT_L()),
    },
}

def get(ds: Name, label: Label):
    return _global_ds_mapping.get(ds, {}).get(label)
