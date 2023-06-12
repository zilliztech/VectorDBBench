"""
Usage:
    >>> from xxx import dataset as ds
    >>> gist_s = ds.get(ds.Name.GIST, ds.Label.SMALL)
    >>> gist_s.dict()
    dataset: {'data': {'name': 'GIST', 'dim': 128, 'metric_type': 'L2', 'label': 'SMALL', 'size': 50000000}, 'data_dir': 'xxx'}
"""

import os
import logging
import pathlib
import math
from hashlib import md5
from enum import Enum, auto
from typing import Any

import s3fs
import pandas as pd
from tqdm import tqdm
from pydantic.dataclasses import dataclass

from ..base import BaseModel
from .. import config
from ..backend.clients import MetricType
from . import utils

log = logging.getLogger(__name__)

@dataclass
class LAION:
    name: str = "LAION"
    dim: int = 768
    metric_type: MetricType = MetricType.COSINE
    use_shuffled: bool = False

    @property
    def dir_name(self) -> str:
        return f"{self.name}_{self.label}_{utils.numerize(self.size)}".lower()

@dataclass
class GIST:
    name: str = "GIST"
    dim: int = 960
    metric_type: MetricType = MetricType.L2
    use_shuffled: bool = False

    @property
    def dir_name(self) -> str:
        return f"{self.name}_{self.label}_{utils.numerize(self.size)}".lower()

@dataclass
class Cohere:
    name: str = "Cohere"
    dim: int = 768
    metric_type: MetricType = MetricType.COSINE
    use_shuffled: bool = config.USE_SHUFFLED_DATA

    @property
    def dir_name(self) -> str:
        return f"{self.name}_{self.label}_{utils.numerize(self.size)}".lower()

@dataclass
class Glove:
    name: str = "Glove"
    dim: int = 200
    metric_type: MetricType = MetricType.COSINE
    use_shuffled: bool = False

    @property
    def dir_name(self) -> str:
        return f"{self.name}_{self.label}_{utils.numerize(self.size)}".lower()

@dataclass
class SIFT:
    name: str = "SIFT"
    dim: int = 128
    metric_type: MetricType = MetricType.COSINE
    use_shuffled: bool = False

    @property
    def dir_name(self) -> str:
        return f"{self.name}_{self.label}_{utils.numerize(self.size)}".lower()

@dataclass
class LAION_L(LAION):
    label: str = "LARGE"
    size: int  = 100_000_000

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
    data:   GIST | Cohere | Glove | SIFT | Any
    test_data: pd.DataFrame | None = None
    train_files : list[str] = []

    def __eq__(self, obj):
        if isinstance(obj, DataSet):
            return self.data.name == obj.data.name and \
                self.data.label == obj.data.label
        return False

    @property
    def data_dir(self) -> pathlib.Path:
        """ data local directory: config.DATASET_LOCAL_DIR/{dataset_name}/{dataset_dirname}

        Examples:
            >>> sift_s = DataSet(data=SIFT_L())
            >>> sift_s.relative_path
            '/tmp/vector_db_bench/dataset/sift/sift_small_500k/'
        """
        return pathlib.Path(config.DATASET_LOCAL_DIR, self.data.name.lower(), self.data.dir_name.lower())

    @property
    def download_dir(self) -> str:
        """ data s3 directory: config.DEFAULT_DATASET_URL/{dataset_dirname}

        Examples:
            >>> sift_s = DataSet(data=SIFT_L())
            >>> sift_s.download_dir
            'assets.zilliz.com/benchmark/sift_small_500k'
        """
        return f"{config.DEFAULT_DATASET_URL}{self.data.dir_name}"

    def __iter__(self):
        return DataSetIterator(self)


    def _validate_local_file(self):
        if not self.data_dir.exists():
            log.info(f"local file path not exist, creating it: {self.data_dir}")
            self.data_dir.mkdir(parents=True)

        fs = s3fs.S3FileSystem(
            anon=True,
            client_kwargs={'region_name': 'us-west-2'}
        )
        dataset_info = fs.ls(self.download_dir, detail=True)
        if len(dataset_info) == 0:
            raise ValueError(f"No data in s3 for dataset: {self.download_dir}")
        path2etag = {info['Key']: info['ETag'].split('"')[1] for info in dataset_info}

        perfix_to_filter = "train" if self.data.use_shuffled else "shuffle_train"
        filtered_keys = [key for key in path2etag.keys() if key.split("/")[-1].startswith(perfix_to_filter)]
        for k in filtered_keys:
            path2etag.pop(k)

        # get local files ended with '.parquet'
        file_names = [p.name for p in self.data_dir.glob("*.parquet")]
        log.info(f"local files: {file_names}")
        log.info(f"s3 files: {path2etag.keys()}")
        downloads = []
        if len(file_names) == 0:
            log.info("no local files, set all to downloading lists")
            downloads = path2etag.keys()
        else:
            # if local file exists, check the etag of local file with s3,
            # make sure data files aren't corrupted.
            for name in tqdm([key.split("/")[-1] for key in path2etag.keys()]):
                s3_path = f"{self.download_dir}/{name}"
                local_path = self.data_dir.joinpath(name)
                log.debug(f"s3 path: {s3_path}, local_path: {local_path}")
                if not local_path.exists():
                    log.info(f"local file not exists: {local_path}, add to downloading lists")
                    downloads.append(s3_path)

                elif not self.match_etag(path2etag.get(s3_path), local_path):
                    log.info(f"local file etag not match with s3 file: {local_path}, add to downloading lists")
                    downloads.append(s3_path)

        for s3_file in tqdm(downloads):
            log.debug(f"downloading file {s3_file} to {self.data_dir}")
            fs.download(s3_file, self.data_dir.as_posix())

    def match_etag(self, expected_etag: str, local_file) -> bool:
        """Check if local files' etag match with S3"""
        def factor_of_1MB(filesize, num_parts):
            x = filesize / int(num_parts)
            y = x % 1048576
            return int(x + 1048576 - y)

        def calc_etag(inputfile, partsize):
            md5_digests = []
            with open(inputfile, 'rb') as f:
                for chunk in iter(lambda: f.read(partsize), b''):
                    md5_digests.append(md5(chunk).digest())
            return md5(b''.join(md5_digests)).hexdigest() + '-' + str(len(md5_digests))

        def possible_partsizes(filesize, num_parts):
            return lambda partsize: partsize < filesize and (float(filesize) / float(partsize)) <= num_parts

        filesize = os.path.getsize(local_file)
        le = ""
        if '-' not in expected_etag: # no spliting uploading
            with open(local_file, 'rb') as f:
                le = md5(f.read()).hexdigest()
                log.debug(f"calculated local etag {le}, expected etag: {expected_etag}")
                return expected_etag == le
        else:
            num_parts = int(expected_etag.split('-')[-1])
            partsizes = [ ## Default Partsizes Map
                8388608, # aws_cli/boto3
                15728640, # s3cmd
                factor_of_1MB(filesize, num_parts) # Used by many clients to upload large files
            ]

            for partsize in filter(possible_partsizes(filesize, num_parts), partsizes):
                le = calc_etag(local_file, partsize)
                log.debug(f"calculated local etag {le}, expected etag: {expected_etag}")
                if expected_etag == le:
                    return True
        return False

    def prepare(self, check=True) -> bool:
        """Download the dataset from S3
         url = f"{config.DEFAULT_DATASET_URL}/{self.data.dir_name}"

         download files from url to self.data_dir, there'll be 4 types of files in the data_dir
             - train*.parquet: for training
             - test.parquet: for testing
             - neighbors.parquet: ground_truth of the test.parquet
             - neighbors_90p.parquet: ground_truth of the test.parquet after filtering 90% data
             - neighbors_head_1p.parquet: ground_truth of the test.parquet after filtering 1% data
             - neighbors_99p.parquet: ground_truth of the test.parquet after filtering 99% data
        """
        if check:
            self._validate_local_file()

        prefix = "shuffle_train" if self.data.use_shuffled else "train"
        self.train_files = sorted([f.name for f in self.data_dir.glob(f'{prefix}*.parquet')])
        log.debug(f"{self.data.name}: available train files {self.train_files}")
        self.test_data = self._read_file("test.parquet")
        return True

    def get_ground_truth(self, filters: int | float | None = None) -> pd.DataFrame:

        file_name = ""
        if filters is None or filters == 100:
            file_name = "neighbors.parquet"
        elif filters == 0.9:
            file_name = "neighbors_90p.parquet"
        elif filters == 0.01:
            file_name = "neighbors_head_1p.parquet"
        elif filters == 0.99:
            file_name = "neighbors_tail_1p.parquet"
        else:
            raise ValueError(f"Filters not supported: {filters}")
        return self._read_file(file_name)

    def _read_file(self, file_name: str) -> pd.DataFrame:
        """read one file from disk into memory"""
        import pyarrow.parquet as pq

        p = pathlib.Path(self.data_dir, file_name)
        log.info(f"reading file into memory: {p}")
        if not p.exists():
            log.warning(f"No such file: {p}")
            return pd.DataFrame()
        data = pq.read_table(p)
        df = data.to_pandas()
        return df


class DataSetIterator:
    def __init__(self, dataset: DataSet):
        self._ds = dataset
        self._idx = 0  # file number
        self._curr: pd.DataFrame | None = None
        self._sub_idx = [0 for i in range(len(self._ds.train_files))] # iter num for each file

    def __next__(self) -> pd.DataFrame:
        """return the data in the next file of the training list"""
        if self._idx < len(self._ds.train_files):
            _sub = self._sub_idx[self._idx]
            if _sub == 0 and self._idx == 0: # init
                file_name = self._ds.train_files[self._idx]
                self._curr = self._ds._read_file(file_name)
                self._iter_num = math.ceil(self._curr.shape[0]/100_000)

            if _sub == self._iter_num:
                if self._idx == len(self._ds.train_files) - 1:
                    self._curr = None
                    raise StopIteration
                else:
                    self._idx += 1
                    _sub = self._sub_idx[self._idx]

                    self._curr = None
                    file_name = self._ds.train_files[self._idx]
                    self._curr = self._ds._read_file(file_name)

            sub_df = self._curr[_sub*100_000: (_sub+1)*100_000]
            self._sub_idx[self._idx] += 1
            log.info(f"Get the [{_sub+1}/{self._iter_num}] batch of {self._idx+1}/{len(self._ds.train_files)} train file")
            return sub_df
        self._curr = None
        raise StopIteration


class Name(Enum):
    GIST = auto()
    Cohere = auto()
    Glove = auto()
    SIFT = auto()
    LAION = auto()


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
    Name.LAION: {
        Label.LARGE: DataSet(data=LAION_L()),
    },
}

def get(ds: Name, label: Label):
    return _global_ds_mapping.get(ds, {}).get(label)
