"""
Usage:
    >>> from xxx import dataset as ds
    >>> gist_s = ds.get(ds.Name.GIST, ds.Label.SMALL)
    >>> gist_s.model_dump()
    dataset: {'data': {'name': 'GIST', 'dim': 128, 'metric_type': 'L2', 'label': 'SMALL', 'size': 50000000}, 'data_dir': 'xxx'}
"""

import os
import logging
import pathlib
from hashlib import md5
from enum import Enum, auto

import s3fs
import pandas as pd
from pydantic import BaseModel, computed_field, ConfigDict
from pydantic.dataclasses import dataclass

from .. import DATASET_LOCAL_DIR, DEFAULT_DATASET_URL
from ..models import MetricType
from . import utils

log = logging.getLogger(__name__)

@dataclass
class GIST:
    name: str = "GIST"
    dim: int = 960
    metric_type: MetricType = MetricType.L2

    @computed_field
    @property
    def dir_name(self) -> str:
        return f"{self.name}_{self.label}_{utils.numerize(self.size)}".lower()

@dataclass
class Cohere:
    name: str = "Cohere"
    dim: int = 768
    metric_type: MetricType = MetricType.COSIN

    @computed_field
    @property
    def dir_name(self) -> str:
        return f"{self.name}_{self.label}_{utils.numerize(self.size)}".lower()

@dataclass
class Glove:
    name: str = "Glove"
    dim: int = 200
    metric_type: MetricType = MetricType.COSIN

    @computed_field
    @property
    def dir_name(self) -> str:
        return f"{self.name}_{self.label}_{utils.numerize(self.size)}".lower()

@dataclass
class SIFT:
    name: str = "SIFT"
    dim: int = 128
    metric_type: MetricType = MetricType.COSIN

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
    test_data: pd.DataFrame | None = None
    ground_truth: pd.DataFrame | None = None
    train_files : list[str] = []

    @computed_field
    @property
    def data_dir(self) -> str:
        # TODO change str into pathlib.Path
        """ data local directory: DATASET_LOCAL_DIR/{dataset_name}/{dataset_dirname}

        Examples:
            >>> sift_s = DataSet(data=SIFT_L())
            >>> sift_s.relative_path
            '/tmp/vector_db_bench/dataset/sift/sift_small_500k/'
        """
        relative_path = os.path.join(self.data.name, self.data.dir_name).lower()
        return os.path.join(DATASET_LOCAL_DIR, relative_path)

    @computed_field
    @property
    def download_dir(self) -> str:
        """ data s3 directory: DEFAULT_DATASET_URL/{dataset_dirname}

        Examples:
            >>> sift_s = DataSet(data=SIFT_L())
            >>> sift_s.download_dir
            'assets.zilliz.com/benchmark/sift_small_500k'
        """
        return f"{DEFAULT_DATASET_URL}{self.data.dir_name}"

    def __iter__(self):
        return DataSetIterator(self)

    def _validate_local_file(self):
        data_dir = pathlib.Path(self.data_dir)

        if not data_dir.exists():
            log.info(f"local file path not exist, creating it: {data_dir}")
            data_dir.mkdir(parents=True)

        fs = s3fs.S3FileSystem(
            anon=True,
            client_kwargs={'region_name': 'us-west-2'}
        )
        dataset_info = fs.ls(self.download_dir, detail=True)
        if len(dataset_info) == 0:
            raise ValueError(f"No data in s3 for dataset: {self.download_dir}")
        path2etag = {info['Key']: info['ETag'].split('"')[1] for info in dataset_info}

        # get local files ended with '.parquet'
        file_names = [p.name for p in data_dir.glob("*.parquet")]
        log.info(f"local files: {file_names}, s3 files: {path2etag}")
        if len(file_names) != len(path2etag):
            log.info("local file number mismatch with s3, downloading...")
            for s3_path in path2etag.keys():
                log.info(f"downloading file {s3_path} to {data_dir}")
                fs.download(s3_path, data_dir.as_posix())
        else:
            # if numbers of local file matches with s3, check the etag of local file,
            # make sure data files aren't corrupted.
            for name in [key.split("/")[-1] for key in path2etag.keys()]:
                s3_path = f"{self.download_dir}/{name}"
                local_path = data_dir.joinpath(name)
                log.debug(f"s3 path: {s3_path}, local_path: {local_path}")
                if name in file_names:
                    if self.match_etag(path2etag.get(s3_path), local_path):
                        continue
                log.info(f"local file {name} missing or etag not match, redownloading...")
                fs.download(s3_path, local_path.as_posix())

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
                log.info(f"calculated local etag {le}, expected etag: {expected_etag}")
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
                log.info(f"calculated local etag {le}, expected etag: {expected_etag}")
                if expected_etag == le:
                    return True
        return False

    def prepare(self) -> bool:
        """Download the dataset from S3
         url = f"{DEFAULT_DATASET_URL}/{self.data.dir_name}"

         download files from url to self.data_dir, there'll be 3 types of files in the data_dir
             - train*.parquet: for training
             - test.parquet: for testing
             - neighbors.parquet: ground_truth of the test.parquet
        """
        self._validate_local_file()

        data_dir = pathlib.Path(self.data_dir)
        self.train_files = sorted([f.name for f in data_dir.glob('train*.parquet')])
        self.test_data = self._read_file("test.parquet")
        self.ground_truth = self._read_file("neighbors.parquet")

        log.debug(f"{self.data.name}: available train files {self.train_files}")
        return True

    def _read_file(self, file_name: str) -> pd.DataFrame:
        """read one file from disk into memory"""
        p = pathlib.Path(self.data_dir, file_name)
        return pd.read_parquet(p)


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

checksums = {
    #  get(Name.GIST, Label.SMALL).data.dir_name: "",
    get(Name.Cohere, Label.SMALL).data.dir_name: "110154351655098665637835926551405536153",

}
