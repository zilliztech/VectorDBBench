import logging
import os
import pathlib
import typing
from abc import ABC, abstractmethod
from enum import Enum

import ir_datasets
from tqdm import tqdm

from vectordb_bench import config

# Set ir_datasets to use tmp directory for both home and temp
# This ensures all downloaded files and temporary data are stored in /tmp
ir_datasets_home = pathlib.Path(config.DATASET_LOCAL_DIR) / "ir_datasets"
ir_datasets_tmp = pathlib.Path(config.DATASET_LOCAL_DIR) / "ir_datasets_tmp"
os.environ.setdefault("IR_DATASETS_HOME", str(ir_datasets_home))
os.environ.setdefault("IR_DATASETS_TMP", str(ir_datasets_tmp))


logging.getLogger("s3fs").setLevel(logging.CRITICAL)

log = logging.getLogger(__name__)

DatasetReader = typing.TypeVar("DatasetReader")


class DatasetSource(Enum):
    S3 = "S3"
    AliyunOSS = "AliyunOSS"
    IR_DATASETS = "IR_DATASETS"

    def reader(self) -> DatasetReader:
        if self == DatasetSource.S3:
            return AwsS3Reader()

        if self == DatasetSource.AliyunOSS:
            return AliyunOSSReader()

        if self == DatasetSource.IR_DATASETS:
            return IRDatasetsReader()

        return None


class DatasetReader(ABC):
    source: DatasetSource
    remote_root: str

    @abstractmethod
    def read(self, dataset: str, files: list[str], local_ds_root: pathlib.Path):
        """read dataset files from remote_root to local_ds_root,

        Args:
            dataset(str): for instance "sift_small_500k"
            files(list[str]):  all filenames of the dataset
            local_ds_root(pathlib.Path): whether to write the remote data.
        """

    @abstractmethod
    def validate_file(self, remote: pathlib.Path, local: pathlib.Path) -> bool:
        pass


class AliyunOSSReader(DatasetReader):
    source: DatasetSource = DatasetSource.AliyunOSS
    remote_root: str = config.ALIYUN_OSS_URL

    def __init__(self):
        import oss2

        self.bucket = oss2.Bucket(oss2.AnonymousAuth(), self.remote_root, "benchmark", True)

    def validate_file(self, remote: pathlib.Path, local: pathlib.Path) -> bool:
        info = self.bucket.get_object_meta(remote.as_posix())

        # check size equal
        remote_size, local_size = info.content_length, local.stat().st_size
        if remote_size != local_size:
            log.info(f"local file: {local} size[{local_size}] not match with remote size[{remote_size}]")
            return False

        return True

    def read(self, dataset: str, files: list[str], local_ds_root: pathlib.Path):
        downloads = []
        if not local_ds_root.exists():
            log.info(f"local dataset root path not exist, creating it: {local_ds_root}")
            local_ds_root.mkdir(parents=True)
            downloads = [
                (
                    pathlib.PurePosixPath("benchmark", dataset, f),
                    local_ds_root.joinpath(f),
                )
                for f in files
            ]

        else:
            for file in files:
                remote_file = pathlib.PurePosixPath("benchmark", dataset, file)
                local_file = local_ds_root.joinpath(file)

                if (not local_file.exists()) or (not self.validate_file(remote_file, local_file)):
                    log.info(f"local file: {local_file} not match with remote: {remote_file}; add to downloading list")
                    downloads.append((remote_file, local_file))

        if len(downloads) == 0:
            return

        log.info(f"Start to downloading files, total count: {len(downloads)}")
        for remote_file, local_file in tqdm(downloads):
            log.debug(f"downloading file {remote_file} to {local_file}")
            self.bucket.get_object_to_file(remote_file.as_posix(), local_file.absolute())

        log.info(f"Succeed to download all files, downloaded file count = {len(downloads)}")


class AwsS3Reader(DatasetReader):
    source: DatasetSource = DatasetSource.S3
    remote_root: str = config.AWS_S3_URL

    def __init__(self):
        import s3fs

        self.fs = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": "us-west-2"})

    def ls_all(self, dataset: str):
        dataset_root_dir = pathlib.Path(self.remote_root, dataset)
        log.info(f"listing dataset: {dataset_root_dir}")
        names = self.fs.ls(dataset_root_dir)
        for n in names:
            log.info(n)
        return names

    def read(self, dataset: str, files: list[str], local_ds_root: pathlib.Path):
        downloads = []
        if not local_ds_root.exists():
            log.info(f"local dataset root path not exist, creating it: {local_ds_root}")
            local_ds_root.mkdir(parents=True)
            downloads = [pathlib.PurePosixPath(self.remote_root, dataset, f) for f in files]

        else:
            for file in files:
                remote_file = pathlib.PurePosixPath(self.remote_root, dataset, file)
                local_file = local_ds_root.joinpath(file)

                if (not local_file.exists()) or (not self.validate_file(remote_file, local_file)):
                    log.info(f"local file: {local_file} not match with remote: {remote_file}; add to downloading list")
                    downloads.append(remote_file)

        if len(downloads) == 0:
            return

        log.info(f"Start to downloading files, total count: {len(downloads)}")
        for s3_file in tqdm(downloads):
            log.debug(f"downloading file {s3_file} to {local_ds_root}")
            self.fs.download(s3_file, local_ds_root.as_posix())

        log.info(f"Succeed to download all files, downloaded file count = {len(downloads)}")

    def validate_file(self, remote: pathlib.Path, local: pathlib.Path) -> bool:
        # info() uses ls() inside, maybe we only need to ls once
        info = self.fs.info(remote)

        # check size equal
        remote_size, local_size = info.get("size"), local.stat().st_size
        if remote_size != local_size:
            log.info(f"local file: {local} size[{local_size}] not match with remote size[{remote_size}]")
            return False

        return True


class IRDatasetsReader(DatasetReader):
    """Reader for ir_datasets based datasets"""

    source: DatasetSource = DatasetSource.IR_DATASETS
    remote_root: str = ""  # Not used for ir_datasets

    def __init__(self):
        self.ir_datasets = ir_datasets

    def read(self, dataset: str, files: list[str], local_ds_root: pathlib.Path):
        """
        Download FTS dataset using ir_datasets API

        Args:
            dataset: ir_datasets dataset name
            files: Expected output files (ignored, not used)
            local_ds_root: Local directory (not used, ir_datasets handles its own cache)
        """
        log.info(f"Downloading FTS dataset '{dataset}' using ir_datasets")

        try:
            # Load dataset using ir_datasets - this will download if needed
            # ir_datasets handles caching automatically
            # Actual data download happens lazily when iterating
            self.ir_datasets.load(dataset)
            log.info(f"Successfully loaded dataset: {dataset}")

        except Exception:
            log.exception(f"Failed to download FTS dataset '{dataset}'")
            raise

    def validate_file(self, remote: pathlib.Path, local: pathlib.Path) -> bool:
        """For ir_datasets, we don't validate against remote files"""
        # ir_datasets handles its own caching and validation
        return local.exists() and local.stat().st_size > 0
