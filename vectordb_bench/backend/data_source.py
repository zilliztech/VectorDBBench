import logging
import pathlib
import typing
from enum import Enum
from tqdm import tqdm
from hashlib import md5
import os
from abc import ABC, abstractmethod

from .. import config

logging.getLogger("s3fs").setLevel(logging.CRITICAL)

log = logging.getLogger(__name__)

DatasetReader = typing.TypeVar("DatasetReader")

class DatasetSource(Enum):
    S3 = "S3"
    AliyunOSS = "AliyunOSS"

    def reader(self) -> DatasetReader:
        if self == DatasetSource.S3:
            return AwsS3Reader()

        if self == DatasetSource.AliyunOSS:
            return AliyunOSSReader()


class DatasetReader(ABC):
    source: DatasetSource
    remote_root: str

    @abstractmethod
    def read(self, dataset: str, files: list[str], local_ds_root: pathlib.Path, check_etag: bool = True):
        """read dataset files from remote_root to local_ds_root,

        Args:
            dataset(str): for instance "sift_small_500k"
            files(list[str]):  all filenames of the dataset
            local_ds_root(pathlib.Path): whether to write the remote data.
            check_etag(bool): whether to check the etag
        """
        pass

    @abstractmethod
    def validate_file(self, remote: pathlib.Path, local: pathlib.Path) -> bool:
        pass


class AliyunOSSReader(DatasetReader):
    source: DatasetSource = DatasetSource.AliyunOSS
    remote_root: str = config.ALIYUN_OSS_URL

    def __init__(self):
        import oss2
        self.bucket = oss2.Bucket(oss2.AnonymousAuth(), self.remote_root, "benchmark", True)

    def validate_file(self, remote: pathlib.Path, local: pathlib.Path, check_etag: bool) -> bool:
        info = self.bucket.get_object_meta(remote.as_posix())

        # check size equal
        remote_size, local_size = info.content_length, os.path.getsize(local)
        if remote_size != local_size:
            log.info(f"local file: {local} size[{local_size}] not match with remote size[{remote_size}]")
            return False

        # check etag equal
        if check_etag:
            return match_etag(info.etag.strip('"').lower(), local)

        return True

    def read(self, dataset: str, files: list[str], local_ds_root: pathlib.Path, check_etag: bool = False):
        downloads = []
        if not local_ds_root.exists():
            log.info(f"local dataset root path not exist, creating it: {local_ds_root}")
            local_ds_root.mkdir(parents=True)
            downloads = [(pathlib.Path("benchmark", dataset, f), local_ds_root.joinpath(f)) for f in files]

        else:
            for file in files:
                remote_file = pathlib.Path("benchmark", dataset, file)
                local_file = local_ds_root.joinpath(file)

                # Don't check etags for Dataset from Aliyun OSS
                if (not local_file.exists()) or (not self.validate_file(remote_file, local_file, False)):
                    log.info(f"local file: {local_file} not match with remote: {remote_file}; add to downloading list")
                    downloads.append((remote_file, local_file))

        if len(downloads) == 0:
            return

        log.info(f"Start to downloading files, total count: {len(downloads)}")
        for remote_file, local_file in tqdm(downloads):
            log.debug(f"downloading file {remote_file} to {local_ds_root}")
            self.bucket.get_object_to_file(remote_file.as_posix(), local_file.as_posix())

        log.info(f"Succeed to download all files, downloaded file count = {len(downloads)}")



class AwsS3Reader(DatasetReader):
    source: DatasetSource = DatasetSource.S3
    remote_root: str = config.AWS_S3_URL

    def __init__(self):
        import s3fs
        self.fs = s3fs.S3FileSystem(
            anon=True,
            client_kwargs={'region_name': 'us-west-2'}
        )

    def ls_all(self, dataset: str):
        dataset_root_dir = pathlib.Path(self.remote_root, dataset)
        log.info(f"listing dataset: {dataset_root_dir}")
        names = self.fs.ls(dataset_root_dir)
        for n in names:
            log.info(n)
        return names


    def read(self, dataset: str, files: list[str], local_ds_root: pathlib.Path, check_etag: bool = True):
        downloads = []
        if not local_ds_root.exists():
            log.info(f"local dataset root path not exist, creating it: {local_ds_root}")
            local_ds_root.mkdir(parents=True)
            downloads = [pathlib.Path(self.remote_root, dataset, f) for f in files]

        else:
            for file in files:
                remote_file = pathlib.Path(self.remote_root, dataset, file)
                local_file = local_ds_root.joinpath(file)

                if (not local_file.exists()) or (not self.validate_file(remote_file, local_file, check_etag)):
                    log.info(f"local file: {local_file} not match with remote: {remote_file}; add to downloading list")
                    downloads.append(remote_file)

        if len(downloads) == 0:
            return

        log.info(f"Start to downloading files, total count: {len(downloads)}")
        for s3_file in tqdm(downloads):
            log.debug(f"downloading file {s3_file} to {local_ds_root}")
            self.fs.download(s3_file, local_ds_root.as_posix())

        log.info(f"Succeed to download all files, downloaded file count = {len(downloads)}")


    def validate_file(self, remote: pathlib.Path, local: pathlib.Path, check_etag: bool) -> bool:
        # info() uses ls() inside, maybe we only need to ls once
        info = self.fs.info(remote)

        # check size equal
        remote_size, local_size = info.get("size"), os.path.getsize(local)
        if remote_size != local_size:
            log.info(f"local file: {local} size[{local_size}] not match with remote size[{remote_size}]")
            return False

        # check etag equal
        if check_etag:
            return match_etag(info.get('ETag', "").strip('"'), local)

        return True


def match_etag(expected_etag: str, local_file) -> bool:
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
