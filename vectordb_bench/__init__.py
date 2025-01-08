import inspect
import pathlib

import environs

from . import log_util

env = environs.Env()
env.read_env(".env", False)


class config:
    ALIYUN_OSS_URL = "assets.zilliz.com.cn/benchmark/"
    AWS_S3_URL = "assets.zilliz.com/benchmark/"

    LOG_LEVEL = env.str("LOG_LEVEL", "INFO")

    DEFAULT_DATASET_URL = env.str("DEFAULT_DATASET_URL", AWS_S3_URL)
    DATASET_LOCAL_DIR = env.path("DATASET_LOCAL_DIR", "/tmp/vectordb_bench/dataset")
    NUM_PER_BATCH = env.int("NUM_PER_BATCH", 100)

    DROP_OLD = env.bool("DROP_OLD", True)
    USE_SHUFFLED_DATA = env.bool("USE_SHUFFLED_DATA", True)

    NUM_CONCURRENCY = env.list(
        "NUM_CONCURRENCY",
        [
            1,
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            40,
            45,
            50,
            55,
            60,
            65,
            70,
            75,
            80,
            85,
            90,
            95,
            100,
        ],
        subcast=int,
    )

    CONCURRENCY_DURATION = 30

    RESULTS_LOCAL_DIR = env.path(
        "RESULTS_LOCAL_DIR",
        pathlib.Path(__file__).parent.joinpath("results"),
    )
    CONFIG_LOCAL_DIR = env.path(
        "CONFIG_LOCAL_DIR",
        pathlib.Path(__file__).parent.joinpath("config-files"),
    )

    K_DEFAULT = 100  # default return top k nearest neighbors during search
    CUSTOM_CONFIG_DIR = pathlib.Path(__file__).parent.joinpath("custom/custom_case.json")

    CAPACITY_TIMEOUT_IN_SECONDS = 24 * 3600  # 24h
    LOAD_TIMEOUT_DEFAULT = 24 * 3600  # 24h
    LOAD_TIMEOUT_768D_1M = 24 * 3600  # 24h
    LOAD_TIMEOUT_768D_10M = 240 * 3600  # 10d
    LOAD_TIMEOUT_768D_100M = 2400 * 3600  # 100d

    LOAD_TIMEOUT_1536D_500K = 24 * 3600  # 24h
    LOAD_TIMEOUT_1536D_5M = 240 * 3600  # 10d

    OPTIMIZE_TIMEOUT_DEFAULT = 24 * 3600  # 24h
    OPTIMIZE_TIMEOUT_768D_1M = 24 * 3600  # 24h
    OPTIMIZE_TIMEOUT_768D_10M = 240 * 3600  # 10d
    OPTIMIZE_TIMEOUT_768D_100M = 2400 * 3600  # 100d

    OPTIMIZE_TIMEOUT_1536D_500K = 24 * 3600  # 24h
    OPTIMIZE_TIMEOUT_1536D_5M = 240 * 3600  # 10d

    def display(self) -> str:
        return [
            i
            for i in inspect.getmembers(self)
            if not inspect.ismethod(i[1]) and not i[0].startswith("_") and "TIMEOUT" not in i[0]
        ]


log_util.init(config.LOG_LEVEL)
