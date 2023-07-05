import environs
import inspect
import pathlib
from . import log_util


env = environs.Env()
env.read_env(".env")

class config:
    LOG_LEVEL = env.str("LOG_LEVEL", "INFO")

    DEFAULT_DATASET_URL = env.str("DEFAULT_DATASET_URL", "assets.zilliz.com/benchmark/")
    DATASET_LOCAL_DIR = env.path("DATASET_LOCAL_DIR", "/tmp/vectordb_bench/dataset")
    NUM_PER_BATCH = env.int("NUM_PER_BATCH", 5000)

    DROP_OLD = env.bool("DROP_OLD", True)
    USE_SHUFFLED_DATA = env.bool("USE_SHUFFLED_DATA", True)

    RESULTS_LOCAL_DIR = pathlib.Path(__file__).parent.joinpath("results")

    CAPACITY_TIMEOUT_IN_SECONDS =  24 * 3600 # 24h
    LOAD_TIMEOUT_1M             = 2.5 * 3600 # 2.5h
    LOAD_TIMEOUT_10M            =  25 * 3600 # 25h
    LOAD_TIMEOUT_100M           = 250 * 3600 # 10.41d

    OPTIMIZE_TIMEOUT_1M         =  15 * 60   # 15min
    OPTIMIZE_TIMEOUT_10M        = 2.5 * 3600 # 2.5h
    OPTIMIZE_TIMEOUT_100M       =  25 * 3600 # 1.04d


    def display(self) -> str:
        tmp = [
            i for i in inspect.getmembers(self)
            if not inspect.ismethod(i[1])
            and not i[0].startswith('_')
            and "TIMEOUT" not in i[0]
        ]
        return tmp

log_util.init(config.LOG_LEVEL)
