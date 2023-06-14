import environs
import inspect
import pathlib
from . import log_util


env = environs.Env()
env.read_env(".env")

class config:
    LOG_LEVEL = env.str("LOG_LEVEL", "INFO")

    DEFAULT_DATASET_URL = env.str("DEFAULT_DATASET_URL", "assets.zilliz.com/benchmark/")
    DATASET_LOCAL_DIR = env.path("DATASET_LOCAL_DIR", "/tmp/vector_db_bench/dataset")
    NUM_PER_BATCH = env.int("NUM_PER_BATCH", 5000)

    DROP_OLD = env.bool("DROP_OLD", True)
    USE_SHUFFLED_DATA = env.bool("USE_SHUFFLED_DATA", True)

    RESULTS_LOCAL_DIR = pathlib.Path(__file__).parent.joinpath("results")
    CASE_TIMEOUT_IN_SECOND = 24 * 60 * 60


    def display(self) -> str:
        tmp = [i for i in inspect.getmembers(self)
            if not inspect.ismethod(i[1]) and not i[0].startswith('_') \
        ]
        return tmp

log_util.init(config.LOG_LEVEL)
