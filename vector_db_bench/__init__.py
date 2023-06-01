import environs
import pathlib
from . import config

env = environs.Env()
env.read_env(".env")

LOG_LEVEL = env.str("LOG_LEVEL", "INFO")
LOG_PATH = env.str("LOG_PATH", '/tmp/vector_db_bench')
LOG_NAME = env.str("LOG_NAME", 'logfile')
TIMEZONE = env.str("TIMEZONE", 'UTC')

DEFAULT_DATASET_URL = env.str("DEFAULT_DATASET_URL", "assets.zilliz.com/benchmark/")
DATASET_LOCAL_DIR = env.path("DATASET_LOCAL_DIR", "/tmp/vector_db_bench/dataset")
NUM_PER_BATCH = env.int("NUM_PER_BATCH", 5000)

DROP_OLD = env.bool("DROP_OLD", True)

config.init(LOG_LEVEL, LOG_PATH, LOG_NAME, TIMEZONE)

RESULTS_LOCAL_DIR = pathlib.Path(__file__).parent.joinpath("results")
