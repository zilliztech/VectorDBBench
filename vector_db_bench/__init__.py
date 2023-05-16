import os
import environs
from . import config

env = environs.Env()
env.read_env(".env")

LOG_LEVEL = env.str("LOG_LEVEL", "DEBUG")
LOG_PATH = env.str("LOG_PATH", '/tmp/vector_db_bench')
LOG_NAME = env.str("LOG_NAME", 'logfile')
TIMEZONE = env.str("TIMEZONE", 'UTC')

DEFAULT_DATASET_URL = env.str("DEFAULT_DATASET_URL", "assets.zilliz.com/benchmark/")
DATASET_LOCAL_DIR = env.path("DATASET_LOCAL_DIR", "/tmp/vector_db_bench/dataset")
RESULTS_LOCAL_DIR = env.path("RESULTS_LOCAL_DIR", "/tmp/vector_db_bench/results")

config.init(LOG_LEVEL, LOG_PATH, LOG_NAME, TIMEZONE)
