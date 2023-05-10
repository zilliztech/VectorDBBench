import environs
from . import config

env = environs.Env()
env.read_env(".env")

LOG_LEVEL = env.str("LOG_LEVEL", "DEBUG")
LOG_PATH = env.str("LOG_PATH", '/tmp/falcon_mark')
LOG_NAME = env.str("LOG_NAME", 'logfile')
TIMEZONE = env.str("TIMEZONE", 'UTC')

DEFAULT_DATASET_URL = env.str("DEFAULT_DATASET_URL", "") # TODO
DATASET_LOCAL_DIR = env.path("DATASET_LOCAL_DIR", "/tmp/falcon_mark/dataset")

config.init(LOG_LEVEL, LOG_PATH, LOG_NAME, TIMEZONE)
