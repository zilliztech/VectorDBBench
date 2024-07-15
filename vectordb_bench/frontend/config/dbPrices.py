from vectordb_bench import config
import ujson
import pathlib

with open(pathlib.Path(config.RESULTS_LOCAL_DIR, "dbPrices.json")) as f:
    DB_DBLABEL_TO_PRICE = ujson.load(f)
