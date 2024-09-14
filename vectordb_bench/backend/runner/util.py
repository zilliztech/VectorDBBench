import logging
import concurrent
from typing import Iterable

from pandas import DataFrame
import numpy as np

log = logging.getLogger(__name__)

def get_data(data_df: DataFrame, normalize: bool) -> tuple[list[list[float]], list[str]]:
    all_metadata = data_df['id'].tolist()
    emb_np = np.stack(data_df['emb'])
    if normalize:
        log.debug("normalize the 100k train data")
        all_embeddings = (emb_np / np.linalg.norm(emb_np, axis=1)[:, np.newaxis]).tolist()
    else:
        all_embeddings = emb_np.tolist()
    return all_embeddings, all_metadata

def is_futures_completed(futures: Iterable[concurrent.futures.Future], interval) -> (Exception, bool):
    try:
        list(concurrent.futures.as_completed(futures, timeout=interval))
    except TimeoutError as e:
        return e, False
    return None, True


def get_future_exceptions(futures: Iterable[concurrent.futures.Future]) -> BaseException | None:
    for f in futures:
        if f.exception() is not None:
            return f.exception()
    return
