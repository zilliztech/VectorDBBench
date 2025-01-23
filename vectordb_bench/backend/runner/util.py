import logging

import numpy as np
from pandas import DataFrame

log = logging.getLogger(__name__)


def get_data(data_df: DataFrame, normalize: bool) -> tuple[list[list[float]], list[str]]:
    all_metadata = data_df["id"].tolist()
    emb_np = np.stack(data_df["emb"])
    if normalize:
        log.debug("normalize the 100k train data")
        all_embeddings = (emb_np / np.linalg.norm(emb_np, axis=1)[:, np.newaxis]).tolist()
    else:
        all_embeddings = emb_np.tolist()
    return all_embeddings, all_metadata
