import time
import concurrent
import multiprocessing as mp
import logging
from typing import Type
import pandas as pd
import numpy as np
from ..clients import api

log = logging.getLogger(__name__)

NUM_PER_BATCH = 5000


class MultiProcessingInsertRunner:
    def __init__(self, db_class: Type[api.VectorDB], df_train: pd.DataFrame):
        self.db = db_class
        self.batches = np.array_split(df_train, df_train.shape[0]/NUM_PER_BATCH)
        log.debug(f"{self.batches[0]}")

        self.batch_ids = [i for i in range(len(self.batches))]

    def insert_data(self, batch_id: int, batch: pd.DataFrame):
        db = self.db()
        log.debug(f"({mp.current_process().name})Batch No.{batch_id:3}: Start inserting {batch.shape[0]} embeddings")
        embeddings, metadatas = self.get_embedding_with_meta(batch)

        insert_results = db.insert_embeddings(
            embeddings=embeddings,
            metadatas=None,
        )
        assert len(insert_results) == batch.shape[0]
        log.debug(f"({mp.current_process().name})Batch No.{batch_id:3}: Finish inserting embeddings")

    def get_embedding_with_meta(self, df_data: pd.DataFrame) -> (list[list[float]], list[dict]):
        # TODO
        metadata = []
        embeddings = []

        for col in df_data.columns:
            if col == 'emb': #Cohere
                embeddings = df_data[col].to_list()
            #  else:
            #      field = {col: df_data[col].to_list()}
            #      metadata.append(field)
        return embeddings, metadata

    def _insert_all_batches_sequentially(self) -> list[int]:
        results = []
        for i in self.batch_ids:
            results.append(self.insert_data(i, self.batches[i]))
        return results

    def _insert_all_batches(self) -> list[int]:
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
            future_iter = executor.map(self.insert_data, self.batch_ids, self.batches)
            results = [r for r in future_iter]
        return results


    def run_sequentially(self) -> list[int]:
        start_time = time.time()
        results = self._insert_all_batches_sequentially()
        duration = time.time() - start_time
        log.info(f'Sequentially inserted {len(self.batch_ids)} batches of {NUM_PER_BATCH} entities in {duration} seconds')
        return results

    def run(self) -> list[int]:
        start_time = time.time()
        results = self._insert_all_batches()
        duration = time.time() - start_time
        log.info(f'Sequentially inserted {len(self.batch_ids)} batches of {NUM_PER_BATCH} entities in {duration} seconds')
        return results
