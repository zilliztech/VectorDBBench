import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import argparse

from datasets import load_dataset

import faiss

def get_args():
    parser = argparse.ArgumentParser(
        description="Prepare dataset and ground truth neighbors for benchmarking."
    )
    parser.add_argument(
        "-d", "--dataset-name",
        type=str,
        default="cryptolab-playground/pubmed-arxiv-abstract-embedding-gemma-300m",
        help="Huggingface dataset name to download.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="./dataset/PUBMED768D400K",
        help="Dataset directory to save the dataset and neighbors.",
    )
    return parser.parse_args()

def download_dataset(
    dataset_name: str, 
    output_dir: str = "./dataset"
) -> None:
    """Download dataset from Huggingface and save as Parquet files."""
    # load dataset
    ds = load_dataset(dataset_name)
    train = ds["train"].to_pandas()
    test = ds["test"].to_pandas()
    
    # write to parquet
    train_table = pa.Table.from_pandas(train)
    pq.write_table(train_table, f"{output_dir}/train.parquet")

    test_table = pa.Table.from_pandas(test)
    pq.write_table(test_table, f"{output_dir}/test.parquet")

def prepare_neighbors(
    data_dir: str = "./dataset/PUBMED768D400K",
) -> None:
    """Prepare ground truth neighbors using brute-force flat search and save as Parquet."""
    # load dataset
    train = pd.read_parquet(f"{data_dir}/train.parquet")
    test = pd.read_parquet(f"{data_dir}/test.parquet")

    train = np.stack(train["emb"].to_list()).astype("float32")
    test = np.stack(test["emb"].to_list()).astype("float32")
    dim = train.shape[1]

    # flat search
    index = faiss.IndexFlatIP(dim)
    index.add(train)

    k = len(test)
    distances, indices = index.search(test, k)
    print(distances.shape, indices.shape)

    # save flat search result as neighbors
    df = pd.DataFrame({
        "id": np.arange(len(indices)),
        "neighbors_id": indices.tolist()
    })
    
    table = pa.Table.from_pandas(df)
    pq.write_table(table, f"{data_dir}/neighbors.parquet")

if __name__ == "__main__":
    args = get_args()

    # download_dataset(args.dataset_name, args.dataset_dir)
    prepare_neighbors(args.dataset_dir)
