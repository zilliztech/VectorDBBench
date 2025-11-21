import faiss
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def main():
    # load dataset
    data_dir = "/data/vectordb_bench/dataset/PUBMED768D400K"
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
    main()