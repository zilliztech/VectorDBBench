def displayParams(st):
    st.markdown(
        """
- `Folder Path` - The path to the folder containing all the files. Please ensure that all files in the folder are in the `Parquet` format.
  - Vectors data files: The file must be named `train.parquet` and should have two columns: `id` as an incrementing `int` and `emb` as an array of `float32`.
  - Query test vectors: The file must be named `test.parquet` and should have two columns: `id` as an incrementing `int` and `emb` as an array of `float32`. 
  - Ground truth file: The file must be named `neighbors.parquet` and should have two columns: `id` corresponding to query vectors and `neighbors_id` as an array of `int`.

- `Train File Count` - If the vector file is too large, you can consider splitting it into multiple files. The naming format for the split files should be `train-[index]-of-[file_count].parquet`. For example, `train-01-of-10.parquet` represents the second file (0-indexed) among 10 split files.

- `Use Shuffled Data` - If you check this option, the vector data files need to be modified. VectorDBBench will load the data labeled with `shuffle`. For example, use `shuffle_train.parquet` instead of `train.parquet` and `shuffle_train-04-of-10.parquet` instead of `train-04-of-10.parquet`. The `id` column in the shuffled data can be in any order.
"""
    )
    st.caption(
        """We recommend limiting the number of test query vectors, like 1,000.""",
        help="""
When conducting concurrent query tests, Vdbbench creates a large number of processes. 
To minimize additional communication overhead during testing, 
we prepare a complete set of test queries for each process, allowing them to run independently.\n
However, this means that as the number of concurrent processes increases, 
the number of copied query vectors also increases significantly, 
which can place substantial pressure on memory resources.
""",
    )
