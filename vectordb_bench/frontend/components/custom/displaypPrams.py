def displayParams(st):
    st.markdown(
        """
- `Folder Path` - The path to the folder containing all the files. Please ensure that all files in the folder are in the `Parquet` format.
  - Vectors data files: The file should have two kinds of columns: `id` as an incrementing `int` and `emb` as an array of `float32`. The name of two columns could be defined on your own.
  - Query test vectors: The file could be named on your own and should have two kinds of columns: `id` as an incrementing `int` and `emb` as an array of `float32`. The `id` column must be named as `id`, and `emb` column could be defined on your own.  
  - Ground truth file: The file could be named on your own and should have two kinds of columns: `id` corresponding to query vectors and `neighbors_id` as an array of `int`. The `id` column must be named as `id`, and `neighbors_id` column could be defined on your own.

- `Train File Name` - If the number of train file is `more than one`, please input all your train file name and `split with ','` without the `.parquet` file extensionthe. For example, if there are two train file and the name of them are `train1.parquet` and `train2.parquet`, then input `train1,train2`.

- `Ground Truth Emb Name` - No matter whether filter file is applied or not, the `neighbors_id` column in ground truth file must have the same name.

- `Scalar Labels File Name ` - If there is a scalar labels file, please input the filename without the .parquet extension. The file should have two columns: `id` as an incrementing `int` and `labels` as an array of `string`. The `id` column must correspond one-to-one with the `id` column in train file..

- `Label percentages` - If you have filter file, please input label percentage you want to real run and `split with ','` when it's `more than one`. If you `don't have` filter file, than `keep the text vacant.`

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
