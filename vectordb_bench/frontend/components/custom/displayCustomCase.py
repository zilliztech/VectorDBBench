from vectordb_bench.frontend.components.custom.getCustomConfig import CustomCaseConfig


def displayCustomCase(customCase: CustomCaseConfig, st, key):

    columns = st.columns([1, 2])
    customCase.dataset_config.name = columns[0].text_input(
        "Name", key=f"{key}_name", value=customCase.dataset_config.name
    )
    customCase.name = f"{customCase.dataset_config.name} (Performace Case)"
    customCase.dataset_config.dir = columns[1].text_input(
        "Folder Path", key=f"{key}_dir", value=customCase.dataset_config.dir
    )

    columns = st.columns(4)
    customCase.dataset_config.dim = columns[0].number_input(
        "dim", key=f"{key}_dim", value=customCase.dataset_config.dim
    )
    customCase.dataset_config.size = columns[1].number_input(
        "size", key=f"{key}_size", value=customCase.dataset_config.size
    )
    customCase.dataset_config.metric_type = columns[2].selectbox(
        "metric type", key=f"{key}_metric_type", options=["L2", "Cosine", "IP"]
    )
    customCase.dataset_config.file_count = columns[3].number_input(
        "train file count",
        key=f"{key}_file_count",
        value=customCase.dataset_config.file_count,
        help="if train file count is more than one, please input all your train file name and split with ','",
    )

    columns = st.columns(3)
    customCase.dataset_config.train_name = columns[0].text_input(
        "train file name",
        key=f"{key}_train_name",
        value=customCase.dataset_config.train_name,
        help="if your file and column in the file is not named as previous explanation, please input the real name (for example: if the file name is `tr.parquet` and column name is `embbb`, then input tr and embbb)",
    )
    customCase.dataset_config.test_name = columns[1].text_input(
        "test file name", key=f"{key}_test_name", value=customCase.dataset_config.test_name
    )
    customCase.dataset_config.gt_name = columns[2].text_input(
        "ground truth file name", key=f"{key}_gt_name", value=customCase.dataset_config.gt_name
    )

    columns = st.columns([1, 1, 2, 2])
    customCase.dataset_config.train_id_name = columns[0].text_input(
        "train id name", key=f"{key}_train_id_name", value=customCase.dataset_config.train_id_name
    )
    customCase.dataset_config.train_col_name = columns[1].text_input(
        "train emb name", key=f"{key}_train_col_name", value=customCase.dataset_config.train_col_name
    )
    customCase.dataset_config.test_col_name = columns[2].text_input(
        "test emb name", key=f"{key}_test_col_name", value=customCase.dataset_config.test_col_name
    )
    customCase.dataset_config.gt_col_name = columns[3].text_input(
        "ground truth emb name", key=f"{key}_gt_col_name", value=customCase.dataset_config.gt_col_name
    )

    columns = st.columns(4)
    customCase.dataset_config.use_shuffled = columns[0].checkbox(
        "use shuffled data", key=f"{key}_use_shuffled", value=customCase.dataset_config.use_shuffled
    )
    customCase.dataset_config.with_gt = columns[1].checkbox(
        "with groundtruth", key=f"{key}_with_gt", value=customCase.dataset_config.with_gt
    )

    customCase.description = st.text_area("description", key=f"{key}_description", value=customCase.description)
