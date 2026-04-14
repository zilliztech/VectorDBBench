from vectordb_bench.frontend.components.custom.getCustomConfig import CustomStreamingCaseConfig


def displayCustomStreamingCase(streamingCase: CustomStreamingCaseConfig, st, key):

    columns = st.columns([1, 2])
    streamingCase.dataset_config.name = columns[0].text_input(
        "Name", key=f"{key}_name", value=streamingCase.dataset_config.name
    )
    streamingCase.dataset_config.dir = columns[1].text_input(
        "Folder Path", key=f"{key}_dir", value=streamingCase.dataset_config.dir
    )

    columns = st.columns(2)
    streamingCase.dataset_config.dim = columns[0].number_input(
        "dim", key=f"{key}_dim", value=streamingCase.dataset_config.dim
    )
    streamingCase.dataset_config.size = columns[1].number_input(
        "size", key=f"{key}_size", value=streamingCase.dataset_config.size
    )

    columns = st.columns(3)
    streamingCase.dataset_config.train_name = columns[0].text_input(
        "train file name",
        key=f"{key}_train_name",
        value=streamingCase.dataset_config.train_name,
    )
    streamingCase.dataset_config.test_name = columns[1].text_input(
        "test file name", key=f"{key}_test_name", value=streamingCase.dataset_config.test_name
    )
    streamingCase.dataset_config.gt_name = columns[2].text_input(
        "ground truth file name", key=f"{key}_gt_name", value=streamingCase.dataset_config.gt_name
    )

    columns = st.columns([1, 1, 2, 2])
    streamingCase.dataset_config.train_id_name = columns[0].text_input(
        "train id name", key=f"{key}_train_id_name", value=streamingCase.dataset_config.train_id_name
    )
    streamingCase.dataset_config.train_col_name = columns[1].text_input(
        "train emb name", key=f"{key}_train_col_name", value=streamingCase.dataset_config.train_col_name
    )
    streamingCase.dataset_config.test_col_name = columns[2].text_input(
        "test emb name", key=f"{key}_test_col_name", value=streamingCase.dataset_config.test_col_name
    )
    streamingCase.dataset_config.gt_col_name = columns[3].text_input(
        "ground truth emb name", key=f"{key}_gt_col_name", value=streamingCase.dataset_config.gt_col_name
    )

    streamingCase.description = st.text_area("description", key=f"{key}_description", value=streamingCase.description)
