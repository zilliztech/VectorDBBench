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
        "train file count", key=f"{key}_file_count", value=customCase.dataset_config.file_count
    )

    columns = st.columns(4)
    customCase.dataset_config.use_shuffled = columns[0].checkbox(
        "use shuffled data", key=f"{key}_use_shuffled", value=customCase.dataset_config.use_shuffled
    )
    customCase.dataset_config.with_gt = columns[1].checkbox(
        "with groundtruth", key=f"{key}_with_gt", value=customCase.dataset_config.with_gt
    )

    customCase.description = st.text_area("description", key=f"{key}_description", value=customCase.description)
