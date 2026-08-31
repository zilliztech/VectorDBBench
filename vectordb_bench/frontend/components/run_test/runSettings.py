from vectordb_bench import config
from vectordb_bench.backend.cases import CaseType
from vectordb_bench.models import CaseConfig

DEFAULT_INSERT_BATCH_SIZE = config.DEFAULT_INSERT_BATCH_SIZE
DEFAULT_STREAMING_INSERT_RATE = config.DEFAULT_STREAMING_INSERT_RATE
MAX_STREAMLIT_INT = (1 << 53) - 1
STREAMING_CASE_TYPES = {
    CaseType.StreamingPerformanceCase,
    CaseType.StreamingCustomDataset,
}


def validate_streaming_insert_rates(
    activedCaseList: list[CaseConfig],
    batch_size: int,
) -> tuple[bool, list[str]]:
    errors = []
    for case_config in activedCaseList:
        if case_config.case_id not in STREAMING_CASE_TYPES:
            continue

        custom_case = case_config.custom_case or {}
        insert_rate = custom_case.get("insert_rate", DEFAULT_STREAMING_INSERT_RATE)
        case_name = case_config.case_id.name
        if insert_rate < batch_size:
            errors.append(
                f"{case_name}: Streaming Insert Rate ({insert_rate}) must be greater than or equal to "
                f"Insert Batch Size ({batch_size})."
            )
        elif insert_rate % batch_size != 0:
            errors.append(
                f"{case_name}: Streaming Insert Rate ({insert_rate}) must be divisible by "
                f"Insert Batch Size ({batch_size})."
            )

    return len(errors) == 0, errors


def runSettings(container, activedCaseList: list[CaseConfig]) -> tuple[int, bool]:
    container.markdown(
        "<div style='height: 24px;'></div>",
        unsafe_allow_html=True,
    )
    container.subheader("Run Settings")
    batch_size = container.number_input(
        "Insert Batch Size",
        min_value=1,
        max_value=MAX_STREAMLIT_INT,
        value=DEFAULT_INSERT_BATCH_SIZE,
        step=100,
        help="Rows or documents in each logical VDBBench insert batch. Backends may split it further.",
    )

    is_valid, errors = validate_streaming_insert_rates(activedCaseList, batch_size)
    for error in errors:
        container.error(error)

    return batch_size, is_valid
