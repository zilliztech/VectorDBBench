from unittest.mock import Mock

import pytest

from vectordb_bench.backend.cases import CaseLabel, CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import IndexType
from vectordb_bench.backend.clients.zvec.config import ZvecConfig, ZvecDiskANNIndexConfig, ZvecHNSWIndexConfig
from vectordb_bench.frontend.components.run_test.caseSelector import dbCaseConfigSetting
from vectordb_bench.frontend.components.run_test.generateTasks import generate_tasks
from vectordb_bench.frontend.config.dbCaseConfigs import MAX_STREAMLIT_INT, UICaseItem
from vectordb_bench.models import CaseConfig, CaseConfigParamType, TaskConfig


@pytest.fixture(
    params=[CaseType.CapacityDim128, CaseType.Performance768D10M, CaseType.StreamingPerformanceCase],
    ids=["load", "performance", "streaming"],
)
def ui_case(request: pytest.FixtureRequest) -> CaseConfig:
    return CaseConfig(case_id=request.param)


def render_zvec_ui(
    case: CaseConfig,
    values: dict[str, object] | None = None,
    config: dict[CaseConfigParamType, object] | None = None,
) -> tuple[dict[CaseConfigParamType, object], dict[str, dict[str, object]]]:
    values = {} if values is None else values
    config = {} if config is None else config
    inputs: dict[str, dict[str, object]] = {}
    container = Mock()
    container.columns.side_effect = lambda count: [container] * count

    def selectbox(_label, options, **kwargs):
        name = kwargs["key"].rsplit("-", 1)[-1]
        inputs[name] = {"options": options, **kwargs}
        value = values.get(name, options[kwargs.get("index", 0)])
        assert value in options
        return value

    def number_input(_label, **kwargs):
        name = kwargs["key"].rsplit("-", 1)[-1]
        inputs[name] = kwargs
        value = values.get(name, kwargs["value"])
        assert isinstance(value, int)
        assert kwargs["min_value"] <= value <= kwargs["max_value"]
        return value

    container.selectbox.side_effect = selectbox
    container.number_input.side_effect = number_input
    item = UICaseItem(cases=[case], label="Zvec test", caseLabel=case.case.label)
    dbCaseConfigSetting(container, {DB.Zvec: {item: config}}, item, [DB.Zvec])
    container.write.assert_not_called()
    return config, inputs


def task_from_ui(case: CaseConfig, config: dict[CaseConfigParamType, object]) -> TaskConfig:
    tasks = generate_tasks(
        [DB.Zvec],
        {DB.Zvec: ZvecConfig(db_label="ui-test", path="unused")},
        [case],
        {DB.Zvec: {case: config}},
        batch_size=1,
    )
    assert len(tasks) == 1
    return tasks[0]


def test_zvec_ui_defaults_to_hnsw(ui_case: CaseConfig) -> None:
    config, inputs = render_zvec_ui(ui_case)

    assert inputs["IndexType"]["options"] == [IndexType.HNSW.value, IndexType.DISKANN.value]
    assert config[CaseConfigParamType.IndexType] == IndexType.HNSW.value
    expected_inputs = {"IndexType", "M", "ef_construction", "quantize_type"}
    if ui_case.case.label != CaseLabel.Load:
        expected_inputs.update({"ef_search", "is_using_refiner"})
    assert set(inputs) == expected_inputs
    task = task_from_ui(ui_case, config)
    assert isinstance(task.db_case_config, ZvecHNSWIndexConfig)
    assert task.db_case_config == ZvecHNSWIndexConfig()


def test_zvec_ui_diskann_parameters_reach_task(ui_case: CaseConfig) -> None:
    config, inputs = render_zvec_ui(
        ui_case,
        {
            "IndexType": IndexType.DISKANN.value,
            "max_degree": 32,
            "build_list_size": 80,
            "pq_chunk_num": 32,
            "search_list_size": 600,
        },
    )

    expected_inputs = {"IndexType", "max_degree", "build_list_size", "pq_chunk_num"}
    if ui_case.case.label != CaseLabel.Load:
        expected_inputs.add("search_list_size")
    assert set(inputs) == expected_inputs
    task = task_from_ui(ui_case, config)
    assert isinstance(task.db_case_config, ZvecDiskANNIndexConfig)
    assert task.db_case_config == ZvecDiskANNIndexConfig(
        max_degree=32,
        build_list_size=80,
        pq_chunk_num=32,
        search_list_size=300 if ui_case.case.label == CaseLabel.Load else 600,
    )


@pytest.mark.parametrize("quantize_type", ["", "fp16", "int8"])
def test_zvec_ui_hnsw_parameters_reach_task(quantize_type: str) -> None:
    case = CaseConfig(case_id=CaseType.Performance768D10M)
    config, _inputs = render_zvec_ui(
        case,
        {
            "M": 32,
            "ef_construction": 250,
            "ef_search": 450,
            "quantize_type": quantize_type,
            "is_using_refiner": True,
        },
    )

    assert task_from_ui(case, config).db_case_config == ZvecHNSWIndexConfig(
        M=32,
        ef_construction=250,
        ef_search=450,
        quantize_type=quantize_type,
        is_using_refiner=True,
    )


@pytest.mark.parametrize(
    ("name", "minimum", "maximum", "default"),
    [
        ("max_degree", 1, 100, 64),
        ("build_list_size", 10, 100, 100),
        ("search_list_size", 1, MAX_STREAMLIT_INT, 300),
        ("pq_chunk_num", 0, 1024, 0),
    ],
)
def test_zvec_ui_diskann_bounds_and_defaults(name: str, minimum: int, maximum: int, default: int) -> None:
    case = CaseConfig(case_id=CaseType.Performance768D10M)
    config, inputs = render_zvec_ui(case, {"IndexType": IndexType.DISKANN.value})

    assert inputs[name]["min_value"] == minimum
    assert inputs[name]["max_value"] == maximum
    assert inputs[name]["value"] == default
    assert task_from_ui(case, config).db_case_config == ZvecDiskANNIndexConfig()


def test_zvec_ui_algorithm_switch_ignores_hidden_parameters() -> None:
    case = CaseConfig(case_id=CaseType.Performance768D10M)
    config, _inputs = render_zvec_ui(case, {"M": 32, "quantize_type": "int8", "is_using_refiner": True})
    config, inputs = render_zvec_ui(
        case,
        {"IndexType": IndexType.DISKANN.value, "max_degree": 32, "pq_chunk_num": 48},
        config,
    )

    assert CaseConfigParamType.quantize_type in config
    assert "quantize_type" not in inputs
    diskann = task_from_ui(case, config).db_case_config
    assert isinstance(diskann, ZvecDiskANNIndexConfig)
    assert diskann == ZvecDiskANNIndexConfig(max_degree=32, pq_chunk_num=48)
    assert "quantize_type" not in diskann.model_dump()

    config, inputs = render_zvec_ui(case, {"IndexType": IndexType.HNSW.value}, config)

    assert CaseConfigParamType.pq_chunk_num in config
    assert "pq_chunk_num" not in inputs
    hnsw = task_from_ui(case, config).db_case_config
    assert isinstance(hnsw, ZvecHNSWIndexConfig)
    assert hnsw == ZvecHNSWIndexConfig()
    assert "pq_chunk_num" not in hnsw.model_dump()
