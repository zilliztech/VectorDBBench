from datetime import datetime

import streamlit as st

from vectordb_bench import config
from vectordb_bench.backend.dataset import DatasetManager
from vectordb_bench.frontend.config import styles
from vectordb_bench.interface import benchmark_runner
from vectordb_bench.models import TaskConfig


def submitTask(container, tasks, isAllValid):
    container.markdown(
        "<div style='height: 24px;'></div>",
        unsafe_allow_html=True,
    )
    container.subheader("STEP 3: Task Label")
    container.markdown(
        "<div style='color: #647489; margin-bottom: 20px; margin-top: -12px;'>This description is used to mark the result. </div>",
        unsafe_allow_html=True,
    )

    taskLabel = taskLabelInput(container)

    container.markdown(
        "<div style='height: 24px;'></div>",
        unsafe_allow_html=True,
    )

    controlPanel(container.container(), tasks, taskLabel, isAllValid)


def taskLabelInput(container):
    defaultTaskLabel = datetime.now().strftime("%Y%m%d%H")
    cols = container.columns(styles.TASK_LABEL_INPUT_COLUMNS)
    return cols[0].text_input("task_label", defaultTaskLabel, label_visibility="collapsed")


def get_max_search_k(tasks: list[TaskConfig]) -> int | None:
    limits = []
    for task in tasks:
        case = task.case_config.case
        if not isinstance(case.dataset, DatasetManager):
            continue
        max_k = case.dataset.max_search_k(case.filters)
        if max_k is not None:
            limits.append(max_k)
    return min(limits, default=None)


def apply_run_settings(
    tasks: list[TaskConfig],
    *,
    k: int,
    concurrencies: list[int],
    concurrency_duration: int,
    load_concurrency: int,
) -> None:
    for task in tasks:
        case = task.case_config.case
        if isinstance(case.dataset, DatasetManager) and case.dataset.data.with_gt:
            case.dataset.resolve_search_files(k=k, filters=case.filters)

    for task in tasks:
        task.case_config.k = k
        task.case_config.concurrency_search_config.num_concurrency = concurrencies
        task.case_config.concurrency_search_config.concurrency_duration = concurrency_duration
        task.load_concurrency = load_concurrency


def advancedSettings(container, tasks: list[TaskConfig]):
    cols = container.columns([1, 2])
    index_already_exists = cols[0].checkbox("Index already exists", value=False)
    cols[1].caption("if selected, inserting and building will be skipped.")

    cols = container.columns([1, 2])
    use_aliyun = cols[0].checkbox("Dataset from Aliyun (Shanghai)", value=False)
    cols[1].caption("if selected, the dataset will be downloaded from Aliyun OSS shanghai, default AWS S3 aws-us-west.")

    cols = container.columns([1, 2])
    max_k = get_max_search_k(tasks)
    k_config = dict(
        min_value=1,
        value=config.K_DEFAULT,
        label_visibility="collapsed",
        key=f"search-k-{max_k or 'unbounded'}",
    )
    if max_k is not None:
        k_config["max_value"] = max_k
    k = cols[0].number_input("k", **k_config)
    cols[1].caption("K value for number of nearest neighbors to search")

    cols = container.columns([1, 2])
    defaultconcurrentInput = ",".join(map(str, config.NUM_CONCURRENCY))
    concurrentInput = cols[0].text_input("Concurrent Input", value=defaultconcurrentInput, label_visibility="collapsed")
    cols[1].caption("num of concurrencies for search tests to get max-qps")

    cols = container.columns([1, 2])
    concurrency_duration = cols[0].number_input(
        "Concurrency Duration", value=config.CONCURRENCY_DURATION, label_visibility="collapsed"
    )
    cols[1].caption("concurrency duration for each concurrency search test")

    cols = container.columns([1, 2])
    load_concurrency = cols[0].number_input(
        "Load Concurrency", min_value=0, value=config.LOAD_CONCURRENCY, label_visibility="collapsed"
    )
    cols[1].caption("number of concurrent workers for data loading in performance cases (0 = cpu_count)")
    return index_already_exists, use_aliyun, k, concurrentInput, concurrency_duration, load_concurrency


def controlPanel(container, tasks: list[TaskConfig], taskLabel, isAllValid):
    index_already_exists, use_aliyun, k, concurrentInput, concurrency_duration, load_concurrency = advancedSettings(
        container, tasks
    )

    def runHandler():
        try:
            concurrentInput_list = [int(item.strip()) for item in concurrentInput.split(",")]
            apply_run_settings(
                tasks,
                k=k,
                concurrencies=concurrentInput_list,
                concurrency_duration=concurrency_duration,
                load_concurrency=load_concurrency,
            )
        except ValueError as exc:
            container.write(str(exc))
            return None

        benchmark_runner.set_drop_old(not index_already_exists)
        benchmark_runner.set_download_address(use_aliyun)
        benchmark_runner.run(tasks, taskLabel)

    def stopHandler():
        benchmark_runner.stop_running()

    @st.fragment(run_every=f"{styles.MAX_AUTO_REFRESH_INTERVAL / 1000}s")
    def _renderLiveStatus():
        if benchmark_runner.has_running():
            currentTaskId = benchmark_runner.get_current_task_id()
            tasksCount = benchmark_runner.get_tasks_count()
            text = f":running: Running Task {currentTaskId} / {tasksCount}"
            if tasksCount > 0:
                st.progress(currentTaskId / tasksCount, text=text)
            cols = st.columns(6)
            cols[0].button(
                "Run Your Test",
                disabled=True,
                on_click=runHandler,
                type="primary",
                key="run-disabled",
            )
            cols[1].button(
                "Stop",
                on_click=stopHandler,
                type="primary",
                key="stop-btn",
            )
        else:
            errorText = benchmark_runner.latest_error or ""
            if len(errorText) > 0:
                st.error(errorText)
            disabled = len(tasks) == 0 or not isAllValid
            if not isAllValid:
                st.error("Make sure all config is valid.")
            elif len(tasks) == 0:
                st.warning("No tests to run.")
            st.button(
                "Run Your Test",
                disabled=disabled,
                on_click=runHandler,
                type="primary",
                key="run-btn",
            )

    _renderLiveStatus()
