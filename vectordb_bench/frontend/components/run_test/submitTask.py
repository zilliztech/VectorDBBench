from datetime import datetime
from vectordb_bench import config
from vectordb_bench.frontend.config import styles
from vectordb_bench.interface import benchmark_runner
from vectordb_bench.models import TaskConfig


def submitTask(st, tasks, isAllValid):
    st.markdown(
        "<div style='height: 24px;'></div>",
        unsafe_allow_html=True,
    )
    st.subheader("STEP 3: Task Label")
    st.markdown(
        "<div style='color: #647489; margin-bottom: 20px; margin-top: -12px;'>This description is used to mark the result. </div>",
        unsafe_allow_html=True,
    )

    taskLabel = taskLabelInput(st)

    st.markdown(
        "<div style='height: 24px;'></div>",
        unsafe_allow_html=True,
    )

    controlPanelContainer = st.container()
    controlPanel(controlPanelContainer, tasks, taskLabel, isAllValid)


def taskLabelInput(st):
    defaultTaskLabel = datetime.now().strftime("%Y%m%d%H")
    columns = st.columns(styles.TASK_LABEL_INPUT_COLUMNS)
    taskLabel = columns[0].text_input("task_label", defaultTaskLabel, label_visibility="collapsed")
    return taskLabel


def advancedSettings(st):
    container = st.columns([1, 2])
    index_already_exists = container[0].checkbox("Index already exists", value=False)
    container[1].caption("if selected, inserting and building will be skipped.")

    container = st.columns([1, 2])
    use_aliyun = container[0].checkbox("Dataset from Aliyun (Shanghai)", value=False)
    container[1].caption(
        "if selected, the dataset will be downloaded from Aliyun OSS shanghai, default AWS S3 aws-us-west."
    )

    container = st.columns([1, 2])
    k = container[0].number_input("k", min_value=1, value=100, label_visibility="collapsed")
    container[1].caption("K value for number of nearest neighbors to search")

    container = st.columns([1, 2])
    defaultconcurrentInput = ",".join(map(str, config.NUM_CONCURRENCY))
    concurrentInput = container[0].text_input(
        "Concurrent Input", value=defaultconcurrentInput, label_visibility="collapsed"
    )
    container[1].caption("num of concurrencies for search tests to get max-qps")
    return index_already_exists, use_aliyun, k, concurrentInput


def controlPanel(st, tasks: list[TaskConfig], taskLabel, isAllValid):
    index_already_exists, use_aliyun, k, concurrentInput = advancedSettings(st)

    def runHandler():
        benchmark_runner.set_drop_old(not index_already_exists)

        try:
            concurrentInput_list = [int(item.strip()) for item in concurrentInput.split(",")]
        except ValueError:
            st.write("please input correct number")
            return None

        for task in tasks:
            task.case_config.k = k
            task.case_config.concurrency_search_config.num_concurrency = concurrentInput_list

        benchmark_runner.set_download_address(use_aliyun)
        benchmark_runner.run(tasks, taskLabel)

    def stopHandler():
        benchmark_runner.stop_running()

    isRunning = benchmark_runner.has_running()

    if isRunning:
        currentTaskId = benchmark_runner.get_current_task_id()
        tasksCount = benchmark_runner.get_tasks_count()
        text = f":running: Running Task {currentTaskId} / {tasksCount}"
        st.progress(currentTaskId / tasksCount, text=text)

        columns = st.columns(6)
        columns[0].button(
            "Run Your Test",
            disabled=True,
            on_click=runHandler,
            type="primary",
        )
        columns[1].button(
            "Stop",
            on_click=stopHandler,
            type="primary",
        )

    else:
        errorText = benchmark_runner.latest_error or ""
        if len(errorText) > 0:
            st.error(errorText)
        disabled = True if len(tasks) == 0 or not isAllValid else False
        if not isAllValid:
            st.error("Make sure all config is valid.")
        elif len(tasks) == 0:
            st.warning("No tests to run.")
        st.button(
            "Run Your Test",
            disabled=disabled,
            on_click=runHandler,
            type="primary",
        )
