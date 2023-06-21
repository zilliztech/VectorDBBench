from datetime import datetime
from vectordb_bench.frontend.const.styles import *
from vectordb_bench.interface import benchMarkRunner


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
    columns = st.columns(TASK_LABEL_INPUT_COLUMNS)
    taskLabel = columns[0].text_input(
        "task_label", defaultTaskLabel, label_visibility="collapsed"
    )
    return taskLabel


def controlPanel(st, tasks, taskLabel, isAllValid):
    isRunning = benchMarkRunner.has_running()
    runHandler = lambda: benchMarkRunner.run(tasks, taskLabel)
    stopHandler = lambda: benchMarkRunner.stop_running()

    if isRunning:
        currentTaskId = benchMarkRunner.get_current_task_id()
        tasksCount = benchMarkRunner.get_tasks_count()
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
        errorText = benchMarkRunner.latest_error or ""
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
