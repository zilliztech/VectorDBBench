import streamlit as st
from vector_db_bench.frontend.const import *
from vector_db_bench.models import TaskConfig, CaseConfig, DBCaseConfig
from vector_db_bench.interface import BenchMarkRunner

st.set_page_config(
    page_title="Falcon Mark - Open VectorDB Bench",
    page_icon="🧊",
    # layout="wide",
    initial_sidebar_state="collapsed",
)


st.title("Run Your Test")


@st.cache_resource
def getBenchMarkRunner():
    print("===> Cache Global BenchMarkRunner")
    return BenchMarkRunner()


benchMarkRunner = getBenchMarkRunner()

# DB Setting
st.divider()
dbContainter = st.container()
dbContainter.header("DB")
dbContainerColumns = dbContainter.columns(CHECKBOX_MAX_COLUMNS)
dbIsActived = {db: False for db in DB_LIST}
for i, db in enumerate(DB_LIST):
    column = dbContainerColumns[i % CHECKBOX_MAX_COLUMNS]
    dbIsActived[db] = column.checkbox(db.name)

activedDbList = [db for db in DB_LIST if dbIsActived[db]]
# print("activedDbList", activedDbList)


# DB Config Setting
dbConfigs = {}
if len(activedDbList) > 0:
    st.divider()
    dbConfigContainers = st.container()
    dbConfigContainers.header("DB Config")

    for i, activeDb in enumerate(activedDbList):
        dbConfigContainer = dbConfigContainers.container()
        dbConfigContainerColumns = dbConfigContainer.columns(
            [1, *[INPUT_WIDTH_RADIO for _ in range(INPUT_MAX_COLUMNS)]], gap="small"
        )
        dbConfigClass = activeDb.config
        properties = dbConfigClass.model_json_schema().get("properties")
        dbConfig = {}
        dbConfigContainerColumns[0].markdown("##### · %s" % activeDb.name)
        for j, property in enumerate(properties.items()):
            column = dbConfigContainerColumns[1 + j % INPUT_MAX_COLUMNS]
            key, value = property
            dbConfig[key] = column.text_input(
                value["title"], key="%s-%s" % (activeDb, key)
            )
        dbConfigs[activeDb] = dbConfigClass(**dbConfig)
# print("dbConfigs", dbConfigs)


# Case
st.divider()
caseContainers = st.container()
caseContainers.header("Case")
caseIsActived = {case["name"]: False for case in CASE_LIST}
for i, case in enumerate(CASE_LIST):
    caseContainer = caseContainers.container()
    columns = caseContainer.columns([1, CASE_INTRO_RATIO], gap="small")
    caseIsActived[case["name"]] = columns[0].checkbox(case["name"].value)
    columns[1].write(case["intro"])
activedCaseList = [case["name"] for case in CASE_LIST if caseIsActived[case["name"]]]
# print("activedCaseList", activedCaseList)

# Case Config Setting
allCaseConfigs = {
    db: {
        case["name"]: {
            # config.label: ""
            # for config in CASE_CONFIG_MAP.get(db, {}).get(case["name"], [])
        }
        for case in CASE_LIST
    }
    for db in DB_LIST
}
if len(activedDbList) > 0 and len(activedCaseList) > 0:
    st.divider()
    caseConfigContainers = st.container()
    caseConfigContainers.header("Case Config")

    for i, db in enumerate(activedDbList):
        caseConfigDBContainer = caseConfigContainers.container()
        caseConfigDBContainer.subheader(db.name)
        for j, case in enumerate(activedCaseList):
            caseConfigDBCaseContainer = caseConfigDBContainer.container()
            columns = caseConfigDBCaseContainer.columns(
                [1, *[INPUT_WIDTH_RADIO for _ in range(INPUT_MAX_COLUMNS)]], gap="small"
            )
            columns[0].markdown("##### · %s" % case.value)

            k = 0
            caseConfig = allCaseConfigs[db][case]
            for config in CASE_CONFIG_MAP.get(db, {}).get(case, []):
                if config.isDisplayed(caseConfig):
                    column = columns[1 + k % INPUT_MAX_COLUMNS]
                    key = "%s-%s-%s" % (db, case, config.label.value)
                    if config.inputType == InputType.Text:
                        caseConfig[config.label] = column.text_input(
                            config.label.value,
                            key=key,
                        )
                    elif config.inputType == InputType.Option:
                        caseConfig[config.label] = column.selectbox(
                            config.label.value,
                            config.inputConfig["options"],
                            key=key,
                        )
                    elif config.inputType == InputType.Number:
                        caseConfig[config.label] = column.number_input(
                            config.label.value,
                            format="%d",
                            step=1,
                            min_value=config.inputConfig["min"],
                            max_value=config.inputConfig["max"],
                            key=key,
                        )
                    k += 1
            if k == 0:
                columns[1].write("no config")
            # print("caseConfig", caseConfig)

# Contruct Task
tasks = [
    TaskConfig(
        db=db.value,
        db_config=dbConfigs[db],
        case_config=CaseConfig(
            case_id=case.value,
            custom_case={},
        ),
        db_case_config=db.case_config_cls(
            allCaseConfigs[db][case].get(CaseConfigParamType.IndexType, None)
        )(**{key.value: value for key, value in allCaseConfigs[db][case].items()}),
    )
    for case in activedCaseList
    for db in activedDbList
]
# print("\n=====>\nTasks:")
# for i, task in enumerate(tasks):
#     print(i, task)

# Control
st.divider()
controlContainer = st.container()

taskCount = 100
currentTaskId = 14
isRunning = False

## Task Progress
progressContainer = controlContainer.container()
if isRunning:
    progressContainer.markdown(f":running: task {currentTaskId} / {taskCount}")
    progressContainer.progress(currentTaskId / taskCount, text="")

## Submit
submitContainer = controlContainer.container()
columns = submitContainer.columns(CHECKBOX_MAX_COLUMNS)


def runHandler():
    benchMarkRunner.run(tasks)


def stopHandler():
    print("stop")
    # benchMarkRunner.stop_running()
    print("stop successful")


# def checkRunning():
#     return benchMarkRunner.has_running


columns[0].button("Run", disabled=isRunning, on_click=runHandler)
columns[1].button("Stop", disabled=not isRunning, on_click=stopHandler)
