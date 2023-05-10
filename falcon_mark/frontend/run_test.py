import streamlit as st
from falcon_mark.frontend.const import *

# from falcon_mark.interface import BenchMarkRunner

st.set_page_config(
    page_title="Falcon Mark - Open VectorDB Bench",
    page_icon="🧊",
    # layout="wide",
    initial_sidebar_state="collapsed",
)


st.title("Run Your Test")


# @st.cache_resource
# def getBenchMarkRunner():
#     print("===> Cache Global BenchMarkRunner")
#     return BenchMarkRunner()


# benchMarkRunner = getBenchMarkRunner()

# DB Setting
st.divider()
dbContainter = st.container()
dbContainter.header("DB")
dbContainerColumns = dbContainter.columns(CHECKBOX_MAX_COLUMNS)
dbIsActived = {db: False for db in DB_LIST}
for i, db in enumerate(DB_LIST):
    column = dbContainerColumns[i % CHECKBOX_MAX_COLUMNS]
    dbIsActived[db] = column.checkbox(db.value)

activedDbList = [db for db in DB_LIST if dbIsActived[db]]
print("activedDbList", activedDbList)


# DB Config Setting
if len(activedDbList) > 0:
    st.divider()
    dbConfigContainers = st.container()
    dbConfigContainers.header("DB Config")
    dbConfigs = {db: {config: "" for config in DB_CONFIG_MAP[db]} for db in activedDbList}

    for i, activeDb in enumerate(activedDbList):
        dbConfigContainer = dbConfigContainers.container()
        dbConfigContainerColumns = dbConfigContainer.columns(
            [1, *[INPUT_WIDTH_RADIO for _ in range(INPUT_MAX_COLUMNS)]], gap="small"
        )
        dbConfig = dbConfigs[activeDb]
        dbConfigContainerColumns[0].markdown("##### · %s" % activeDb.value)
        for j, config in enumerate(dbConfig.keys()):
            column = dbConfigContainerColumns[1 + j % INPUT_MAX_COLUMNS]
            dbConfig[config] = column.text_input(
                config.value, key="%s-%s" % (activeDb, config)
            )
    print("dbConfigSetting", dbConfigs)


# Case
st.divider()
caseContainers = st.container()
caseContainers.header("Case")
caseIsActived = {case["name"]: False for case in CASE_LIST}
for i, case in enumerate(CASE_LIST):
    caseContainer = caseContainers.container()
    columns = caseContainer.columns(
        [1, INPUT_WIDTH_RADIO * INPUT_MAX_COLUMNS], gap="small"
    )
    caseIsActived[case["name"]] = columns[0].checkbox(case["name"].value)
    columns[1].write(case["intro"])
activedCaseList = [case["name"] for case in CASE_LIST if caseIsActived[case["name"]]]
print("activeOptionList", activedCaseList)

# Case Config Setting
if len(activedDbList) > 0 and len(activedCaseList) > 0:
    st.divider()
    caseConfigContainers = st.container()
    caseConfigContainers.header("Case Config")

    allCaseConfigs = {
        db: {
            case["name"]: {
                config["name"]: ""
                for config in CASE_CONFIG_MAP.get(db, {}).get(case["name"], [])
            }
            for case in CASE_LIST
        }
        for db in DB_LIST
    }
    for i, db in enumerate(activedDbList):
        caseConfigDBContainer = caseConfigContainers.container()
        caseConfigDBContainer.subheader(db.value)
        for j, case in enumerate(activedCaseList):
            caseConfigDBCaseContainer = caseConfigDBContainer.container()
            columns = caseConfigDBCaseContainer.columns(
                [1, *[INPUT_WIDTH_RADIO for _ in range(INPUT_MAX_COLUMNS)]], gap="small"
            )
            columns[0].markdown("##### · %s" % case.value)

            k = 0
            caseConfig = allCaseConfigs[db][case]
            for config in CASE_CONFIG_MAP.get(db, {}).get(case, []):
                if config.get("checked", DEFAULT_CONFIG_CHECKED)(caseConfig):
                    InputType = config.get("inputType", InputType.String)
                    column = columns[1 + k % INPUT_MAX_COLUMNS]
                    key = "%s-%s-%s" % (db, case, config["name"])
                    if InputType == InputType.String:
                        caseConfig[config["name"]] = column.text_input(
                            config["name"].value,
                            key=key,
                        )
                    elif InputType == InputType.Option:
                        caseConfig[config["name"]] = column.selectbox(
                            config["name"].value,
                            config["options"],
                            key=key,
                        )
                    elif InputType == InputType.Int:
                        caseConfig[config["name"]] = column.number_input(
                            config["name"].value,
                            format="%d",
                            step=1,
                            min_value=config["min"],
                            max_value=config["max"],
                            key=key,
                        )
                    k += 1
            if k == 0:
                columns[1].write("no config")
    print("allCaseConfigs", allCaseConfigs)


# Control
st.divider()
controlContainer = st.container()

taskCount = 100
currentTaskId = 14
isRunning = True

## Task Progress
progressContainer = controlContainer.container()
if isRunning:
    progressContainer.markdown(f":running: task {currentTaskId} / {taskCount}")
    progressContainer.progress(currentTaskId / taskCount, text="")

## Submit
submitContainer = controlContainer.container()
columns = submitContainer.columns(CHECKBOX_MAX_COLUMNS)


def runHandler():
    print("run")
    # benchMarkRunner.run()
    print("run successful")


def stopHandler():
    print("stop")
    # benchMarkRunner.stop_running()
    print("stop successful")


def checkRunning():
    return benchMarkRunner.has_running


columns[0].button("Run", disabled=isRunning, on_click=runHandler)
columns[1].button("Stop", disabled=not isRunning, on_click=stopHandler)
