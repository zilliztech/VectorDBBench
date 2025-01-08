import streamlit as st
from vectordb_bench.frontend.components.run_test.autoRefresh import autoRefresh
from vectordb_bench.frontend.components.run_test.caseSelector import caseSelector
from vectordb_bench.frontend.components.run_test.dbConfigSetting import dbConfigSettings
from vectordb_bench.frontend.components.run_test.dbSelector import dbSelector
from vectordb_bench.frontend.components.run_test.generateTasks import generate_tasks
from vectordb_bench.frontend.components.run_test.hideSidebar import hideSidebar
from vectordb_bench.frontend.components.run_test.initStyle import initStyle
from vectordb_bench.frontend.components.run_test.submitTask import submitTask
from vectordb_bench.frontend.components.check_results.nav import NavToResults
from vectordb_bench.frontend.components.check_results.headerIcon import drawHeaderIcon
from vectordb_bench.frontend.components.check_results.stPageConfig import initRunTestPageConfig


def main():
    # set page config
    initRunTestPageConfig(st)

    # init style
    initStyle(st)

    # header
    drawHeaderIcon(st)

    # hide sidebar
    hideSidebar(st)

    # nav to results
    NavToResults(st)

    # header
    st.title("Run Your Test")
    # st.write("description [todo]")

    # select db
    dbSelectorContainer = st.container()
    activedDbList = dbSelector(dbSelectorContainer)

    # db config setting
    dbConfigs = {}
    isAllValid = True
    if len(activedDbList) > 0:
        dbConfigContainer = st.container()
        dbConfigs, isAllValid = dbConfigSettings(dbConfigContainer, activedDbList)

    # select case and set db_case_config
    caseSelectorContainer = st.container()
    activedCaseList, allCaseConfigs = caseSelector(caseSelectorContainer, activedDbList)

    # generate tasks
    tasks = generate_tasks(activedDbList, dbConfigs, activedCaseList, allCaseConfigs) if isAllValid else []

    # submit
    submitContainer = st.container()
    submitTask(submitContainer, tasks, isAllValid)

    # nav to results
    NavToResults(st, key="footer-nav-to-results")

    # autofresh
    autoRefresh()


if __name__ == "__main__":
    main()
