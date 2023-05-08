import streamlit as st
from vector_db_bench.frontend.components.run_test.autoRefresh import autoRefresh
from vector_db_bench.frontend.components.run_test.caseSelector import caseSelector
from vector_db_bench.frontend.components.run_test.dbConfigSetting import dbConfigSettings
from vector_db_bench.frontend.components.run_test.dbSelector import dbSelector
from vector_db_bench.frontend.components.run_test.generateTasks import generate_tasks
from vector_db_bench.frontend.components.check_results.headerIcon import drawHeaderIcon
from vector_db_bench.frontend.components.run_test.hideSidebar import hideSidebar
from vector_db_bench.frontend.components.check_results.nav import NavToResults
from vector_db_bench.frontend.components.run_test.submitTask import submitTask


def main():
    st.set_page_config(
        page_title="VectorDB Benchmark",
        page_icon="https://assets.zilliz.com/favicon_f7f922fe27.png",
        # layout="wide",
        initial_sidebar_state="collapsed",
    )
    # header
    drawHeaderIcon(st)

    # hide sidebar
    hideSidebar(st)

    # nav to results
    NavToResults(st)

    # header
    st.title("Run Your Test")
    st.write("description [todo]")

    # select db
    dbSelectorContainer = st.container()
    activedDbList = dbSelector(dbSelectorContainer)

    # db config setting
    dbConfigs = {}
    if len(activedDbList) > 0:
        dbConfigContainer = st.container()
        dbConfigs = dbConfigSettings(dbConfigContainer, activedDbList)

    # select case and set db_case_config
    caseSelectorContainer = st.container()
    activedCaseList, allCaseConfigs = caseSelector(caseSelectorContainer, activedDbList)

    # generate tasks
    tasks = generate_tasks(activedDbList, dbConfigs, activedCaseList, allCaseConfigs)

    # submit
    submitContainer = st.container()
    submitTask(submitContainer, tasks)

    # autofresh
    autoRefresh()


if __name__ == "__main__":
    main()
