from vectordb_bench.frontend.components.check_results.data import getChartData
from vectordb_bench.frontend.const import *
import streamlit as st


def getshownData(results, st):
    # hide the nav
    st.markdown(
        "<style> div[data-testid='stSidebarNav'] {display: none;} </style>",
        unsafe_allow_html=True,
    )

    st.header("Filters")

    shownResults = getshownResults(results, st)
    showDBNames, showCases = getShowDbsAndCases(shownResults, st)

    shownData, failedTasks = getChartData(shownResults, showDBNames, showCases)

    return shownData, failedTasks, showCases


def getshownResults(results, st):
    resultSelectOptions = [
        result.task_label
        if result.task_label != result.run_id
        else f"res-{result.run_id[:4]}"
        for result in results
    ]
    if len(resultSelectOptions) == 0:
        st.write(
            "There are no results to display. Please wait for the task to complete or run a new task."
        )
        return []

    selectedResultSelectedOptions = st.multiselect(
        "Select the task results you need to analyze.",
        resultSelectOptions,
        # label_visibility="hidden",
        default=resultSelectOptions,
    )
    selectedResult = []
    for option in selectedResultSelectedOptions:
        result = results[resultSelectOptions.index(option)].results
        selectedResult += result

    return selectedResult


def getShowDbsAndCases(result, st):
    # expanderStyles
    st.markdown(
        """
<style>
    section[data-testid='stSidebar'] div[data-testid='stExpander'] div[data-testid='stVerticalBlock'] { gap: 0.2rem; }
    div[data-testid='stExpander'] {background-color: #ffffff;}
    section[data-testid='stSidebar'] .streamlit-expanderHeader p {font-size: 16px; font-weight: 600;}
    section[data-testid='stSidebar']
        div[data-testid='stExpander']
            div[data-testid='stVerticalBlock'] 
                button {
                    padding: 0 0.5rem;
                    margin-bottom: 8px;
                    float: right;
                }
<style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown

    allDbNames = list(set({res.task_config.db_name for res in result}))
    allDbNames.sort()
    allCasesSet = set({res.task_config.case_config.case_id for res in result})
    allCases = [case.value for case in CASE_LIST if case in allCasesSet]

    # DB Filter
    dbFilterContainer = st.container()
    showDBNames = filterView(
        dbFilterContainer,
        "DB Filter",
        allDbNames,
        col=1,
        sessionStateKey=DB_SELECT_ALL,
    )

    # Case Filter
    caseFilterContainer = st.container()
    showCases = filterView(
        caseFilterContainer,
        "Case Filter",
        allCases,
        col=1,
        sessionStateKey=CASE_SELECT_ALL,
    )

    return showDBNames, showCases


def filterView(container, header, options, col, sessionStateKey, optionLables=None):
    expander = container.expander(header, True)
    selectAllColumns = expander.columns(SIDEBAR_CONTROL_COLUMNS, gap="small")
    selectAllButton = selectAllColumns[SIDEBAR_CONTROL_COLUMNS - 2].button(
        "select all",
        key=f"{header}-select-all",
        # type="primary",
    )
    clearAllButton = selectAllColumns[SIDEBAR_CONTROL_COLUMNS - 1].button(
        "clear all",
        key=f"{header}-clear-all",
        # type="primary",
    )
    if selectAllButton:
        st.session_state[sessionStateKey] = True
        st.session_state[getSelectAllKey(sessionStateKey)] += 1
    if clearAllButton:
        st.session_state[sessionStateKey] = False
        st.session_state[getSelectAllKey(sessionStateKey)] += 1
    columns = expander.columns(
        col,
        gap="small",
    )
    isActive = {option: st.session_state[sessionStateKey] for option in options}
    if optionLables is None:
        optionLables = options
    for i, option in enumerate(options):
        isActive[option] = columns[i % col].checkbox(
            optionLables[i],
            value=isActive[option],
            key=f"{optionLables[i]}-{st.session_state[getSelectAllKey(sessionStateKey)]}",
        )

    return [option for option in options if isActive[option]]
