from vectordb_bench.backend.cases import Case
from vectordb_bench.frontend.components.check_results.data import getChartData
from vectordb_bench.frontend.components.check_results.expanderStyle import initSidebarExanderStyle
from vectordb_bench.frontend.const.dbCaseConfigs import CASE_LIST
from vectordb_bench.frontend.const.styles import *
import streamlit as st

from vectordb_bench.models import CaseResult, TestResult


def getshownData(results: list[TestResult], st):
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


def getshownResults(results: list[TestResult], st) -> list[CaseResult]:
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
    selectedResult: list[CaseResult] = []
    for option in selectedResultSelectedOptions:
        result = results[resultSelectOptions.index(option)].results
        selectedResult += result

    return selectedResult


def getShowDbsAndCases(result: list[CaseResult], st) -> tuple[list[str], list[Case]]:
    initSidebarExanderStyle(st)
    allDbNames = list(set({res.task_config.db_name for res in result}))
    allDbNames.sort()
    allCasesSet = set({res.task_config.case_config.case_id for res in result})
    allCases: list[Case] = [case.case_cls() for case in CASE_LIST if case in allCasesSet]

    # DB Filter
    dbFilterContainer = st.container()
    showDBNames = filterView(
        dbFilterContainer,
        "DB Filter",
        allDbNames,
        col=1,
    )

    # Case Filter
    caseFilterContainer = st.container()
    showCases = filterView(
        caseFilterContainer,
        "Case Filter",
        [case for case in allCases],
        col=1,
        optionLables=[case.name for case in allCases],
    )

    return showDBNames, showCases


def filterView(container, header, options, col, optionLables=None):
    selectAllState = f"{header}-select-all-state"
    if selectAllState not in st.session_state:
        st.session_state[selectAllState] = True

    countKeyState = f"{header}-select-all-count-key"
    if countKeyState not in st.session_state:
        st.session_state[countKeyState] = 0

    expander = container.expander(header, True)
    selectAllColumns = expander.columns(SIDEBAR_CONTROL_COLUMNS, gap="small")
    selectAllButton = selectAllColumns[SIDEBAR_CONTROL_COLUMNS - 2].button(
        "select all",
        key=f"{header}-select-all-button",
        # type="primary",
    )
    clearAllButton = selectAllColumns[SIDEBAR_CONTROL_COLUMNS - 1].button(
        "clear all",
        key=f"{header}-clear-all-button",
        # type="primary",
    )
    if selectAllButton:
        st.session_state[selectAllState] = True
        st.session_state[countKeyState] += 1
    if clearAllButton:
        st.session_state[selectAllState] = False
        st.session_state[countKeyState] += 1
    columns = expander.columns(
        col,
        gap="small",
    )
    if optionLables is None:
        optionLables = options
    isActive = {option: st.session_state[selectAllState] for option in optionLables}
    for i, option in enumerate(optionLables):
        isActive[option] = columns[i % col].checkbox(
            optionLables[i],
            value=isActive[option],
            key=f"{optionLables[i]}-{st.session_state[countKeyState]}",
        )

    return [options[i] for i, option in enumerate(optionLables) if isActive[option]]
