from vectordb_bench.backend.cases import Case
from vectordb_bench.frontend.components.check_results.data import getChartData
from vectordb_bench.frontend.components.check_results.expanderStyle import initSidebarExanderStyle
from vectordb_bench.frontend.config.dbCaseConfigs import CASE_NAME_ORDER
from vectordb_bench.frontend.config.styles import *
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
    showDBNames, showCaseNames = getShowDbsAndCases(shownResults, st)

    shownData, failedTasks = getChartData(shownResults, showDBNames, showCaseNames)

    return shownData, failedTasks, showCaseNames


def getshownResults(results: list[TestResult], st) -> list[CaseResult]:
    resultSelectOptions = [
        result.task_label if result.task_label != result.run_id else f"res-{result.run_id[:4]}" for result in results
    ]
    if len(resultSelectOptions) == 0:
        st.write("There are no results to display. Please wait for the task to complete or run a new task.")
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


def getShowDbsAndCases(result: list[CaseResult], st) -> tuple[list[str], list[str]]:
    initSidebarExanderStyle(st)
    allDbNames = list(set({res.task_config.db_name for res in result}))
    allDbNames.sort()
    allCases: list[Case] = [
        res.task_config.case_config.case_id.case_cls(res.task_config.case_config.custom_case) for res in result
    ]
    allCaseNameSet = set({case.name for case in allCases})
    allCaseNames = [case_name for case_name in CASE_NAME_ORDER if case_name in allCaseNameSet] + [
        case_name for case_name in allCaseNameSet if case_name not in CASE_NAME_ORDER
    ]

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
    showCaseNames = filterView(
        caseFilterContainer,
        "Case Filter",
        [caseName for caseName in allCaseNames],
        col=1,
    )

    return showDBNames, showCaseNames


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
