from vector_db_bench.frontend.components.check_results.data import getChartData
from vector_db_bench.frontend.utils import displayCaseText
from vector_db_bench.frontend.const import *


def getshownData(results, st):
    # hide the nav
    st.markdown(
        "<style> div[data-testid='stSidebarNav'] {display: none;} </style>",
        unsafe_allow_html=True,
    )

    st.header("Filters")

    shownResults = getshownResults(results, st)
    showDBNames, showCases = getShowDbsAndCases(shownResults, st)

    shownData = getChartData(shownResults, showDBNames, showCases)

    return shownData, showCases


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
    st.markdown("<style> section[data-testid='stSidebar'] div[data-testid='stExpander'] div[data-testid='stVerticalBlock'] { gap: 0.2rem; }  </style>", unsafe_allow_html=True,)
    st.markdown(
        "<style> div[data-testid='stExpander'] {background-color: #ffffff;} </style>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<style> section[data-testid='stSidebar'] .streamlit-expanderHeader p {font-size: 16px; font-weight: 600;} </style>",
        unsafe_allow_html=True,
    )

    allDbNames = list(set({res.task_config.db_name for res in result}))
    allDbNames.sort()
    allCasesSet = set({res.task_config.case_config.case_id for res in result})
    allCases = [case["name"].value for case in CASE_LIST if case["name"] in allCasesSet]

    # dbFilterContainer = st.container()
    # dbFilterContainer.subheader("DB Filter")
    dbFilterContainer = st.expander("DB Filter", True)
    showDBNames = filterView(allDbNames, dbFilterContainer, col=1)

    # caseFilterContainer = st.container()
    # caseFilterContainer.subheader("Case Filter")
    caseFilterContainer = st.expander("Case Filter", True)
    showCases = filterView(
        allCases,
        caseFilterContainer,
        col=1,
        optionLables=[displayCaseText(case) for case in allCases],
    )

    return showDBNames, showCases


def filterView(options, st, col, optionLables=None):
    columns = st.columns(
        col,
        gap="small",
    )
    isActive = {option: True for option in options}
    if optionLables is None:
        optionLables = options
    for i, option in enumerate(options):
        isActive[option] = columns[i % col].checkbox(
            optionLables[i], value=isActive[option]
        )

    return [option for option in options if isActive[option]]
