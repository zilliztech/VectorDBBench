from vector_db_bench.frontend.utils import displayCaseText
from vector_db_bench.frontend.const import *


def getShowDbsAndCases(result, st):
    allDbNames = list(set({res.task_config.db_name for res in result}))
    allCasesSet = set({res.task_config.case_config.case_id for res in result})
    allCases = [case["name"].value for case in CASE_LIST if case["name"] in allCasesSet]

    dbFilterContainer = st.container()
    dbFilterContainer.subheader("DB Filter")
    showDBNames = filterView(allDbNames, dbFilterContainer, col=3)

    caseFilterContainer = st.container()
    caseFilterContainer.subheader("Case Filter")
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
