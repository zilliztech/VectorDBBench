from vectordb_bench.backend.clients import DB
from vectordb_bench.frontend.components.run_test.inputWidget import inputWidget
from collections import defaultdict
from vectordb_bench.frontend.config.dbCaseConfigs import (
    UI_CASE_CLUSTERS,
    UICaseItem,
    UICaseItemCluster,
    get_case_config_inputs,
    get_custom_case_cluter,
)
from vectordb_bench.frontend.config.styles import (
    CASE_CONFIG_SETTING_COLUMNS,
    CHECKBOX_INDENT,
    DB_CASE_CONFIG_SETTING_COLUMNS,
)

from vectordb_bench.frontend.utils import addHorizontalLine
from vectordb_bench.models import CaseConfig


def caseSelector(st, activedDbList: list[DB]):
    st.markdown(
        "<div style='height: 24px;'></div>",
        unsafe_allow_html=True,
    )
    st.subheader("STEP 2: Choose the case(s)")
    st.markdown(
        "<div style='color: #647489; margin-bottom: 24px; margin-top: -12px;'>Choose at least one case you want to run the test for. </div>",
        unsafe_allow_html=True,
    )

    activedCaseList: list[CaseConfig] = []
    dbToCaseClusterConfigs = defaultdict(lambda: defaultdict(dict))
    dbToCaseConfigs = defaultdict(lambda: defaultdict(dict))
    caseClusters = UI_CASE_CLUSTERS + [get_custom_case_cluter()]
    for caseCluster in caseClusters:
        activedCaseList += caseClusterExpander(st, caseCluster, dbToCaseClusterConfigs, activedDbList)
    for db in dbToCaseClusterConfigs:
        for uiCaseItem in dbToCaseClusterConfigs[db]:
            for case in uiCaseItem.get_cases():
                dbToCaseConfigs[db][case] = dbToCaseClusterConfigs[db][uiCaseItem]

    return activedCaseList, dbToCaseConfigs


def caseClusterExpander(st, caseCluster: UICaseItemCluster, dbToCaseClusterConfigs, activedDbList: list[DB]):
    expander = st.expander(caseCluster.label, False)
    activedCases: list[CaseConfig] = []
    for uiCaseItem in caseCluster.uiCaseItems:
        if uiCaseItem.isLine:
            addHorizontalLine(expander)
        else:
            activedCases += caseItemCheckbox(expander, dbToCaseClusterConfigs, uiCaseItem, activedDbList)
    return activedCases


def caseItemCheckbox(st, dbToCaseClusterConfigs, uiCaseItem: UICaseItem, activedDbList: list[DB]):
    selected = st.checkbox(uiCaseItem.label)
    st.markdown(
        f"<div style='color: #1D2939; margin: -8px 0 20px {CHECKBOX_INDENT}px; font-size: 14px;'>{uiCaseItem.description}</div>",
        unsafe_allow_html=True,
    )

    caseConfigSetting(st.container(), uiCaseItem)

    if selected:
        dbCaseConfigSetting(st.container(), dbToCaseClusterConfigs, uiCaseItem, activedDbList)

    return uiCaseItem.get_cases() if selected else []


def caseConfigSetting(st, uiCaseItem: UICaseItem):
    config_inputs = uiCaseItem.extra_custom_case_config_inputs
    if len(config_inputs) == 0:
        return

    columns = st.columns(
        [
            1,
            *[DB_CASE_CONFIG_SETTING_COLUMNS / CASE_CONFIG_SETTING_COLUMNS] * CASE_CONFIG_SETTING_COLUMNS,
        ]
    )
    columns[0].markdown(
        f"<div style='margin: 0 0 24px {CHECKBOX_INDENT}px; font-size: 18px; font-weight: 600;'>Custom Config</div>",
        unsafe_allow_html=True,
    )
    for i, config_input in enumerate(config_inputs):
        column = columns[1 + i % CASE_CONFIG_SETTING_COLUMNS]
        key = f"custom-config-{uiCaseItem.label}-{config_input.label.value}"
        uiCaseItem.tmp_custom_config[config_input.label.value] = inputWidget(column, config=config_input, key=key)


def dbCaseConfigSetting(st, dbToCaseClusterConfigs, uiCaseItem: UICaseItem, activedDbList: list[DB]):
    for db in activedDbList:
        columns = st.columns(1 + DB_CASE_CONFIG_SETTING_COLUMNS)
        # column 0 - title
        dbColumn = columns[0]
        dbColumn.markdown(
            f"<div style='margin: 0 0 24px {CHECKBOX_INDENT}px; font-size: 18px; font-weight: 600;'>{db.name}</div>",
            unsafe_allow_html=True,
        )
        k = 0
        dbCaseConfig = dbToCaseClusterConfigs[db][uiCaseItem]
        for config in get_case_config_inputs(db, uiCaseItem.caseLabel):
            if config.isDisplayed(dbCaseConfig):
                column = columns[1 + k % DB_CASE_CONFIG_SETTING_COLUMNS]
                key = "%s-%s-%s" % (db, uiCaseItem.label, config.label.value)
                dbCaseConfig[config.label] = inputWidget(column, config, key)
                k += 1
        if k == 0:
            columns[1].write("Auto")
