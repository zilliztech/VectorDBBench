from vectordb_bench.frontend.config.styles import *
from vectordb_bench.frontend.config.dbCaseConfigs import *
from collections import defaultdict

from vectordb_bench.frontend.utils import addHorizontalLine


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
            for case in uiCaseItem.cases:
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

    if selected:
        caseConfigSetting(st.container(), dbToCaseClusterConfigs, uiCaseItem, activedDbList)

    return uiCaseItem.cases if selected else []


def caseConfigSetting(st, dbToCaseClusterConfigs, uiCaseItem: UICaseItem, activedDbList: list[DB]):
    for db in activedDbList:
        columns = st.columns(1 + CASE_CONFIG_SETTING_COLUMNS)
        # column 0 - title
        dbColumn = columns[0]
        dbColumn.markdown(
            f"<div style='margin: 0 0 24px {CHECKBOX_INDENT}px; font-size: 18px; font-weight: 600;'>{db.name}</div>",
            unsafe_allow_html=True,
        )
        k = 0
        caseConfig = dbToCaseClusterConfigs[db][uiCaseItem]
        for config in CASE_CONFIG_MAP.get(db, {}).get(uiCaseItem.caseLabel, []):
            if config.isDisplayed(caseConfig):
                column = columns[1 + k % CASE_CONFIG_SETTING_COLUMNS]
                key = "%s-%s-%s" % (db, uiCaseItem.label, config.label.value)
                if config.inputType == InputType.Text:
                    caseConfig[config.label] = column.text_input(
                        config.displayLabel if config.displayLabel else config.label.value,
                        key=key,
                        help=config.inputHelp,
                        value=config.inputConfig["value"],
                    )
                elif config.inputType == InputType.Option:
                    caseConfig[config.label] = column.selectbox(
                        config.displayLabel if config.displayLabel else config.label.value,
                        config.inputConfig["options"],
                        key=key,
                        help=config.inputHelp,
                    )
                elif config.inputType == InputType.Number:
                    caseConfig[config.label] = column.number_input(
                        config.displayLabel if config.displayLabel else config.label.value,
                        # format="%d",
                        step=config.inputConfig.get("step", 1),
                        min_value=config.inputConfig["min"],
                        max_value=config.inputConfig["max"],
                        key=key,
                        value=config.inputConfig["value"],
                        help=config.inputHelp,
                    )
                elif config.inputType == InputType.Float:
                    caseConfig[config.label] = column.number_input(
                        config.displayLabel if config.displayLabel else config.label.value,
                        step=config.inputConfig.get("step", 0.1),
                        min_value=config.inputConfig["min"],
                        max_value=config.inputConfig["max"],
                        key=key,
                        value=config.inputConfig["value"],
                        help=config.inputHelp,
                    )
                elif config.inputType == InputType.Bool:
                    caseConfig[config.label] = column.checkbox(
                        config.displayLabel if config.displayLabel else config.label.value,
                        value=config.inputConfig["value"],
                        help=config.inputHelp,
                    )
                k += 1
        if k == 0:
            columns[1].write("Auto")
