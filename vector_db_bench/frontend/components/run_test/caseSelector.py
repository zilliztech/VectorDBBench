from vector_db_bench.frontend.const import *
from vector_db_bench.frontend.utils import displayCaseText


def caseSelector(st, activedDbList):
    st.markdown(
        "<div style='height: 24px;'></div>",
        unsafe_allow_html=True,
    )
    st.subheader("STEP 2: Choose the case(s)")
    st.markdown(
        "<div style='color: #647489; margin-bottom: 24px; margin-top: -12px;'>Choose at least one case you want to run the test for. </div>",
        unsafe_allow_html=True,
    )

    caseIsActived = {case["name"]: False for case in CASE_LIST}
    allCaseConfigs = {db: {case["name"]: {} for case in CASE_LIST} for db in DB_LIST}
    for case in CASE_LIST:
        caseItemContainer = st.container()
        caseIsActived[case["name"]] = caseItem(
            caseItemContainer, allCaseConfigs, case, activedDbList
        )
        if case.get("divider"):
            caseItemContainer.markdown(
                "<div style='border: 1px solid #cccccc60; margin-bottom: 24px;'></div>",
                unsafe_allow_html=True,
            )
    activedCaseList = [
        case["name"] for case in CASE_LIST if caseIsActived[case["name"]]
    ]
    return activedCaseList, allCaseConfigs

def caseItem(st, allCaseConfigs, case, activedDbList):
    selected = st.checkbox(displayCaseText(case["name"].value))
    st.markdown(
        f"<div style='color: #1D2939; margin: -8px 0 20px {CHECKBOX_INDENT}px; font-size: 14px;'>{case['intro']}</div>",
        unsafe_allow_html=True,
    )

    if selected:
        caseConfigSettingContainer = st.container()
        caseConfigSetting(
            caseConfigSettingContainer, allCaseConfigs, case["name"], activedDbList
        )

    return selected


def caseConfigSetting(st, allCaseConfigs, case, activedDbList):
    for db in activedDbList:
        columns = st.columns(1 + CASE_CONFIG_SETTING_COLUMNS)
        # column 0 - title
        dbColumn = columns[0]
        dbColumn.markdown(
            f"<div style='margin: 0 0 24px {CHECKBOX_INDENT}px; font-size: 18px; font-weight: 600;'>{db.name}</div>",
            unsafe_allow_html=True,
        )
        caseConfig = allCaseConfigs[db][case]
        k = 0
        for config in CASE_CONFIG_MAP.get(db, {}).get(case, []):
            if config.isDisplayed(caseConfig):
                column = columns[1 + k % CASE_CONFIG_SETTING_COLUMNS]
                key = "%s-%s-%s" % (db, case, config.label.value)
                if config.inputType == InputType.Text:
                    caseConfig[config.label] = column.text_input(
                        config.label.value,
                        key=key,
                        value=config.inputConfig["value"],
                    )
                elif config.inputType == InputType.Option:
                    caseConfig[config.label] = column.selectbox(
                        config.label.value,
                        config.inputConfig["options"],
                        key=key,
                    )
                elif config.inputType == InputType.Number:
                    caseConfig[config.label] = column.number_input(
                        config.label.value,
                        format="%d",
                        step=1,
                        min_value=config.inputConfig["min"],
                        max_value=config.inputConfig["max"],
                        key=key,
                        value=config.inputConfig["value"],
                    )
                k += 1
        if k == 0:
            columns[1].write("Auto")
