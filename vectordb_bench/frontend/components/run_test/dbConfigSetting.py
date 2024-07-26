from pydantic import ValidationError
from vectordb_bench.frontend.config.styles import *
from vectordb_bench.frontend.utils import inputIsPassword


def dbConfigSettings(st, activedDbList):
    expander = st.expander("Configurations for the selected databases", True)

    dbConfigs = {}
    isAllValid = True
    for activeDb in activedDbList:
        dbConfigSettingItemContainer = expander.container()
        dbConfig = dbConfigSettingItem(dbConfigSettingItemContainer, activeDb)
        try:
            dbConfigs[activeDb] = activeDb.config_cls(**dbConfig)
        except ValidationError as e:
            isAllValid = False
            errTexts = []
            for err in e.raw_errors:
                errLocs = err.loc_tuple()
                errInfo = err.exc
                errText = f"{', '.join(errLocs)} - {errInfo}"
                errTexts.append(errText)

            dbConfigSettingItemContainer.error(f"{'; '.join(errTexts)}")

    return dbConfigs, isAllValid


def dbConfigSettingItem(st, activeDb):
    st.markdown(
        f"<div style='font-weight: 600; font-size: 20px; margin-top: 16px;'>{activeDb.value}</div>",
        unsafe_allow_html=True,
    )
    columns = st.columns(DB_CONFIG_SETTING_COLUMNS)

    dbConfigClass = activeDb.config_cls
    properties = dbConfigClass.schema().get("properties")
    propertiesItems = list(properties.items())
    moveDBLabelToLast(propertiesItems)
    dbConfig = {}
    for j, property in enumerate(propertiesItems):
        column = columns[j % DB_CONFIG_SETTING_COLUMNS]
        key, value = property
        dbConfig[key] = column.text_input(
            key,
            key="%s-%s" % (activeDb, key),
            value=value.get("default", ""),
            type="password" if inputIsPassword(key) else "default",
        )
    return dbConfig


def moveDBLabelToLast(propertiesItems):
    propertiesItems.sort(key=lambda x: 1 if x[0] == "db_label" else 0)
