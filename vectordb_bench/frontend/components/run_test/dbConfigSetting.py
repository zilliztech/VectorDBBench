from pydantic import ValidationError
from vectordb_bench.backend.clients import DB
from vectordb_bench.frontend.config.styles import DB_CONFIG_SETTING_COLUMNS
from vectordb_bench.frontend.utils import inputIsPassword


def dbConfigSettings(st, activedDbList: list[DB]):
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


def dbConfigSettingItem(st, activeDb: DB):
    st.markdown(
        f"<div style='font-weight: 600; font-size: 20px; margin-top: 16px;'>{activeDb.value}</div>",
        unsafe_allow_html=True,
    )
    columns = st.columns(DB_CONFIG_SETTING_COLUMNS)

    dbConfigClass = activeDb.config_cls
    properties = dbConfigClass.schema().get("properties")
    dbConfig = {}
    idx = 0

    # db config (unique)
    for key, property in properties.items():
        if key not in dbConfigClass.common_short_configs() and key not in dbConfigClass.common_long_configs():
            column = columns[idx % DB_CONFIG_SETTING_COLUMNS]
            idx += 1
            dbConfig[key] = column.text_input(
                key,
                key="%s-%s" % (activeDb.name, key),
                value=property.get("default", ""),
                type="password" if inputIsPassword(key) else "default",
            )
    # db config (common short labels)
    for key in dbConfigClass.common_short_configs():
        column = columns[idx % DB_CONFIG_SETTING_COLUMNS]
        idx += 1
        dbConfig[key] = column.text_input(
            key,
            key="%s-%s" % (activeDb.name, key),
            value="",
            type="default",
            placeholder="optional, for labeling results",
        )

    # db config (common long text_input)
    for key in dbConfigClass.common_long_configs():
        dbConfig[key] = st.text_area(
            key,
            key="%s-%s" % (activeDb.name, key),
            value="",
            placeholder="optional",
        )
    return dbConfig
