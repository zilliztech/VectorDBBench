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
        try:
            dbConfig = dbConfigSettingItem(dbConfigSettingItemContainer, activeDb)
            dbConfigs[activeDb] = activeDb.config_cls(**dbConfig)
            # Probe client module so missing optional deps surface now, not on Run.
            _ = activeDb.init_cls
        except ModuleNotFoundError as e:
            isAllValid = False
            dbConfigSettingItemContainer.error(f"{activeDb.value} needs `{e.name}` but it is not installed.")
        except ValidationError as e:
            isAllValid = False
            errTexts = [f"{', '.join(str(x) for x in err['loc'])} - {err['msg']}" for err in e.errors()]
            dbConfigSettingItemContainer.error("; ".join(errTexts))

    return dbConfigs, isAllValid


def dbConfigSettingItem(st, activeDb: DB):
    st.markdown(
        f"<div style='font-weight: 600; font-size: 20px; margin-top: 16px;'>{activeDb.value}</div>",
        unsafe_allow_html=True,
    )
    columns = st.columns(DB_CONFIG_SETTING_COLUMNS)

    dbConfigClass = activeDb.config_cls
    schema = dbConfigClass.schema()
    property_items = schema.get("properties").items()
    required_fields = set(schema.get("required", []))
    dbConfig = {}
    idx = 0

    # db config (unique)
    for key, property in property_items:
        if key not in dbConfigClass.common_short_configs() and key not in dbConfigClass.common_long_configs():
            column = columns[idx % DB_CONFIG_SETTING_COLUMNS]
            idx += 1
            input_value = column.text_input(
                key,
                key=f"{activeDb.name}-{key}",
                value=property.get("default", ""),
                type="password" if inputIsPassword(key) else "default",
                placeholder="optional" if key not in required_fields else None,
            )
            if key in required_fields or input_value:
                dbConfig[key] = input_value

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
