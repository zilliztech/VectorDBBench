from vector_db_bench.frontend.const import *
from vector_db_bench.frontend.utils import inputIsPassword


def dbConfigSettings(st, activedDbList):
    st.markdown(
        "<style> .streamlit-expanderHeader p {font-size: 20px; font-weight: 600;}</style>",
        unsafe_allow_html=True,
    )
    expander = st.expander("Configurations for the selected databases", True)

    dbConfigs = {}
    for activeDb in activedDbList:
        dbConfigSettingItemContainer = expander.container()
        dbConfig = dbConfigSettingItem(dbConfigSettingItemContainer, activeDb)
        dbConfigs[activeDb] = dbConfig
        
    return dbConfigs

def dbConfigSettingItem(st, activeDb):
    st.markdown(
        f"<div style='font-weight: 600; font-size: 20px; margin-top: 16px;'>{activeDb.value}</div>",
        unsafe_allow_html=True,
    )
    columns = st.columns(DB_CONFIG_SETTING_COLUMNS)

    activeDbCls = activeDb.init_cls
    dbConfigClass = activeDbCls.config_cls()
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
    return dbConfigClass(**dbConfig)


def moveDBLabelToLast(propertiesItems):
    propertiesItems.sort(key=lambda x: 1 if x[0] == 'db_label' else 0)
    