from streamlit.runtime.media_file_storage import MediaFileStorageError
from vectordb_bench.frontend.config.styles import DB_SELECTOR_COLUMNS, DB_TO_ICON
from vectordb_bench.frontend.config.dbCaseConfigs import DB_LIST


def dbSelector(st):
    st.markdown(
        "<div style='height: 12px;'></div>",
        unsafe_allow_html=True,
    )
    st.subheader("STEP 1: Select the database(s)")
    st.markdown(
        "<div style='color: #647489; margin-bottom: 24px; margin-top: -12px;'>Choose at least one case you want to run the test for. </div>",
        unsafe_allow_html=True,
    )

    dbContainerColumns = st.columns(DB_SELECTOR_COLUMNS, gap="small")
    dbIsActived = {db: False for db in DB_LIST}

    for i, db in enumerate(DB_LIST):
        column = dbContainerColumns[i % DB_SELECTOR_COLUMNS]
        dbIsActived[db] = column.checkbox(db.name)
        try:
            column.image(DB_TO_ICON.get(db, ""))
        except MediaFileStorageError:
            column.warning(f"{db.name} image not available")
            pass
    activedDbList = [db for db in DB_LIST if dbIsActived[db]]

    return activedDbList
