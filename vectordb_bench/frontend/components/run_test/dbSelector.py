from vectordb_bench.frontend.const.styles import *
from vectordb_bench.frontend.const.dbCaseConfigs import DB_LIST


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

    # style - image; column gap; checkbox font;
    st.markdown(
        """
        <style>
            div[data-testid='stImage'] {margin: auto;}
            div[data-testid='stHorizontalBlock'] {gap: 8px;}
            .stCheckbox p { color: #000; font-size: 18px; font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    for i, db in enumerate(DB_LIST):
        column = dbContainerColumns[i % DB_SELECTOR_COLUMNS]
        dbIsActived[db] = column.checkbox(db.name)
        column.image(DB_TO_ICON.get(db, ""))
    activedDbList = [db for db in DB_LIST if dbIsActived[db]]

    return activedDbList
