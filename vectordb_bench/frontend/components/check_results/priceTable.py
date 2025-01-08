from vectordb_bench.backend.clients import DB
import pandas as pd
from collections import defaultdict
import streamlit as st

from vectordb_bench.frontend.config.dbPrices import DB_DBLABEL_TO_PRICE


def priceTable(container, data):
    dbAndLabelSet = {(d["db"], d["db_label"]) for d in data if d["db"] != DB.Milvus.value}

    dbAndLabelList = list(dbAndLabelSet)
    dbAndLabelList.sort()

    table = pd.DataFrame(
        [
            {
                "DB": db,
                "Label": db_label,
                "Price per hour": DB_DBLABEL_TO_PRICE.get(db, {}).get(db_label, 0),
            }
            for db, db_label in dbAndLabelList
        ]
    )
    height = len(table) * 35 + 38

    expander = container.expander("Price List (Editable).")
    editTable = expander.data_editor(
        table,
        use_container_width=True,
        hide_index=True,
        height=height,
        disabled=("DB", "Label"),
        column_config={
            "Price per hour": st.column_config.NumberColumn(
                min_value=0,
                format="$ %f",
            )
        },
    )

    priceMap = defaultdict(lambda: defaultdict(float))
    for _, row in editTable.iterrows():
        db, db_label, price = row
        priceMap[db][db_label] = price

    return priceMap
