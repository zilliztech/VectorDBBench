from vectordb_bench.frontend.const import *
import streamlit as st


def init_state():
    if DB_SELECT_ALL not in st.session_state:
        st.session_state[DB_SELECT_ALL] = True

    if getSelectAllKey(DB_SELECT_ALL) not in st.session_state:
        st.session_state[getSelectAllKey(DB_SELECT_ALL)] = 0

    if CASE_SELECT_ALL not in st.session_state:
        st.session_state[CASE_SELECT_ALL] = True

    if getSelectAllKey(CASE_SELECT_ALL) not in st.session_state:
        st.session_state[getSelectAllKey(CASE_SELECT_ALL)] = 0
