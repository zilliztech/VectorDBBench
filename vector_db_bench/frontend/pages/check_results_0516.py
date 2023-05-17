import streamlit as st
from vector_db_bench.frontend.const import *
from vector_db_bench.models import TaskConfig, CaseConfig, DBCaseConfig
from vector_db_bench.interface import BenchMarkRunner, benchMarkRunner

st.set_page_config(
    page_title="Falcon Mark - Open VectorDB Bench",
    page_icon="🧊",
    # layout="wide",
    initial_sidebar_state="collapsed",
)


st.title("Check Results")

results = benchMarkRunner.get_results()
