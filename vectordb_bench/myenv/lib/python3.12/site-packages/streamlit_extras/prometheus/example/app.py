import time
from random import random

import streamlit as st
from metrics import EXEC_TIME, SLIDER_COUNT

st.header("Streamlit app with Prometheus metrics")

"""
1. Enter an app name and slider value and press Submit
1. At a terminal, do `curl localhost:8501/_stcore/metrics` to view the metrics generated
1. Note you can run this across multiple sessions and it aggregates the counter
"""

start_time = time.time()


app_name = st.text_input("App name", "prometheus_app")
latest = st.slider("Latest value", 0, 20, 3)
if st.button("Submit"):
    SLIDER_COUNT.labels(app_name).inc(latest)
    st.toast("Successfully submitted")

# Add a little variability to the response
time.sleep(random() / 2)
exec_time = time.time() - start_time
st.write(f"Exec time was {exec_time}")
EXEC_TIME.labels("main").observe(exec_time)
