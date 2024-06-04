from prometheus_client import Counter, Histogram
from streamlit_extras.prometheus import streamlit_registry

registry = streamlit_registry()

SLIDER_COUNT = Counter(
    name="slider_count",
    documentation="Total submitted count of the slider",
    labelnames=("app_name",),
    registry=registry,
)

EXEC_TIME = Histogram(
    name="page_exec_seconds",
    documentation="Execution time of each page run",
    labelnames=("page",),
    registry=registry,
)
