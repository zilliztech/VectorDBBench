import streamlit as st
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
from vector_db_bench.frontend.const import *
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Falcon Mark - Check Results",
    page_icon="🧊",
    # layout="wide",
    initial_sidebar_state="collapsed",
)


st.title("Result Check")

results = ["result_1", "result_2", "result_3", "result_4", "result_5"]
caseCount = 3
dbCount = 4
metricCount = 6

dbs = [f"db-{db_id}" for db_id in range(dbCount)]
cases = [f"case-{case_id}" for case_id in range(caseCount)]
metrics = [f"metric-{metric_id}" for metric_id in range(metricCount)]

resultsData = [
    {
        "db": db,
        "case": case,
        **{metric: np.random.random() * 10**i for i, metric in enumerate(metrics)},
    }
    for case in cases
    for db in dbs
]

# print("resultsData", resultsData)


# Result Seletor
selectorContainer = st.container()
with selectorContainer:
    selectorContainer.header("Results")
    selectedResult = selectorContainer.selectbox(
        "results", results, label_visibility="hidden"
    )
    # selectedResult = selectorContainer.multiselect('', results, max_selections=1)


# Result Tables

# Result Charts
# allData = pd.DataFrame(resultsData)
allData = resultsData
chartContainers = st.container()


with chartContainers:
    chartContainers.header("Chart")

    for caseId in range(caseCount):
        case = f"case-{caseId}"
        chartContainer = chartContainers.container()
        chartContainer.header(case)

        with chartContainer:
            fig = make_subplots(rows=len(metrics), cols=1)

            legendContainer = chartContainer.container()
            legendDiv = lambda i: f"""
            <div style="margin-right: 24px; display: flex; align-items: center;">
                <div style="margin-right: 12px; background: {COLOR_SCHEME[i]}; width: {LEGEND_RECT_WIDTH}px; height: {LEGEND_RECT_HEIGHT}px"></div>
                <div style="font-size: {LEGEND_TEXT_FONT_SIZE}px; font-weight: semi-bold;">{dbs[i]}</div>
            </div>
            """
            
            legendsHtml = " ".join([legendDiv(i) for i in range(dbCount)])
            components.html(
                f"""
                <div style="display: flex; float: right">
                    {legendsHtml}
                </div>
                """,
                height=30,
            )

            data = [d for d in allData if d["case"] == case]

            for row, metric in enumerate(metrics):
                subChartContainer = chartContainers.container()
                fig = px.bar(
                    data,
                    x=metric,
                    y="db",
                    color="db",
                    title="",
                    height=dbCount * 30,
                    # barmode="group",
                    # pattern_shape="db",
                    orientation="h",
                    hover_data={
                        "db": True,
                        # 'case': True,
                        metric: ":.2f",
                    },
                    # hover_data=f"{metric}",
                    color_discrete_sequence=COLOR_SCHEME,
                    text_auto=f".2f",
                )
                fig.update_xaxes(showticklabels=False, visible=False)
                fig.update_yaxes(showticklabels=False, title=metric)
                fig.update_layout(
                    margin=dict(l=20, r=20, t=0, b=0, pad=4), showlegend=False
                )

                subChartContainer.plotly_chart(fig, use_container_width=True)


# Share
