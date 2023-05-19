import streamlit as st
from vector_db_bench.frontend.const import *
from vector_db_bench.models import TaskConfig, CaseConfig, DBCaseConfig
from vector_db_bench.interface import BenchMarkRunner, benchMarkRunner
from dataclasses import asdict
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Falcon Mark - Open VectorDB Bench",
    page_icon="🧊",
    # layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("Check Results")

results = benchMarkRunner.get_results()
resultSelectOptions = [f"result-{i+1}" for i, result in enumerate(results)]

# Result Seletor
selectorContainer = st.container()
with selectorContainer:
    selectorContainer.header("Results")

    selectedResultSelectedOption = selectorContainer.selectbox(
        "results", resultSelectOptions, label_visibility="hidden"
    )
selectedResult = results[resultSelectOptions.index(selectedResultSelectedOption)]

allData = [
    {
        "db": res.task_config.db.value,
        "case": res.task_config.case_config.case_id.value,
        "db_case_config": res.task_config.db_case_config.model_dump(),
        **asdict(res.metrics),
        "metrics": {key for key, value in asdict(res.metrics).items() if value > 1e-7},
    }
    for res in selectedResult.results
]

dbs = list({d["db"] for d in allData})
cases = list({d["case"] for d in allData})


## Charts
chartContainers = st.container()
with chartContainers:
    for case in cases:
        chartContainer = chartContainers.container()
        chartContainer.header(case)
        with chartContainer:
            data = [d for d in allData if d["case"] == case]
            metrics = set()
            for d in data:
                metrics = metrics.union(d["metrics"])
            metrics = list(metrics)
            fig = make_subplots(rows=len(metrics), cols=1)

            legendContainer = chartContainer.container()
            legendDiv = (
                lambda i: f"""
            <div style="margin-right: 24px; display: flex; align-items: center;">
                <div style="margin-right: 12px; background: {COLOR_MAP[dbs[i]]}; width: {LEGEND_RECT_WIDTH}px; height: {LEGEND_RECT_HEIGHT}px"></div>
                <div style="font-size: {LEGEND_TEXT_FONT_SIZE}px; font-weight: semi-bold;">{dbs[i]}</div>
            </div>
            """
            )

            legendsHtml = " ".join([legendDiv(i) for i, _ in enumerate(dbs)])
            components.html(
                f"""
                <div style="display: flex; float: right">
                    {legendsHtml}
                </div>
                """,
                height=30,
            )

            for row, metric in enumerate(metrics):
                subChartContainer = chartContainers.container()
                title = subChartContainer.container()
                title.markdown(f"**{metric}** (lower is better)")
                chart = subChartContainer.container()
                fig = px.bar(
                    data,
                    x=metric,
                    y="db",
                    color="db",
                    # title=f"{metric}",
                    height=len(dbs) * 30,
                    # barmode="group",
                    # pattern_shape="db",
                    orientation="h",
                    hover_data={
                        "db": True,
                        # 'case': True,
                        # metric: ":.2s",
                    },
                    # hover_data=f"{metric}",
                    color_discrete_map=COLOR_MAP,
                    # text_auto=f".2f",
                    text_auto=True,
                )
                fig.update_xaxes(showticklabels=False, visible=False)
                fig.update_yaxes(showticklabels=False, visible=False)
                fig.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0, pad=4), showlegend=False
                )

                chart.plotly_chart(fig, use_container_width=True)
