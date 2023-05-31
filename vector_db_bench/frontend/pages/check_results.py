import streamlit as st
from vector_db_bench.frontend.const import *
from vector_db_bench.metric import isLowerIsBetterMetric
from vector_db_bench.interface import benchMarkRunner
from dataclasses import asdict
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
from collections import defaultdict
import numpy as np

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
    selectedResultSelectedOptions = selectorContainer.multiselect(
        "results", resultSelectOptions, label_visibility="hidden", default=resultSelectOptions
    )

db_case_flag = defaultdict(lambda: defaultdict(lambda: 0))
selectedResult = []
for option in selectedResultSelectedOptions:
    result = results[resultSelectOptions.index(option)].results
    for res in result:
        selectedResult.append(res)


allData = [
    {
        "db": res.task_config.db.value,
        "case": res.task_config.case_config.case_id.value,
        "db_case_config": res.task_config.db_case_config.dict(),
        **asdict(res.metrics),
        "metrics": {key for key, value in asdict(res.metrics).items() if value > 1e-7},
    }
    for res in selectedResult
]

# allData = [
#     {
#         "db": DB_LIST[i % len(DB_LIST)].value,
#         "case": res.task_config.case_config.case_id.value,
#         "db_case_config": res.task_config.db_case_config.dict(),
#         **{
#             key: value * np.random.random()
#             for key, value in asdict(res.metrics).items()
#         },
#         "metrics": {key for key, value in asdict(res.metrics).items() if value > 1e-7},
#     }
#     for i, res in enumerate(selectedResult)
# ]

dbs = list({d["db"] for d in allData})
cases = list({d["case"] for d in allData})


## Charts
chartContainers = st.container()
with chartContainers:
    for case in cases:
        chartContainer = chartContainers.container()
        chartContainer.header(case)
        dbCount = defaultdict(int)
        with chartContainer:
            data = [d for d in allData if d["case"] == case]
            for d in data:
                d["alias"] = dbCount[d["db"]]
                d["db_name"] = (
                    f"{d['db']}-{dbCount[d['db']]}" if dbCount[d["db"]] > 0 else d["db"]
                )
                dbCount[d["db"]] += 1
            metrics = set()
            for d in data:
                metrics = metrics.union(d["metrics"])
            metrics = list(metrics)
            fig = make_subplots(rows=len(metrics), cols=1)

            legendContainer = chartContainer.container()
            legendDiv = (
                lambda i: f"""
            <div style="margin-right: 20px; display: flex; align-items: center;">
                <div style="margin-right: 10px; background: {COLOR_MAP[dbs[i]]}; width: {LEGEND_RECT_WIDTH}px; height: {LEGEND_RECT_HEIGHT}px"></div>
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
                height=((len(dbs) - 1) // 5 + 1) * 30,
            )

            for row, metric in enumerate(metrics):
                subChartContainer = chartContainers.container()
                title = subChartContainer.container()
                title.markdown(
                    f"**{metric}** ({'less' if isLowerIsBetterMetric(metric) else 'more'} is better)"
                )
                chart = subChartContainer.container()
                dataWithMetric = [d for d in data if d.get(metric, 0) > 1e-7]
                height = len(dataWithMetric) * 28
                fig = px.bar(
                    dataWithMetric,
                    x=metric,
                    y="db_name",
                    color="db",
                    # title=f"{metric}",
                    height=height,
                    # barmode="group",
                    pattern_shape="alias",
                    pattern_shape_sequence=["", "+", "\\", ".", "|", "/", "-"],
                    orientation="h",
                    hover_data={
                        "db": False,
                        "alias": False,
                        "db_name": True,
                        # 'case': True,
                        # metric: ":.2s",
                    },
                    # hover_data=f"{metric}",
                    color_discrete_map=COLOR_MAP,
                    # text_auto=f".2f",
                    text_auto=True,
                    # text=metric,
                    # template="ggplot2",
                )
                fig.update_xaxes(showticklabels=False, visible=False)
                fig.update_yaxes(showticklabels=False, visible=False)
                fig.update_traces(
                    textposition="outside",
                    marker=dict(
                        # size=[10, 50, 60],
                        # color="red"
                        # line_color="black",
                        # line=dict(color="MediumPurple", width=2)
                        pattern=dict(
                            fillmode="overlay", fgcolor="#fff", fgopacity=1, size=7
                        )
                    ),
                )
                fig.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0, pad=4), showlegend=False
                )

                chart.plotly_chart(fig, use_container_width=True)
