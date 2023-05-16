import streamlit as st
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px

st.set_page_config(
    page_title="Falcon Mark - Check Results",
    page_icon="🧊",
    # layout="wide",
    initial_sidebar_state="collapsed",
)


st.title("Result Check")

results = ["result_1", "result_2", "result_3", "result_4", "result_5"]
caseCount = 3
dbCount = 2

resultsData = [
    {
        "db": f"db-{db}",
        "case": f"case-{case}",
        **{
            f"metric-{metric}": np.random.random() * 10**metric for metric in range(6)
        },
    }
    for case in range(caseCount)
    for db in range(dbCount)
]


def standardlize(data, metrics, fromZero=False):
    ranges = [
        [
            0 if fromZero else min([d.get(metric, 0) for d in data]),
            max([d.get(metric, 0) for d in data]),
        ]
        for metric in metrics
    ]
    return [
        {
            **d,
            **{
                f"format_{metric}": (d.get(metric, 0) - ranges[i][0])
                / (ranges[i][1] - ranges[i][0])
                for i, metric in enumerate(metrics)
            },
        }
        for d in data
    ]


metrics = [f"metric-{metric}" for metric in range(6)]

standardData = standardlize(resultsData, metrics, True)


def flatData(data, metrics):
    return [
        {
            **d,
            "metric": metric,
            "format_value": d[f"format_{metric}"],
            "value": d[metric],
        }
        for d in data
        for metric in metrics
    ]


allData = flatData(standardData, metrics)

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
chartContainers = st.container()
with chartContainers:
    chartContainers.header("Chart")

    for caseId in range(caseCount):
        case = f"case-{caseId}"
        chartContainer = chartContainers.container()
        chartContainer.header(case)

        with chartContainer:
            data = [d for d in allData if d["case"] == case]

            fig = px.bar(
                data,
                x="format_value",
                y="metric",
                color="db",
                title="",
                barmode="group",
                # pattern_shape="db",
                # text="value",
                # texttemplate="%{metric}",
                orientation="h",
                hover_data={
                    # "metric-1": ":.2f",
                    # f"format_{metric}": False,
                    # "db": False,
                    "metric": False,
                    "format_value": False,
                    "value": True,
                }
                # text_auto=True,
            )
            fig.update_xaxes(showticklabels=False, visible=False)
            fig.update_yaxes(title="")
            st.plotly_chart(fig, use_container_width=True)


caseChartContainers = st.container()
with caseChartContainers:
    caseChartContainers.header("Metric Chart")

    for metric in metrics:
        chartContainer = caseChartContainers.container()
        chartContainer.header(metric)

        with chartContainer:
            data = [d for d in allData if d["metric"] == metric]

            fig = px.bar(
                data,
                x="format_value",
                y="case",
                color="db",
                title="",
                barmode="group",
                # pattern_shape="db",
                orientation="h",
                hover_data={
                    # "metric-1": ":.2f",
                    # f"format_{metric}": False,
                    # "db": False,
                    "metric": False,
                    "case": False,
                    "format_value": False,
                    "value": True,
                }
                # text_auto=True,
            )
            fig.update_xaxes(showticklabels=False, visible=False)
            fig.update_yaxes(title="")
            st.plotly_chart(fig, use_container_width=True)


# Share
