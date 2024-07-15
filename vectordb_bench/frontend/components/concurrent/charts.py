from vectordb_bench.frontend.components.check_results.expanderStyle import (
    initMainExpanderStyle,
)
import plotly.express as px

from vectordb_bench.frontend.config.styles import COLOR_MAP


def drawChartsByCase(allData, showCaseNames: list[str], st):
    initMainExpanderStyle(st)
    for caseName in showCaseNames:
        chartContainer = st.expander(caseName, True)
        caseDataList = [data for data in allData if data["case_name"] == caseName]
        data = [
            {
                "conc_num": caseData["conc_num_list"][i],
                "qps": caseData["conc_qps_list"][i],
                "latency_p99": caseData["conc_latency_p99_list"][i] * 1000,
                "db_name": caseData["db_name"],
                "db": caseData["db"],
            }
            for caseData in caseDataList
            for i in range(len(caseData["conc_num_list"]))
        ]
        drawChart(data, chartContainer)


def getRange(metric, data, padding_multipliers):
    minV = min([d.get(metric, 0) for d in data])
    maxV = max([d.get(metric, 0) for d in data])
    padding = maxV - minV
    rangeV = [
        minV - padding * padding_multipliers[0],
        maxV + padding * padding_multipliers[1],
    ]
    return rangeV


def drawChart(data, st):
    if len(data) == 0:
        return

    x = "latency_p99"
    xrange = getRange(x, data, [0.05, 0.1])

    y = "qps"
    yrange = getRange(y, data, [0.2, 0.1])

    color = "db"
    color_discrete_map = COLOR_MAP
    color = "db_name"
    color_discrete_map = None
    line_group = "db_name"
    text = "conc_num"

    data.sort(key=lambda a: a["conc_num"])

    fig = px.line(
        data,
        x=x,
        y=y,
        color=color,
        color_discrete_map=color_discrete_map,
        line_group=line_group,
        text=text,
        markers=True,
        hover_data={
            "conc_num": True,
        },
        height=720,
    )
    fig.update_xaxes(range=xrange, title_text="Latency P99 (ms)")
    fig.update_yaxes(range=yrange, title_text="QPS")
    fig.update_traces(textposition="bottom right", texttemplate="conc-%{text:,.4~r}")

    st.plotly_chart(
        fig,
        use_container_width=True,
    )
