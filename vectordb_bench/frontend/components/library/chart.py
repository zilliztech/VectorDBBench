import plotly.express as px


def drawChartByCase(st, data, labels, **kwargs):
    caseList = list(set([d["case"] for d in data]))
    caseList.sort()
    for case in caseList:
        st.subheader(case)
        caseData = [d for d in data if d["case"] == case]
        for d in caseData:
            for label in labels:
                d[label] = d.get(label, 0)
        st.markdown("##### QPS - Recall")
        drawQPSAndRecallChart(st.container(), caseData, labels=labels, **kwargs)
        st.markdown("##### Load Duration")
        drawLoadDurationChart(st.container(), caseData, labels=labels, **kwargs)


def getRange(metric, data, padding_multipliers=[0.05, 0.1]):
    minV = min([d.get(metric, 0) for d in data])
    maxV = max([d.get(metric, 0) for d in data])
    padding = maxV - minV
    rangeV = [
        minV - padding * padding_multipliers[0],
        maxV + padding * padding_multipliers[1],
    ]
    return rangeV


def drawQPSAndRecallChart(st, data, colorAttr, shapeAttr, textAttr, labels, **kwargs):
    x = "qps"
    xrange = getRange(x, data, [0.05, 0.3])

    y = "recall"
    yrange = getRange(y, data, [0.5, 0.2])
    yrange[0] = max(0, yrange[0])

    data.sort(key=lambda a: a[x])

    fig = px.scatter(
        data,
        x=x,
        y=y,
        text=textAttr,
        color=colorAttr,
        symbol=shapeAttr,
        hover_data={
            textAttr: False,
            colorAttr: False,
            shapeAttr: False,
            **{label: True for label in labels},
        },
    )
    fig.update_xaxes(range=xrange)
    fig.update_yaxes(range=yrange)
    fig.update_traces(textposition="middle right")
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0, pad=8),
        legend=dict(
            orientation="h", yanchor="bottom", y=1, xanchor="right", x=1, title=""
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


def drawLoadDurationChart(st, data, colorAttr, **kwargs):
    x = "load_duration"
    y = "dbLabel"
    
    y_to_data = {
        d[y]: d
        for d in data
    }
    _data = y_to_data.values()
    
    fig = px.bar(
        _data,
        y=y,
        x=x,
        color=colorAttr,
        # barmode="group",
        text_auto=True,
        orientation="h",
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0, pad=8),
        legend=dict(
            orientation="h", yanchor="bottom", y=1, xanchor="right", x=1, title=""
        ),
    )
    st.plotly_chart(fig, use_container_width=True)
