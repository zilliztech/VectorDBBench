from __future__ import annotations

from functools import partial
from typing import Iterable, Tuple

import altair as alt
import pandas as pd
import streamlit as st

try:
    from streamlit import cache_data  # streamlit >= 1.18.0
except ImportError:
    from streamlit import experimental_memo as cache_data  # streamlit >= 0.89

from .. import extra

try:
    from altair.utils.plugin_registry import NoSuchEntryPoint
except ImportError:
    from entrypoints import NoSuchEntryPoint

try:
    alt.themes.enable("streamlit")
except NoSuchEntryPoint:
    st.altair_chart = partial(st.altair_chart, theme="streamlit")


@cache_data
def get_data() -> pd.DataFrame:
    source = pd.read_csv(
        "https://raw.githubusercontent.com/vega/vega-datasets/next/data/stocks.csv"
    )
    source = source[source.date.gt("2004-01-01")]
    return source


@cache_data(ttl=60 * 60 * 24)
def get_chart(data: pd.DataFrame) -> alt.Chart:
    hover = alt.selection_single(
        fields=["date"],
        nearest=True,
        on="mouseover",
        empty="none",
    )

    lines = (
        alt.Chart(data, height=500, title="Evolution of stock prices")
        .mark_line()
        .encode(
            x=alt.X("date", title="Date"),
            y=alt.Y("price", title="Price"),
            color="symbol",
        )
    )

    # Draw points on the line, and highlight based on selection
    points = lines.transform_filter(hover).mark_circle(size=65)

    # Draw a rule at the location of the selection
    tooltips = (
        alt.Chart(data)
        .mark_rule()
        .encode(
            x="yearmonthdate(date)",
            y="price",
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("date", title="Date"),
                alt.Tooltip("price", title="Price (USD)"),
            ],
        )
        .add_selection(hover)
    )

    return (lines + points + tooltips).interactive()


@extra
def get_annotations_chart(
    annotations: Iterable[Tuple],
    y: float = 0,
    min_date: str | None = None,
    max_date: str | None = None,
    marker: str = "⬇",
    marker_size: float = 20,
    marker_offset_x: float = 0,
    market_offset_y: float = -10,
    marker_align: str = "center",
) -> alt.Chart:
    """
    Creates an Altair Chart with annotation markers on the horizontal axis.
    Useful to highlight certain events on top of another time series Altair Chart.
    More here https://share.streamlit.io/streamlit/example-app-time-series-annotation/main

    Args:
        annotations (Iterable[Tuple]): Iterable of annotations defined by tuples with date and annotation.
        y (float, optional): Height at which the annotation marker should be. Defaults to 0.
        min_date (str, optional): Only annotations older than min_date will be displayed. Defaults to None.
        max_date (str, optional): Only annotations more recent than max_date will be displayed. Defaults to None.
        marker (str, optional): Marker to be used to indicate there is an annotation. Defaults to "⬇".
        marker_size (float, optional): Size of the marker (font size). Defaults to 20.
        marker_offset_x (float, optional): Horizontal offset. Defaults to 0.
        market_offset_y (float, optional): Vertical offset. Defaults to -10.
        marker_align (str, optional): Text-align property of the marker ("left", "right", "center"). Defaults to "center".

    Returns:
        alt.Chart: Altair Chart with annotation markers on the horizontal axis
    """

    # Make a DataFrame for annotations
    annotations_df = pd.DataFrame(
        annotations,
        columns=["date", "annotation"],
    )

    annotations_df.date = pd.to_datetime(annotations_df.date)
    annotations_df["y"] = y
    if min_date:
        annotations_df = annotations_df[annotations_df.date.gt(min_date)]
    if max_date:
        annotations_df = annotations_df[annotations_df.date.lt(max_date)]

    encode_params = {"x": "date:T", "y": alt.Y("y:Q"), "tooltip": "annotation"}

    if "url" in annotations_df.columns:
        encode_params["href"] = "url"

    annotations_chart = (
        alt.Chart(annotations_df)
        .mark_text(
            size=marker_size,
            text=marker,
            dy=market_offset_y,
            dx=marker_offset_x,
            align=marker_align,
            color="black",
        )
        .encode(**encode_params)
    )

    return annotations_chart


def example() -> None:
    data: pd.DataFrame = get_data()
    chart: alt.TopLevelMixin = get_chart(data=data)

    chart += get_annotations_chart(
        annotations=[
            ("Mar 01, 2008", "Pretty good day for GOOG"),
            ("Dec 01, 2007", "Something's going wrong for GOOG & AAPL"),
            ("Nov 01, 2008", "Market starts again thanks to..."),
            ("Dec 01, 2009", "Small crash for GOOG after..."),
        ],
    )

    st.altair_chart(chart, use_container_width=True)  # type: ignore


__title__ = "Chart annotations"
__desc__ = "Add annotations to specific timestamps in your time series in Altair!"
__icon__ = "⬇"
__examples__ = [example]
__author__ = "Arnaud Miribel"
__github__repo__ = "https://github.com/streamlit/example-app-time-series-annotation"
__streamlit_cloud_url__ = "https://streamlit-example-app-time-series-annotati-streamlit-app-vmbrzi.streamlitapp.com/"
