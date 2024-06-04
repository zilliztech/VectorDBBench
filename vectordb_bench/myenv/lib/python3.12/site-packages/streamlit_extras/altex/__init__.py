from functools import partial
from typing import Optional, Union

import altair as alt
import numpy as np
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
def _url_to_dataframe(url: str) -> pd.DataFrame:
    """Collects a CSV/JSON file from a URL and load it into a dataframe, with appropriate caching (memo)

    Args:
        url (str): URL of the CSV/JSON file

    Returns:
        pd.DataFrame: Resulting dataframe
    """
    if url.endswith(".csv"):
        return pd.read_csv(url)
    elif url.endswith(".json"):
        return pd.read_json(url)
    else:
        raise Exception("URL must end with .json or .csv")


weather_data_url = (
    "https://raw.githubusercontent.com/tvst/plost/master/data/seattle-weather.csv"
)
stocks_data_url = (
    "https://raw.githubusercontent.com/vega/vega/main/docs/data/stocks.csv"
)
barley_data_url = (
    "https://raw.githubusercontent.com/vega/vega/main/docs/data/barley.json"
)


def get_weather_data():
    return _url_to_dataframe(weather_data_url)


def get_stocks_data():
    return _url_to_dataframe(stocks_data_url).assign(
        date=lambda df: pd.to_datetime(df.date)
    )


def get_barley_data():
    return _url_to_dataframe(barley_data_url)


def get_random_data():
    return pd.DataFrame(
        np.random.randn(20, 7),
        columns=list("abcdefg"),
    ).reset_index()


def _drop_nones(iterable: Union[dict, list]):
    """Remove nones for iterable.
    If dict, drop keys when value is None
    If list, drop values when value equal None

    Args:
        iterable (Union[dict, str]): Input iterable

    Raises:
        Exception: TypeError

    Returns:
        Union[dict, str]: Input interable without Nones
    """
    if isinstance(iterable, dict):
        return {k: v for k, v in iterable.items() if v is not None}
    if isinstance(iterable, list):
        return [x for x in iterable if x is not None]
    raise TypeError(f"Iterable of type {type(iterable)} is not supported")


def _get_shorthand(param: Union[str, alt.X, alt.Y]):
    """Get Altair shorthand from parameter, if exists

    Args:
        param (Union[str, alt.X, alt.Y]): Param x/y

    Returns:
        str: Parameter itself or shorthand when alt.X/alt.Y object
    """
    if param is None:
        return None
    elif not isinstance(param, str):
        return param.shorthand
    else:
        return param


def _update_axis_config(
    axis: Union[alt.X, alt.Y, str], output_type: Union[alt.X, alt.Y], updates: dict
) -> Union[alt.X, alt.Y]:
    """Update x and y configs

    Args:
        axis (Union[alt.X, alt.Y, str]): Chart input for x
        output_type (Union[alt.X, alt.Y]): Chart input for y

    Raises:
        Exception: TypeError when input has invalid type

    Examples:
    >>> _update_axis_config(alt.X("x"), alt.Y, {"axis": "None"})

    Returns:
        alt.X/alt.Y: Updated config for x/y
    """
    if isinstance(axis, (alt.X, alt.Y)):
        axis_config = axis.to_dict()
        for key, value in updates.items():
            axis_config[key] = value
        return output_type(**axis_config)
    elif isinstance(axis, str):
        return output_type(shorthand=axis, **updates)
    else:
        raise TypeError("Input x/y must be of type str or alt.X or alt.Y")


@extra
def _chart(
    mark_function: str,
    data: pd.DataFrame,
    x: Union[alt.X, str],
    y: Union[alt.Y, str],
    color: Optional[Union[alt.Color, str]] = None,
    opacity: Optional[Union[alt.value, float]] = None,
    column: Optional[Union[alt.Column, str]] = None,
    rolling: Optional[int] = None,
    title: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    spark: bool = False,
    autoscale_y: bool = False,
) -> alt.Chart:
    """Create an Altair chart with a simple API.
    Supported charts include line, bar, point, area, histogram, sparkline, sparkbar, sparkarea.

    Args:
        mark_function (str): Altair mark function, example line/bar/point
        data (pd.DataFrame): Dataframe to use for the chart
        x (Union[alt.X, str]): Column for the x axis
        y (Union[alt.Y, str]): Column for the y axis
        color (Optional[Union[alt.Color, str]], optional): Color a specific group of your data. Defaults to None.
        opacity (Optional[Union[alt.value, float]], optional): Change opacity of marks. Defaults to None.
        column (Optional[Union[alt.Column, str]], optional): Groupby a specific column. Defaults to None.
        rolling (Optional[int], optional): Rolling average window size. Defaults to None.
        title (Optional[str], optional): Title of the chart. Defaults to None.
        width (Optional[int], optional): Width of the chart. Defaults to None.
        height (Optional[int], optional): Height of the chart. Defaults to None.
        spark (bool, optional): Whether or not to make spark chart, i.e. a chart without axes nor ticks nor legend. Defaults to False.
        autoscale_y (bool, optional): Whether or not to autoscale the y axis. Defaults to False.

    Returns:
        alt.Chart: Altair chart
    """

    x_ = _get_shorthand(x)
    y_ = _get_shorthand(y)
    color_ = _get_shorthand(color)

    tooltip_config = _drop_nones([x_, y_, color_])

    chart_config = _drop_nones(
        {
            "data": data,
            "title": title,
            "mark": mark_function,
            "width": width,
            "height": height,
        }
    )

    chart = alt.Chart(**chart_config)

    if rolling is not None:
        rolling_column = f"{y_} ({rolling}-average)"
        y = f"{rolling_column}:Q"
        transform_config = {
            rolling_column: f"mean({y_})",
            "frame": [-rolling, 0],
            "groupby": [str(color)],
        }
        chart = chart.transform_window(**transform_config)

    if spark:
        chart = chart.configure_view(strokeWidth=0).configure_axis(
            grid=False, domain=False
        )
        x_axis = _update_axis_config(x, alt.X, {"axis": None})
        y_axis = _update_axis_config(y, alt.Y, {"axis": None})
    else:
        x_axis = x
        y_axis = y

    if autoscale_y:
        y_axis = _update_axis_config(y_axis, alt.Y, {"scale": alt.Scale(zero=False)})

    encode_config = _drop_nones(
        {
            "x": x_axis,
            "y": y_axis,
            "color": color,
            "tooltip": tooltip_config,
            "opacity": alt.value(opacity) if isinstance(opacity, float) else opacity,
            "column": column,
        }
    )

    chart = chart.encode(**encode_config)

    return chart


def chart(use_container_width: bool = True, **kwargs):
    """Display an Altair chart in Streamlit

    Args:
        **kwargs: See function _chart()
        use_container_width (bool, optional): Whether or not the displayed chart uses all available width. Defaults to True.
    """

    if "width" in kwargs:
        use_container_width = False

    st.altair_chart(
        _chart(**kwargs),
        use_container_width=use_container_width,
    )


def _partial(*args, **kwargs):
    """Alternative to 'functools.partial' where __name__ attribute
    can be set manually, since the default partial does not create it.
    """
    __name__ = kwargs.pop("__name__", "foo")
    func = partial(*args, **kwargs)
    func.__name__ = __name__
    return func


@extra
def scatter_chart(**kwargs):
    return chart(mark_function="point", __name__="scatter_chart", **kwargs)


# scatter_chart = _partial(chart, mark_function="point", __name__="scatter_chart")
line_chart = _partial(chart, mark_function="line", __name__="line_chart")
area_chart = _partial(chart, mark_function="area", __name__="area_chart")
bar_chart = _partial(chart, mark_function="bar", __name__="bar_chart")
hist_chart = _partial(bar_chart, y="count()", __name__="hist_chart")
sparkline_chart = _partial(line_chart, spark=True, __name__="sparkline_chart")
sparkbar_chart = _partial(bar_chart, spark=True, __name__="sparkbar_chart")
sparkhist_chart = _partial(hist_chart, spark=True, __name__="sparkhist_chart")
sparkarea_chart = _partial(area_chart, spark=True, __name__="sparkarea_chart")


@cache_data
def example_line():
    stocks = get_stocks_data()

    line_chart(
        data=stocks.query("symbol == 'GOOG'"),
        x="date",
        y="price",
        title="A beautiful simple line chart",
    )


@cache_data
def example_multi_line():
    stocks = get_stocks_data()
    line_chart(
        data=stocks,
        x="date",
        y="price",
        color="symbol",
        title="A beautiful multi line chart",
    )


@cache_data
def example_bar():
    stocks = get_stocks_data()
    bar_chart(
        data=stocks.query("symbol == 'GOOG'"),
        x="date",
        y="price",
        title="A beautiful bar chart",
    )


@cache_data
def example_hist():
    stocks = get_stocks_data()
    hist_chart(
        data=stocks.assign(price=stocks.price.round(0)),
        x="price",
        title="A beautiful histogram",
    )


@cache_data
def example_scatter_opacity():
    weather = get_weather_data()
    scatter_chart(
        data=weather,
        x=alt.X("wind:Q", title="Custom X title"),
        y=alt.Y("temp_min:Q", title="Custom Y title"),
        title="A beautiful scatter chart with custom opacity",
        opacity=0.2,
    )


@cache_data
def example_bar_horizontal():
    weather = get_weather_data()
    bar_chart(
        data=weather.head(15),
        x="temp_max:Q",
        y=alt.Y("date:O", title="Temperature"),
        title="A beautiful horizontal bar chart",
    )


@cache_data
def example_bar_log():
    weather = get_weather_data()
    bar_chart(
        data=weather,
        x=alt.X("temp_max:Q", title="Temperature"),
        y=alt.Y(
            "count()",
            title="Count of records",
            scale=alt.Scale(type="symlog"),
        ),
        title="A beautiful histogram... with log scale",
    )


@cache_data
def example_bar_sorted():
    weather = get_weather_data()
    bar_chart(
        data=weather.sort_values(by="temp_max", ascending=False).head(25),
        x=alt.X("date", sort="-y"),
        y=alt.Y("temp_max:Q"),
        title="A beautiful sorted-by-value bar chart",
    )


@cache_data
def example_scatter():
    weather = get_weather_data()
    scatter_chart(
        data=weather,
        x=alt.X("wind:Q", title="Custom X title"),
        y=alt.Y("temp_min:Q", title="Custom Y title"),
        title="A beautiful scatter chart",
    )


@cache_data
def example_hist_time():
    weather = get_weather_data()
    hist_chart(
        data=weather,
        x="week(date):T",
        y="day(date):T",
        color=alt.Color(
            "median(temp_max):Q",
            legend=None,
        ),
        title="A beautiful time hist chart",
    )


@cache_data
def example_sparkline():
    stocks = get_stocks_data()
    sparkline_chart(
        data=stocks.query("symbol == 'GOOG'"),
        x="date",
        y="price",
        title="A beautiful sparkline chart",
        rolling=7,
        height=150,
    )


@cache_data
def example_minisparklines():
    stocks = get_stocks_data()

    left, middle, right = st.columns(3)
    with left:
        data = stocks.query("symbol == 'GOOG'")
        st.metric("GOOG", int(data["price"].mean()))
        sparkline_chart(
            data=data,
            x="date",
            y="price:Q",
            height=80,
            autoscale_y=True,
        )
    with middle:
        data = stocks.query("symbol == 'MSFT'")
        st.metric("MSFT", int(data["price"].mean()))
        sparkline_chart(
            data=data,
            x="date",
            y="price:Q",
            height=80,
            autoscale_y=True,
        )
    with right:
        data = stocks.query("symbol == 'AAPL'")
        st.metric("AAPL", int(data["price"].mean()))
        sparkline_chart(
            data=data,
            x="date",
            y="price:Q",
            height=80,
            autoscale_y=True,
        )


@cache_data
def example_sparkbar():
    stocks = get_stocks_data()
    sparkbar_chart(
        data=stocks.query("symbol == 'GOOG'"),
        x="date",
        y="price",
        title="A beautiful sparkbar chart",
        height=150,
    )


@cache_data
def example_sparkarea():
    random_data = get_random_data()
    df = pd.melt(
        random_data,
        id_vars="index",
        value_vars=list("abcdefg"),
    )

    sparkarea_chart(
        data=df,
        x="index",
        y="value",
        color=alt.Color("variable", legend=None),
        title="A beautiful (also probably useless) sparkarea chart",
        opacity=alt.value(0.6),
        height=200,
    )


@cache_data
def example_bar_stacked():
    barley = get_barley_data()
    bar_chart(
        data=barley,
        x=alt.X("variety", title="Variety"),
        y="sum(yield)",
        color="site",
        title="A beautiful stacked bar chart",
    )


@cache_data
def example_bar_normalized():
    barley = get_barley_data()
    bar_chart(
        data=barley,
        x=alt.X("variety:N", title="Variety"),
        y=alt.Y("sum(yield):Q", stack="normalize"),
        color="site:N",
        title="A beautiful normalized stacked bar chart",
    )


@cache_data
def example_bar_normalized_custom():
    barley = get_barley_data()
    bar_chart(
        data=barley,
        x=alt.X("variety", title="Variety"),
        y="sum(yield)",
        color=alt.Color("site", scale=alt.Scale(scheme="lighttealblue"), legend=None),
        title="A beautiful stacked bar chart (without legend, custom colors)",
    )


@cache_data
def example_bar_grouped():
    barley = get_barley_data()
    bar_chart(
        data=barley,
        x="year:O",
        y="sum(yield):Q",
        color="year:N",
        column="site:N",
        title="A beautiful grouped bar charts",
        width=90,
        use_container_width=False,
    )


__title__ = "Altex"
__desc__ = (
    "A simple wrapper on top of Altair to make Streamlit charts in an"
    " express API. If you're lazy and/or familiar with Altair, this is "
    " probably a good fit! Inspired by plost and plotly-express."
)
__icon__ = "👸"
__examples__ = {
    example_line: [line_chart, get_stocks_data],
    example_multi_line: [line_chart, get_stocks_data],
    example_bar: [bar_chart, get_stocks_data],
    example_hist: [hist_chart, get_stocks_data],
    example_scatter: [scatter_chart, get_weather_data],
    example_sparkline: [sparkline_chart, get_stocks_data],
    example_minisparklines: [sparkline_chart, get_stocks_data],
    example_sparkbar: [sparkbar_chart, get_stocks_data],
    example_sparkarea: [sparkarea_chart, get_random_data],
    example_hist_time: [hist_chart, get_weather_data],
    example_bar_sorted: [bar_chart, get_weather_data],
    example_bar_normalized: [bar_chart, get_barley_data],
    example_bar_grouped: [bar_chart, get_barley_data],
    example_bar_horizontal: [bar_chart, get_weather_data],
    example_bar_log: [bar_chart, get_weather_data],
    example_scatter_opacity: [scatter_chart, get_weather_data],
    example_bar_normalized_custom: [bar_chart, get_barley_data],
}
__author__ = "Arnaud Miribel"
__experimental_playground__ = False
