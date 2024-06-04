import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from faker.providers import BaseProvider
from matplotlib import pyplot as plt
from streamlit_extras import altex

from .common import st_command_with_default

try:
    from streamlit import cache_data, cache_resource  # streamlit >= 1.18.0
    
except ImportError:
    from streamlit import experimental_memo as cache_data, experimental_singleton as cache_resource  # streamlit >= 0.89


@cache_data
def url_to_dataframe(url: str, parse_dates: list = ["date"]) -> pd.DataFrame:
    """Collects a CSV/JSON file from a URL and load it into a dataframe, with appropriate caching (memo)
    Args:
        url (str): URL of the CSV/JSON file
        parse_dates (list): Columns where date parsing should be done
    Returns:
        pd.DataFrame: Resulting dataframe
    """
    if url.endswith(".csv"):
        return pd.read_csv(url, parse_dates=parse_dates)
    elif url.endswith(".json"):
        return pd.read_json(url)
    else:
        raise Exception("URL must end with .json or .csv")


SEATTLE_WEATHER_URL = (
    "https://raw.githubusercontent.com/tvst/plost/master/data/seattle-weather.csv"
)
SP500_URL = "https://raw.githubusercontent.com/tvst/plost/master/data/sp500.csv"

STOCKS_DATA_URL = (
    "https://raw.githubusercontent.com/vega/vega/main/docs/data/stocks.csv"
)
BARLEY_DATA_URL = (
    "https://raw.githubusercontent.com/vega/vega/main/docs/data/barley.json"
)


@cache_resource
def get_datasets():
    N = 50
    rand = pd.DataFrame()
    rand["a"] = np.arange(N)
    rand["b"] = np.random.rand(N)
    rand["c"] = np.random.rand(N)

    N = 500
    events = pd.DataFrame()
    events["time_delta_s"] = np.random.randn(N)
    events["servers"] = np.random.choice(["server 1", "server 2", "server 3"], N)

    N = 500
    randn = pd.DataFrame(
        np.random.randn(N, 4),
        columns=["a", "b", "c", "d"],
    )

    N = 200
    pageviews = pd.DataFrame()
    pageviews["pagenum"] = [f"page-{i:03d}" for i in range(N)]
    pageviews["pageviews"] = np.random.randint(0, 1000, N)

    return dict(
        rand=rand,
        randn=randn,
        events=events,
        pageviews=pageviews,
        stocks=url_to_dataframe(STOCKS_DATA_URL),
        seattle_weather=url_to_dataframe(SEATTLE_WEATHER_URL),
        sp500=url_to_dataframe(SP500_URL),
        barley=url_to_dataframe(BARLEY_DATA_URL),
    )


datasets = get_datasets()


@cache_data
def _line_chart():
    st.line_chart(
        data=datasets["stocks"].query("symbol == 'GOOG'"),
        x="date",
        y="price",
        # title="A beautiful simple line chart",
    )


@cache_data
def _multi_line_chart():
    altex.line_chart(
        data=datasets["stocks"],
        x="date",
        y="price",
        color="symbol",
        # title="A beautiful multi line chart",
    )


def _bar_chart():
    symbol = np.random.choice(datasets["stocks"].symbol.unique(), 1)[0]
    st.bar_chart(
        data=datasets["stocks"].query(
            f"(symbol == '{symbol}') & (date >= '2006-01-01')"
        ),
        x="date",
        y="price",
        # title="A beautiful bar chart",
    )


@cache_data
def _hist_chart():
    altex.hist_chart(
        data=datasets["stocks"].assign(price=datasets["stocks"].price.round(0)),
        x="price",
        title="A beautiful histogram",
    )


@cache_data
def _scatter_chart():
    altex.scatter_chart(
        data=datasets["seattle_weather"],
        x=alt.X("wind:Q", title="Custom X title"),
        y=alt.Y("temp_min:Q", title="Custom Y title"),
        # title="A beautiful scatter chart with custom opacity",
        opacity=0.2,
    )


@cache_data
def _bar_chart_horizontal():
    altex.bar_chart(
        data=datasets["seattle_weather"].head(15),
        x="temp_max:Q",
        y=alt.Y("date:O", title="Temperature"),
        # title="A beautiful horizontal bar chart",
    )


@cache_data
def _bar_chart_log():
    altex.bar_chart(
        data=datasets["seattle_weather"],
        x=alt.X("temp_max:Q", title="Temperature"),
        y=alt.Y(
            "count()",
            title="Count of records",
            scale=alt.Scale(type="symlog"),
        ),
        # title="A beautiful histogram... with log scale",
    )


@cache_data
def _bar_chart_sorted():
    altex.bar_chart(
        data=datasets["seattle_weather"]
        .sort_values(by="temp_max", ascending=False)
        .head(25),
        x=alt.X("date", sort="-y"),
        y=alt.Y("temp_max:Q"),
        # title="A beautiful sorted-by-value bar chart",
    )


@cache_data
def _time_heatmap_chart():
    altex.hist_chart(
        data=datasets["seattle_weather"],
        x="week(date):T",
        y="day(date):T",
        color=alt.Color(
            "median(temp_max):Q",
            legend=None,
        ),
        # title="A beautiful time hist chart",
    )


@cache_data
def _sparkline_chart():
    altex.line_chart(
        data=datasets["stocks"].query("symbol == 'GOOG'"),
        x="date",
        y="price",
        # title="A beautiful sparkline chart",
        rolling=7,
        height=150,
    )


@cache_data
def _sparkbar_chart():
    altex.bar_chart(
        data=datasets["stocks"].query("symbol == 'GOOG'"),
        x="date",
        y="price",
        # title="A beautiful sparkbar chart",
        height=150,
    )


@cache_data
def _bar_stacked_chart():
    altex.bar_chart(
        data=datasets["barley"],
        x=alt.X("variety", title="Variety"),
        y="sum(yield)",
        color="site",
        # title="A beautiful stacked bar chart",
    )


@cache_data
def _bar_normalized_chart():
    altex.bar_chart(
        data=datasets["barley"],
        x=alt.X("variety:N", title="Variety"),
        y=alt.Y("sum(yield):Q", stack="normalize"),
        color="site:N",
        # title="A beautiful normalized stacked bar chart",
    )


@cache_data
def _bar_grouped_chart():
    altex.bar_chart(
        data=datasets["barley"],
        x="year:O",
        y="sum(yield):Q",
        color="year:N",
        column="site:N",
        # title="A beautiful grouped bar charts",
        width=90,
        use_container_width=False,
    )


class StreamlitChartProvider(BaseProvider):
    def altair_chart(self):
        return self.random_element(
            [
                _bar_grouped_chart,
                _bar_chart_horizontal,
                _bar_chart_log,
                _bar_chart_sorted,
                _bar_stacked_chart,
                _scatter_chart,
                _sparkline_chart,
                _sparkbar_chart,
                _time_heatmap_chart,
                _hist_chart,
                _bar_normalized_chart,
            ]
        )()

    def line_chart(self):
        return self.random_element([_line_chart, _multi_line_chart])()

    def bar_chart(self):
        return self.random_element([_bar_chart])()

    def map(self, **kwargs):
        return st_command_with_default(
            st.map,
            {
                "data": pd.DataFrame(
                    np.random.randn(self.random_int(600, 1000), 2) / [50, 50]
                    + [37.76, -122.4],
                    columns=["lat", "lon"],
                )
            },
            **kwargs,
        )

    def pyplot(self, **kwargs):
        arr = np.random.normal(1, 1, size=100)
        fig, ax = plt.subplots()
        ax.hist(arr, bins=20)
        return st_command_with_default(st.pyplot, {"fig": fig}, **kwargs)

    def vega_lite_chart(self, **kwargs):
        data = pd.DataFrame(np.random.randn(200, 3), columns=["a", "b", "c"])
        return st_command_with_default(
            st.vega_lite_chart,
            {
                "data": data,
                "spec": {
                    "mark": {"type": "circle", "tooltip": True},
                    "encoding": {
                        "x": {"field": "a", "type": "quantitative"},
                        "y": {"field": "b", "type": "quantitative"},
                        "size": {"field": "c", "type": "quantitative"},
                        "color": {"field": "c", "type": "quantitative"},
                    },
                },
            },
            **kwargs,
        )

    def plotly_chart(self):
        raise NotImplementedError

    def bokeh_chart(self):
        raise NotImplementedError

    def pydeck_chart(self):
        raise NotImplementedError

    def graphviz_chart(self):
        raise NotImplementedError
