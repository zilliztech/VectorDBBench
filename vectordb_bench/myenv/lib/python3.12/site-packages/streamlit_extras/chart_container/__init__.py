from contextlib import contextmanager
from typing import Generator, Sequence

import pandas as pd
import streamlit as st

try:
    from streamlit import cache_data  # streamlit >= 1.18.0
except ImportError:
    from streamlit import experimental_memo as cache_data  # streamlit >= 0.89

from .. import extra


@cache_data
def _to_csv(data: pd.DataFrame):
    return data.to_csv().encode("utf-8")


@cache_data
def _to_parquet(data: pd.DataFrame):
    return data.to_parquet()


_SUPPORTED_EXPORTS = {
    "CSV": {
        "function": _to_csv,
        "extension": ".csv",
        "mime": "text/csv",
    },
    "Parquet": {
        "function": _to_parquet,
        "extension": ".parquet",
    },
}

_SUPPORTED_EXPORT_KEYS = list(_SUPPORTED_EXPORTS.keys())


@extra  # type: ignore
@contextmanager
def chart_container(
    data: pd.DataFrame,
    tabs: Sequence[str] = (
        "Chart 📈",
        "Dataframe 📄",
        "Export 📁",
    ),
    export_formats: Sequence[str] = _SUPPORTED_EXPORT_KEYS,
) -> Generator:
    """Embed chart in a (chart, data, export, explore) tabs container to let the viewer explore and export its underlying data.

    Args:
        data (pd.DataFrame): Dataframe used in the dataframe tab.
        tabs (Sequence, optional): Tab labels. Defaults to ("Chart 📈", "Dataframe 📄", "Export 📁").
        export_formats (Sequence, optional): Export file formats. Defaults to ("CSV", "Parquet")
    """

    assert all(
        export_format in _SUPPORTED_EXPORTS for export_format in export_formats
    ), f"Input format is not supported, please use one within {_SUPPORTED_EXPORTS.keys()}"

    if "chart_container_widget_key" not in st.session_state:
        st.session_state["chart_container_widget_key"] = 0

    def _get_random_widget_key() -> str:
        st.session_state.chart_container_widget_key += 1
        return st.session_state.chart_container_widget_key

    tab_1, tab_2, tab_3 = st.tabs(tabs)

    with tab_1:
        yield

    with tab_2:
        st.dataframe(data, use_container_width=True)

    with tab_3:
        st.caption("Export limited to 1 million rows.")
        export_data = data.head(1_000_000)
        for chosen_export_format in export_formats:
            export_utils = _SUPPORTED_EXPORTS[chosen_export_format]
            exporter = export_utils["function"]
            extension = export_utils["extension"]
            st.download_button(
                f"Download data as {extension}",
                data=exporter(export_data),
                file_name="data" + extension,
                mime=export_utils.get("mime"),
                key=_get_random_widget_key(),
            )


def _get_random_data():
    return pd.DataFrame(
        {
            "a": {
                "0": -0.0303305049,
                "1": -0.1351647028,
                "2": 1.5870779911,
                "3": -1.3836484569,
                "4": 0.098720535,
                "5": -0.3647604289,
                "6": 0.9750224677,
                "7": 1.9491871558,
                "8": 1.2551013828,
                "9": 0.7843967348,
                "10": -2.1718875405,
                "11": -0.5735476161,
                "12": -0.1762156189,
                "13": -0.7407221386,
                "14": 0.0250880712,
                "15": 0.2507830305,
                "16": -1.6129273914,
                "17": -0.4366293705,
                "18": 0.6685293897,
                "19": -0.6817020803,
            },
            "b": {
                "0": -1.0967545261,
                "1": -1.7796399982,
                "2": -0.3882711116,
                "3": -0.9394858767,
                "4": -1.2070381231,
                "5": 0.1399939192,
                "6": -0.2630170362,
                "7": -1.860974035,
                "8": -0.7970238226,
                "9": -0.2423244595,
                "10": -1.5030435135,
                "11": -0.5431451721,
                "12": 0.7148690473,
                "13": 0.2743864135,
                "14": -0.4985624255,
                "15": 0.8893880137,
                "16": -0.1999725001,
                "17": 0.7686599597,
                "18": -0.309580067,
                "19": 1.4339088721,
            },
            "c": {
                "0": -1.5651083451,
                "1": 0.2913816082,
                "2": 1.1995339283,
                "3": -1.4542600362,
                "4": -0.3883294163,
                "5": 0.7431884696,
                "6": -0.7326326269,
                "7": -0.6609734474,
                "8": 0.8558914313,
                "9": 1.8783811641,
                "10": -1.6879024826,
                "11": -1.3691829023,
                "12": 1.0217629925,
                "13": 1.0185489658,
                "14": 0.2889080313,
                "15": -0.0821493491,
                "16": 0.3449393609,
                "17": 0.2998609069,
                "18": -0.3677563609,
                "19": 0.4388375331,
            },
        }
    )


def example_one():
    chart_data = _get_random_data()
    with chart_container(chart_data):
        st.write("Here's a cool chart")
        st.area_chart(chart_data)


def example_two():
    chart_data = _get_random_data()
    with chart_container(chart_data):
        st.write(
            "I can use a subset of the data for my chart... "
            "but still give all the necessary context in "
            "`chart_container`!"
        )
        st.area_chart(chart_data[["a", "b"]])


__title__ = "Chart container"
__desc__ = "Embed your chart in a nice tabs container to let viewers explore and export its underlying data."
__icon__ = "🖼️"
__examples__ = [
    example_one,
    example_two,
]
__author__ = "Arnaud Miribel"
