from typing import Iterable

import pandas as pd
import streamlit as st

try:
    from streamlit import cache_data  # streamlit >= 1.18.0
except ImportError:
    from streamlit import experimental_memo as cache_data  # streamlit >= 0.89

from .. import extra


@cache_data
def get_dataframe() -> pd.DataFrame:
    df = pd.DataFrame(
        [
            [2768571, 130655, 1155027, 34713051, 331002277],
            [1448753, 60632, 790040, 3070447, 212558178],
            [654405, 9536, 422931, 19852167, 145934619],
            [605216, 17848, 359891, 8826585, 1379974505],
            [288477, 9860, 178245, 1699369, 32969875],
        ],
        columns=[
            "Total Cases",
            "Total Deaths",
            "Total Recovered",
            "Total Tests",
            "Population",
        ],
    )

    # Create a list named country to store all the image paths
    country = [
        "https://www.countries-ofthe-world.com/flags-normal/flag-of-United-States-of-America.png",
        "https://www.countries-ofthe-world.com/flags-normal/flag-of-Brazil.png",
        "https://www.countries-ofthe-world.com/flags-normal/flag-of-Russia.png",
        "https://www.countries-ofthe-world.com/flags-normal/flag-of-India.png",
        "https://www.countries-ofthe-world.com/flags-normal/flag-of-Peru.png",
    ]
    # Assigning the new list as a new column of the dataframe
    df["Flag"] = country
    return df


@extra
@cache_data
def table_with_images(df: pd.DataFrame, url_columns: Iterable) -> str:
    """
    Generate the HTML of a table with images rendered in it.

    Args:
        df (pd.DataFrame): Original dataframe
        url_columns (Iterable): Column in df which contains URLs

    Returns:
        table_html (str): HTML of the table with images
    """

    df_ = df.copy()

    @cache_data
    def _path_to_image_html(path):
        return '<img src="' + path + '" width="60" >'

    for column in url_columns:
        df_[column] = df_[column].apply(_path_to_image_html)

    table_html = df_.to_html(escape=False)

    return table_html


df = get_dataframe()


def example(df: pd.DataFrame):
    st.caption("Input dataframe (notice 'Flag' column is full of URLs)")
    st.write(df)
    df_html = table_with_images(df=df, url_columns=("Flag",))
    st.caption("Ouput")
    st.markdown(df_html, unsafe_allow_html=True)


__title__ = "Image in tables"
__desc__ = "Transform URLs into images in your dataframes"
__icon__ = "🚩"
__examples__ = [example]
__inputs__ = dict(df=df)
__author__ = "dataprofessor"
__streamlit_cloud_url__ = (
    "https://dataprofessor-st-demo-image-table-streamlit-app-1x7rnd.streamlitapp.com/"
)
__github_repo__ = "dataprofessor/st-demo-image-table"
__stlite__ = False
