from typing import Any, Dict

import pandas as pd
import streamlit as st
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

from .. import extra


@extra
def dataframe_explorer(df: pd.DataFrame, case: bool = True) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe
        case (bool, optional): If True, text inputs will be case sensitive. Defaults to True.

    Returns:
        pd.DataFrame: Filtered dataframe
    """

    random_key_base = pd.util.hash_pandas_object(df)

    df = df.copy()

    # Try to convert datetimes into standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect(
            "Filter dataframe on",
            df.columns,
            key=f"{random_key_base}_multiselect",
        )
        filters: Dict[str, Any] = dict()
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                left.write("↳")
                filters[column] = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                    key=f"{random_key_base}_{column}",
                )
                df = df[df[column].isin(filters[column])]
            elif is_numeric_dtype(df[column]):
                left.write("↳")
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                filters[column] = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                    key=f"{random_key_base}_{column}",
                )
                df = df[df[column].between(*filters[column])]
            elif is_datetime64_any_dtype(df[column]):
                left.write("↳")
                filters[column] = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                    key=f"{random_key_base}_{column}",
                )
                if len(filters[column]) == 2:
                    filters[column] = tuple(map(pd.to_datetime, filters[column]))
                    start_date, end_date = filters[column]
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                left.write("↳")
                filters[column] = right.text_input(
                    f"Pattern in {column}",
                    key=f"{random_key_base}_{column}",
                )
                if filters[column]:
                    df = df[df[column].str.contains(filters[column], case=case)]

    return df


def generate_fake_dataframe(size, cols, col_names=None, intervals=None, seed=None):
    from itertools import cycle

    import numpy as np
    import pandas as pd

    categories_dict = {
        "animals": [
            "cow",
            "rabbit",
            "duck",
            "shrimp",
            "pig",
            "goat",
            "crab",
            "deer",
            "bee",
            "sheep",
            "fish",
            "turkey",
            "dove",
            "chicken",
            "horse",
        ],
        "names": [
            "James",
            "Mary",
            "Robert",
            "Patricia",
            "John",
            "Jennifer",
            "Michael",
            "Linda",
            "William",
            "Elizabeth",
            "Ahmed",
            "Barbara",
            "Richard",
            "Susan",
            "Salomon",
            "Juan Luis",
        ],
        "cities": [
            "Stockholm",
            "Denver",
            "Moscow",
            "Marseille",
            "Palermo",
            "Tokyo",
            "Lisbon",
            "Oslo",
            "Nairobi",
            "Río de Janeiro",
            "Berlin",
            "Bogotá",
            "Manila",
            "Madrid",
            "Milwaukee",
        ],
        "colors": [
            "red",
            "orange",
            "yellow",
            "green",
            "blue",
            "indigo",
            "purple",
            "pink",
            "silver",
            "gold",
            "beige",
            "brown",
            "grey",
            "black",
            "white",
        ],
    }
    default_intervals = {
        "i": (0, 10),
        "f": (0, 100),
        "c": ("names", 12),
        "d": ("2020-01-01", "2020-12-31"),
    }
    rng = np.random.default_rng(seed)

    first_c = default_intervals["c"][0]
    categories_names = cycle(
        [first_c] + [c for c in categories_dict.keys() if c != first_c]
    )
    default_intervals["c"] = (categories_names, default_intervals["c"][1])

    if isinstance(col_names, list):
        assert len(col_names) == len(cols), (
            f"The fake DataFrame should have {len(cols)} columns but col_names"
            f" is a list with {len(col_names)} elements"
        )
    elif col_names is None:
        suffix = {"c": "cat", "i": "int", "f": "float", "d": "date"}
        col_names = [f"column_{str(i)}_{suffix.get(col)}" for i, col in enumerate(cols)]

    if isinstance(intervals, list):
        assert len(intervals) == len(cols), (
            f"The fake DataFrame should have {len(cols)} columns but intervals"
            f" is a list with {len(intervals)} elements"
        )
    else:
        if isinstance(intervals, dict):
            assert (
                len(set(intervals.keys()) - set(default_intervals.keys())) == 0
            ), "The intervals parameter has invalid keys"
            default_intervals.update(intervals)
        intervals = [default_intervals[col] for col in cols]
    df = pd.DataFrame()
    for col, col_name, interval in zip(cols, col_names, intervals):
        if interval is None:
            interval = default_intervals[col]
        assert (len(interval) == 2 and isinstance(interval, tuple)) or isinstance(
            interval, list
        ), (
            f"This interval {interval} is neither a tuple of two elements nor"
            " a list of strings."
        )
        if col in ("i", "f", "d"):
            start, end = interval
        if col == "i":
            df[col_name] = rng.integers(start, end, size)
        elif col == "f":
            df[col_name] = rng.uniform(start, end, size)
        elif col == "c":
            if isinstance(interval, list):
                categories = np.array(interval)
            else:
                cat_family, length = interval
                if isinstance(cat_family, cycle):
                    cat_family = next(cat_family)
                assert cat_family in categories_dict.keys(), (
                    f"There are no samples for category '{cat_family}'."
                    " Consider passing a list of samples or use one of the"
                    f" available categories: {categories_dict.keys()}"
                )
                categories = rng.choice(
                    categories_dict[cat_family],
                    length,
                    replace=False,
                    shuffle=True,
                )
            df[col_name] = rng.choice(categories, size, shuffle=True)
        elif col == "d":
            df[col_name] = rng.choice(pd.date_range(start, end), size)
    return df


def example_one():
    dataframe = generate_fake_dataframe(
        size=500, cols="dfc", col_names=("date", "income", "person"), seed=1
    )
    filtered_df = dataframe_explorer(dataframe, case=False)
    st.dataframe(filtered_df, use_container_width=True)


__title__ = "Dataframe explorer UI"
__desc__ = (
    "Let your viewers explore dataframes themselves! Learn more about it on"
    " this [blog"
    " post](https://blog.streamlit.io/auto-generate-a-dataframe-filtering-ui-in-streamlit-with-filter_dataframe/)"
)
__icon__ = "🔭"
__examples__ = [example_one]
__author__ = "Streamlit Data Team!"
__streamlit_cloud_url__ = "https://st-filter-dataframe.streamlitapp.com/"
__github_repo__ = "tylerjrichards/st-filter-dataframe"
