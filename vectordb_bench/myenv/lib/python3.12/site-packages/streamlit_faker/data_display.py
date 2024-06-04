from itertools import cycle
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
from faker import Faker
from faker.providers import BaseProvider

from .common import st_command_with_default

_DATAFRAME_CATEGORIES_DICT = {
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

fake = Faker()


class StreamlitDataDisplayProvider(BaseProvider):
    def _metric_unit(self) -> str:
        return self.random_element(
            ["s", "ms", "%", "$", "€", "USD", "users", "°C", "°F"]
        )

    def _sign(self) -> str:
        return self.random_element(["+", "-"])

    @staticmethod
    def _pandas_dataframe_builder(
        size: int, cols: List, col_names: List = None, intervals=None, seed=None
    ):

        default_intervals = {
            "i": (0, 10),
            "f": (0, 100),
            "c": ("names", 5),
            "d": ("2020-01-01", "2020-12-31"),
        }

        rng = np.random.default_rng(seed)

        first_c = default_intervals["c"][0]
        categories_names = cycle(
            [first_c] + [c for c in _DATAFRAME_CATEGORIES_DICT.keys() if c != first_c]
        )

        default_intervals["c"] = (
            categories_names,
            default_intervals["c"][1],
        )

        if isinstance(col_names, list):
            assert len(col_names) == len(cols), (
                f"The fake DataFrame should have {len(cols)} columns but col_names"
                f" is a list with {len(col_names)} elements"
            )
        elif col_names is None:
            suffix = {"c": "cat", "i": "int", "f": "float", "d": "date"}
            col_names = [
                f"column_{str(i)}_{suffix.get(col)}" for i, col in enumerate(cols)
            ]

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
                    assert cat_family in _DATAFRAME_CATEGORIES_DICT.keys(), (
                        f"There are no samples for category '{cat_family}'."
                        " Consider passing a list of samples or use one of the"
                        f" available categories: {_DATAFRAME_CATEGORIES_DICT.keys()}"
                    )
                    categories = rng.choice(
                        _DATAFRAME_CATEGORIES_DICT[cat_family],
                        length,
                        replace=False,
                        shuffle=True,
                    )
                df[col_name] = rng.choice(categories, size, shuffle=True)
            elif col == "d":
                df[col_name] = rng.choice(pd.date_range(start, end), size)
        return df

    def metric(self, **kwargs):
        unit = self._metric_unit()
        return st_command_with_default(
            st.metric,
            {
                "label": fake.word().title(),
                "value": f"{fake.random_int()} {unit}",
                "delta": f"{self._sign()} {fake.random_int(1, 100)} {unit}",
            },
            **kwargs,
        )

    def json(self, **kwargs):
        return st_command_with_default(
            st.json,
            {
                "body": fake.simple_profile(),
                "expanded": True,
            },
            **kwargs,
        )

    def _pandas_dataframe(self):
        size = self.random_int(5, 20)
        num_cols = self.random_int(3, 6)
        col_names = [fake.name() for _ in range(num_cols)]
        cols = [self.random_element(list("cifd")) for _ in range(num_cols)]
        return self._pandas_dataframe_builder(
            size=size,
            cols=cols,
            col_names=col_names,
        )

    def dataframe(self, **kwargs):
        return st_command_with_default(
            st.dataframe,
            {"data": self._pandas_dataframe()},
            **kwargs,
        )

    def table(self, **kwargs):
        return st_command_with_default(
            st.table,
            {"data": self._pandas_dataframe()},
            **kwargs,
        )
