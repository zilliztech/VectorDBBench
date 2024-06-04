from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitAPIException

from streamlit_extras import stylable_container

from .. import extra

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator

SpecType = Union[int, Sequence[Union[int, float]]]


class GridDeltaGenerator:
    def __init__(
        self,
        parent_dg: "DeltaGenerator",
        spec: List[SpecType],
        *,
        gap: Optional[str] = "small",
        repeat: bool = True,
    ):
        self._parent_dg = parent_dg
        self._container_queue: List["DeltaGenerator"] = []
        self._number_of_rows = 0
        self._spec = spec
        self._gap = gap
        self._repeat = repeat

    def _get_next_cell_container(self) -> "DeltaGenerator":
        if not self._container_queue:
            if not self._repeat and self._number_of_rows > 0:
                raise StreamlitAPIException("The row is already filled up.")

            # Create a new row using st.columns:
            self._number_of_rows += 1
            spec = self._spec[self._number_of_rows % len(self._spec) - 1]
            self._container_queue.extend(self._parent_dg.columns(spec, gap=self._gap))

        return self._container_queue.pop(0)

    def __getattr__(self, name):
        return getattr(self._get_next_cell_container(), name)

    # TODO: context manager support doesn't work yet
    # def __enter__(self):
    #     return self._get_next_cell_container().__enter__()

    # def __exit__(self, exc_type, exc_val, exc_tb):
    #     return self._get_next_cell_container().__exit__(exc_type, exc_val, exc_tb)


@extra
def grid(
    *spec: SpecType,
    gap: Optional[str] = "small",
    vertical_align: Literal["top", "center", "bottom"] = "top",
):
    """
    Insert a multi-element, grid container into your app.

    This function inserts a container into your app that arranges
    multiple elements in a grid layout as defined by the provided spec.
    Elements can be added to the returned container by calling methods directly
    on the returned object.

    Args:
        *spec (int | Iterable[int]): One or many row specs controlling the number and width of cells in each row.
            Each spec can be one of:
                * An integer specifying the number of cells. In this case, all cells have equal
                width.
                * An iterable of numbers (int or float) specifying the relative width of
                each cell. E.g., ``[0.7, 0.3]`` creates two cells, the first
                one occupying 70% of the available width and the second one 30%.
                Or, ``[1, 2, 3]`` creates three cells where the second one is twice
                as wide as the first one, and the third one is three times that width.
                The function iterates over the provided specs in a round-robin order. Upon filling a row,
                it moves on to the next spec, or the first spec if there are no
                more specs.
        gap (Optional[str], optional): The size of the gap between cells, specified as "small", "medium", or "large".
            This parameter defines the visual space between grid cells. Defaults to "small".
        vertical_align (Literal["top", "center", "bottom"], optional): The vertical alignment of the cells in the row.
            Defaults to "top".
    """

    container = stylable_container.stylable_container(
        key=f"grid_{vertical_align}",
        css_styles=[
            """
div[data-testid="column"] > div[data-testid="stVerticalBlockBorderWrapper"] > div {
height: 100%;
}
""",
            """
div[data-testid="column"] > div {
height: 100%;
}
""",
            f"""
div[data-testid="column"] > div[data-testid="stVerticalBlockBorderWrapper"] > div > div[data-testid="stVerticalBlock"] > div.element-container {{
    {"margin-top: auto;" if vertical_align in ["center", "bottom"] else ""}
    {"margin-bottom: auto;" if vertical_align == "center" else ""}
}}
""",
            f"""
div[data-testid="column"] > div > div[data-testid="stVerticalBlock"] > div.element-container {{
    {"margin-top: auto;" if vertical_align in ["center", "bottom"] else ""}
    {"margin-bottom: auto;" if vertical_align == "center" else ""}
}}
""",
        ],
    )

    return GridDeltaGenerator(
        parent_dg=container, spec=list(spec), gap=gap, repeat=True
    )


def example():
    random_df = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])

    my_grid = grid(2, [2, 4, 1], 1, 4, vertical_align="bottom")

    # Row 1:
    my_grid.dataframe(random_df, use_container_width=True)
    my_grid.line_chart(random_df, use_container_width=True)
    # Row 2:
    my_grid.selectbox("Select Country", ["Germany", "Italy", "Japan", "USA"])
    my_grid.text_input("Your name")
    my_grid.button("Send", use_container_width=True)
    # Row 3:
    my_grid.text_area("Your message", height=40)
    # Row 4:
    my_grid.button("Example 1", use_container_width=True)
    my_grid.button("Example 2", use_container_width=True)
    my_grid.button("Example 3", use_container_width=True)
    my_grid.button("Example 4", use_container_width=True)
    # Row 5 (uses the spec from row 1):
    with my_grid.expander("Show Filters", expanded=True):
        st.slider("Filter by Age", 0, 100, 50)
        st.slider("Filter by Height", 0.0, 2.0, 1.0)
        st.slider("Filter by Weight", 0.0, 100.0, 50.0)
    my_grid.dataframe(random_df, use_container_width=True)


__title__ = "Grid Layout"
__desc__ = "A multi-element container that places elements on a specified grid layout."
__icon__ = "💠"
__examples__ = [example]
__author__ = "Lukas Masuch"
__experimental_playground__ = False
