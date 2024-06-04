from __future__ import annotations

from typing import Any, Iterable

import pandas as pd
import streamlit as st

from streamlit_extras import extra


def _transform_arguments(*args, **kwargs) -> tuple[str, Iterable[Any], dict[str, Any]]:
    no_selection_label = kwargs.pop("no_selection_label", "---")

    _args = list(args)

    # Get the options from either the args or kwargs
    try:
        options = _args.pop(1)
    except IndexError:
        options = kwargs["options"]

    # Prepend the no-selection option to the list of options
    if no_selection_label not in options:
        if isinstance(options, pd.Series):
            options = list(pd.concat([pd.Series([no_selection_label]), options]))
        elif isinstance(options, pd.DataFrame):
            # If the options are a DataFrame, the options are just the first column
            options = options.iloc[:, 0]
            options = list(pd.concat([pd.Series([no_selection_label]), options]))
        else:
            options = [no_selection_label] + list(options)
        kwargs["options"] = options

    return no_selection_label, _args, kwargs


@extra
def selectbox(*args, **kwargs):
    """
    A selectbox that returns None unless the user has explicitly selected one of the
    options. All arguments are passed to st.selectbox except for `no_selection_label`, which is
    used to specify the label of the option that represents no selection.

    Args:
        no_selection_label (str): The label to use for the no-selection option. Defaults to "---".
    """
    no_selection_label, _args, _kwargs = _transform_arguments(*args, **kwargs)

    result = st.selectbox(*_args, **_kwargs)
    if result == no_selection_label:
        return None
    return result


def example():
    st.write(
        """
        This is an example of a selectbox that returns None unless the user has
        explicitly selected one of the options.

        The selectbox below has no default value, so it will return None until the
        user selects an option.
        """
    )
    result = selectbox("Select an option", ["A", "B", "C"])
    st.write("Result:", result)

    result = selectbox(
        "Select an option with different label",
        ["A", "B", "C"],
        no_selection_label="<None>",
    )
    st.write("Result:", result)


__title__ = "No-Default Selectbox"
__desc__ = """
Just like st.selectbox, but with no default value -- returns None if nothing is selected.

Meant to be a solution to https://github.com/streamlit/streamlit/issues/949
"""
__icon__ = "🗳️"
__examples__ = [example]
__author__ = "Zachary Blackwood"
__experimental_playground__ = False
