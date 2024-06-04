import contextlib
import textwrap
import traceback

import streamlit as st

from .. import extra

###
# Extension from echo() in streamlit/echo.py
###


@extra
@contextlib.contextmanager
def echo_expander(code_location="above", expander=True, label="Show code"):
    """
    Execute code, and show the code that was executed, but in an expander.

    Args:
        code_location (str, optional): Whether to show the echoed code above or below. Defaults to "above".
        expander (bool, optional): Whether the code block should occur in an expander. Defaults to True.
        label (str, optional): If expander is True, then the label for the expander. Defaults to "Show code".
    """

    from streamlit import empty, source_util

    if code_location == "above":
        placeholder = empty()
    else:
        placeholder = st

    try:
        # Get stack frame *before* running the echoed code. The frame's
        # line number will point to the `st.echo` statement we're running.
        frame = traceback.extract_stack()[-3]
        filename, start_line = frame.filename, frame.lineno

        # Read the file containing the source code of the echoed statement.
        with source_util.open_python_file(filename) as source_file:
            source_lines = source_file.readlines()

        # Use ast to parse the Python file and find the code block to display
        import ast

        ap = ast.parse("".join(source_lines))

        ap_map = {}

        def map_ast(a):
            if not hasattr(a, "body"):
                return
            for b in a.body:
                ap_map[b.lineno] = b
                map_ast(b)

        map_ast(ap)
        lines_to_display = source_lines[
            ap_map[start_line].body[0].lineno - 1 : ap_map[start_line].end_lineno
        ]
        code_string = textwrap.dedent("".join(lines_to_display))

        # Run the echoed code...
        yield

        # And draw the code string to the app!
        if expander:
            placeholder.expander(label).code(code_string, "python")
        else:
            placeholder.code(code_string, "python")

    except FileNotFoundError as err:
        placeholder.warning("Unable to display code. %s" % err)


def example1():
    with echo_expander():
        import streamlit as st

        st.markdown(
            """
            This component is a combination of `st.echo` and `st.expander`.
            The code inside the `with echo_expander()` block will be executed,
            and the code can be shown/hidden behind an expander
            """
        )


def example2():
    with echo_expander(code_location="below", label="Simple Dataframe example"):
        import pandas as pd
        import streamlit as st

        df = pd.DataFrame(
            [[1, 2, 3, 4, 5], [11, 12, 13, 14, 15]],
            columns=("A", "B", "C", "D", "E"),
        )
        st.dataframe(df)


__title__ = "Echo Expander"
__desc__ = "Execute code, and show the code that was executed, but in an expander."
__icon__ = "🆒"
__author__ = "Brian Hess"
__examples__ = [example1, example2]
__experimental_playground__ = False
