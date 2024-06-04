from __future__ import annotations

import inspect
import textwrap
from typing import Callable, List

from streamlit.components.v1 import html

from .. import extra


@extra
def sandbox(
    code: str | Callable[[], None],
    stlite_version: str | None = None,
    requirements: List[str] | None = None,
    height: int = 700,
    scrolling: bool = False,
) -> None:
    """
    Execute untrusted Streamlit code in a sandboxed environment.

    This function allows you to execute untrusted Streamlit code inside the user's web browser
    by using stlite (https://github.com/whitphx/stlite) instead of the App server. This is useful
    for apps that generate  and execute Streamlit (or Python) code at runtime based on some user
    instructions. Doing this inside the main Streamlit app would be unsafe since the user could
    execute arbitrary code on the server.

    There are a few limitations to this approach:
    * stlite does not support the full set of Streamlit features. See the stlite documentation
      for more details on limitations: https://github.com/whitphx/stlite#limitations
    * Since the code is executed inside the user's browser, it cannot access any files, session state,
      or other functionalities of the server.
    * The available compute resource depend on the user's machine. So, this is not suited for
      heavy computations.

    Args:
      code (str | Callable[[], None]): The code to execute. This can either be a string containing the code or a function.
        If a function is passed, the source code will be extracted automatically. The function
        is required to be fully self-contained and not reference any variables outside of its
        scope.
      stlite_version (str | None, optional): The version of stlite to use.
        If None, the latest version will be used.. Defaults to None.
      requirements (List[str] | None, optional): A list of Python packages
        to install before executing the code. If None, the following
        packages will be installed: pandas, numpy, plotly, altair.
      height (int, optional): The height of the embedded app in pixels. Defaults to 700.
      scrolling (bool, optional): Whether to allow scrolling inside the embedded app.
        Defaults to False.
    """

    stlite_css_url = "https://cdn.jsdelivr.net/gh/LukasMasuch/stlite-demo/static/stlite/stlite/stlite.css"
    stlite_js_url = (
        "https://cdn.jsdelivr.net/gh/LukasMasuch/stlite-demo/static/stlite/stlite.js"
    )

    if stlite_version is not None:
        stlite_css_url = f"https://cdn.jsdelivr.net/npm/@stlite/mountable@{stlite_version}/build/stlite.css"
        stlite_js_url = f"https://cdn.jsdelivr.net/npm/@stlite/mountable@{stlite_version}/build/stlite.js"

    if inspect.isfunction(code):
        function_name = code.__name__
        code = textwrap.dedent(inspect.getsource(code))
        code += f"\n\n{function_name}()"

    if not requirements:
        requirements = [
            "pandas",
            "numpy",
            "plotly",
            "altair",
        ]
    html(
        f"""
<!DOCTYPE html>
<html>
  <head>
    <meta charset="UTF-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta
      name="viewport"
      content="width=device-width, initial-scale=1, shrink-to-fit=no"
    />
    <title>Embedded Streamlit App</title>
    <link
      rel="stylesheet"
      href="{stlite_css_url}"
    />
  </head>
  <body>
    <div id="root"></div>
    <script src="{stlite_js_url}"></script>
    <script>
      if (window.location.search !== "?embed=true{"&embed_options=disable_scrolling" if scrolling is False else ""}") {{
        window.location.search = "?embed=true{"&embed_options=disable_scrolling" if scrolling is False else ""}";
      }}
      stlite.mount(
  {{
    requirements: ["{'","'.join(requirements)}"], // Packages to install
    entrypoint: "streamlit_app.py",
    files: {{
      "streamlit_app.py": `
import streamlit as st

st.markdown('<style>[data-baseweb~="modal"]{{visibility: hidden;}}</style>', unsafe_allow_html=True,)

{code}
`,
    }},
  }},
        document.getElementById("root")
      );
    </script>
  </body>
</html>
        """,
        height=height,
        scrolling=scrolling,
    )


def example():
    def embedded_app():
        import numpy as np
        import pandas as pd
        import plotly.express as px
        import streamlit as st

        @st.cache_data
        def get_data():
            dates = pd.date_range(start="01-01-2020", end="01-01-2023")
            data = np.random.randn(len(dates), 1).cumsum(axis=0)
            return pd.DataFrame(data, index=dates, columns=["Value"])

        data = get_data()

        value = st.slider(
            "Select a range of values",
            int(data.min()),
            int(data.max()),
            (int(data.min()), int(data.max())),
        )
        filtered_data = data[(data["Value"] >= value[0]) & (data["Value"] <= value[1])]
        st.plotly_chart(px.line(filtered_data, y="Value"))

    sandbox(embedded_app)


__title__ = "Stlite Sandbox"
__desc__ = "Execute untrusted Streamlit code in a sandboxed environment."
__icon__ = "📦"
__examples__ = [example]
__author__ = "Lukas Masuch"
__experimental_playground__ = False
