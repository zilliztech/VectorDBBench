from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from .. import extra
from ..keyboard_text import key, load_key_css


@extra
def keyboard_to_url(
    key: str | None = None,
    key_code: int | None = None,
    url: str | None = None,
):
    """

    Map a keyboard key to open a new tab with a given URL.

    Args:
        key (str, optional): Key to trigger (example 'k'). Defaults to None.
        key_code (int, optional): If key doesn't work, try hard-coding the key_code instead. Defaults to None.
        url (str, optional): Opens the input URL in new tab. Defaults to None.
    """

    assert not (
        key and key_code
    ), """You can not provide key and key_code.
    Either give key and we'll try to find its associated key_code. Or directly
    provide the key_code."""

    assert (key or key_code) and url, """You must provide key or key_code, and a URL"""

    if key:
        key_code_js_row = f"const keyCode = '{key}'.toUpperCase().charCodeAt(0);"
    elif key_code:
        key_code_js_row = f"const keyCode = {key_code};"
    else:
        raise ValueError("You must provide key or key_code")

    components.html(
        f"""
<script>
const doc = window.parent.document;
buttons = Array.from(doc.querySelectorAll('button[kind=primary]'));
{key_code_js_row}
doc.addEventListener('keydown', function(e) {{
    e = e || window.event;
    var target = e.target || e.srcElement;
    // Only trigger the events if they're not happening in an input/textarea/select/button field
    if ( !/INPUT|TEXTAREA|SELECT|BUTTON/.test(target.nodeName) ) {{
        switch (e.keyCode) {{
            case keyCode:
                window.open('{url}', '_blank').focus();
                break;
        }}
    }}
}});
</script>
""",
        height=0,
        width=0,
    )


def example():
    # Main function
    keyboard_to_url(key="S", url="https://www.github.com/streamlit/streamlit")

    load_key_css()
    st.write(
        f"""Now hit {key("S", False)} on your keyboard...!""",
        unsafe_allow_html=True,
    )


__title__ = "Keyboard to URL"
__desc__ = (
    "Create bindings so that hitting a key on your keyboard opens an URL in a"
    " new tab!"
)
__icon__ = "🎯"
__examples__ = [example]
__author__ = "Arnaud Miribel"
__author__ = "Arnaud Miribel"
