from __future__ import annotations

import inspect
import time
from typing import Any, List

import numpy as np
import pandas as pd
import streamlit as st

from .. import extra

_LOREM_IPSUM = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut
labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco
laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in
voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat
non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
"""


@extra
def write(*args: Any, unsafe_allow_html: bool = False, **kwargs) -> List[Any]:
    """Drop-in replacement for `st.write` with streaming support.

    This function is a drop-in replacement for `st.write` that adds additional capabilities:
    * Supports streaming data via generator functions.
    * Executes callable objects (e.g. functions) and writes the return value.
    """
    if not args:
        return []

    written_content: List[Any] = []
    string_buffer: List[str] = []

    def flush_buffer():
        if string_buffer:
            text_content = " ".join(string_buffer)
            text_container = st.empty()
            text_container.markdown(text_content)
            written_content.append(text_content)
            string_buffer[:] = []

    for arg in args:
        # Order matters!
        if isinstance(arg, str):
            string_buffer.append(arg)
        elif callable(arg) or inspect.isgenerator(arg):
            flush_buffer()
            if inspect.isgeneratorfunction(arg) or inspect.isgenerator(arg):
                # This causes greyed out effect since this element is missing on rerun:
                stream_container = None
                streamed_response = ""

                def flush_stream_response():
                    nonlocal streamed_response
                    nonlocal stream_container
                    if streamed_response and stream_container:
                        stream_container.write(
                            streamed_response,
                            unsafe_allow_html=unsafe_allow_html,
                            **kwargs,
                        )
                        written_content.append(streamed_response)
                        stream_container = None
                        streamed_response = ""

                generator = arg() if inspect.isgeneratorfunction(arg) else arg
                for chunk in generator:
                    if isinstance(chunk, str):
                        first_text = False
                        if not stream_container:
                            stream_container = st.empty()
                            first_text = True
                        streamed_response += chunk
                        # Only add the streaming symbol on the second text chunk
                        stream_container.write(
                            streamed_response + ("" if first_text else " ▌"),
                            unsafe_allow_html=unsafe_allow_html,
                            **kwargs,
                        )
                    elif callable(chunk):
                        flush_stream_response()
                        chunk()
                        written_content.append(chunk)
                    else:
                        flush_stream_response()
                        st.write(chunk, unsafe_allow_html=unsafe_allow_html, **kwargs)
                        written_content.append(chunk)
                flush_stream_response()

            else:
                return_value = arg()
                written_content.append(arg)
                if return_value is not None:
                    flush_buffer()
                    st.write(
                        return_value, unsafe_allow_html=unsafe_allow_html, **kwargs
                    )
        else:
            flush_buffer()
            st.write(arg, unsafe_allow_html=unsafe_allow_html, **kwargs)
            written_content.append(arg)
    flush_buffer()
    return written_content


def example():
    def stream_example():
        for word in _LOREM_IPSUM.split():
            yield word + " "
            time.sleep(0.1)

        # Also supports any other object supported by `st.write`
        yield pd.DataFrame(
            np.random.randn(5, 10),
            columns=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        )

        for word in _LOREM_IPSUM.split():
            yield word + " "
            time.sleep(0.05)

    if st.button("Stream data"):
        write(stream_example)


__title__ = "Streaming Write"
__desc__ = "Drop-in replacement for `st.write` with streaming support."
__icon__ = "🌊"
__examples__ = [example]
__author__ = "Lukas Masuch"
__experimental_playground__ = False
__stlite__ = False
