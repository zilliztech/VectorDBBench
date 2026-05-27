import streamlit as st
from tornado.iostream import StreamClosedError
from tornado.websocket import WebSocketClosedError, WebSocketProtocol13

from vectordb_bench.frontend.components.check_results.headerIcon import drawHeaderIcon
from vectordb_bench.frontend.components.custom.initStyle import initStyle
from vectordb_bench.frontend.components.welcome.explainPrams import explainPrams
from vectordb_bench.frontend.components.welcome.welcomePrams import welcomePrams
from vectordb_bench.frontend.config.styles import FAVICON, PAGE_TITLE

# Consume expected WS-close errors on streamlit's fire-and-forget writes
# (streamlit#9787, unfixed upstream).
_orig_write_message = WebSocketProtocol13.write_message


def _write_message_with_consumer(self, message, binary=False):
    task = _orig_write_message(self, message, binary=binary)

    def _consume(t):
        exc = t.exception()
        if exc is None or isinstance(exc, (WebSocketClosedError, StreamClosedError)):
            return
        t.get_loop().call_exception_handler({"message": "websocket write failed", "exception": exc, "task": t})

    task.add_done_callback(_consume)
    return task


WebSocketProtocol13.write_message = _write_message_with_consumer


def main():
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=FAVICON,
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # header
    drawHeaderIcon(st)

    # init style
    initStyle(st)

    # page
    welcomePrams(st)

    # description
    explainPrams(st)


if __name__ == "__main__":
    main()
