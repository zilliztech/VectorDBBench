from __future__ import annotations

import logging
import sys
import unittest.mock as mock
from contextlib import contextmanager
from io import StringIO
from typing import Callable, TextIO

import streamlit as st
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx

from streamlit_extras import extra

__all__ = ["redirect", "stdout", "stderr", "logcapture"]


@extra
@contextmanager
def redirect(src: TextIO, dst: Callable, terminator: str = "\n"):
    """Redirect STDOUT and STDERR to streamlit functions."""
    with StringIO() as buffer:

        def new_write(b):
            buffer.write(b + terminator)
            dst(buffer.getvalue())

        # Test if we are actually running in the streamlit script thread before we redirect
        if get_script_run_ctx() is not None:
            old_write = src.write
            try:
                src.write = new_write  # type: ignore
                yield
            finally:
                src.write = old_write  # type: ignore
        else:
            yield


@extra
@contextmanager
def stdout(dst: Callable, terminator: str = "\n"):
    """
    Capture STDOUT and redirect it to a callable `dst`

    Args:
        dst (Callable): A function callable with a single string argument. The entire captured contents will be
            passed to this function every time a new string is written. It is designed to be compatible with
            st.empty().* functions as callbacks.
        terminator (str, optional): If a `terminator` is specified, it is added onto each call to stdout.write/print.
            This defaults to a newline which causes them to display on separate lines within an st.empty.write `dst.
            If using this with st.empty.code as `dst` it is recommended to set `terminator` to empty string. Defaults to "\n".
    """
    with redirect(sys.stdout, dst, terminator):
        yield


@extra
@contextmanager
def stderr(dst: Callable, terminator="\n"):
    """
    Capture STDERR and redirect it to a callable `dst`.

    Args:
        dst (callable[str]): A funciton callable with a single string argument. The entire captured contents will be
            passed to this function every time a new string is written. It is designed to be compatible with
            st.empty().* functions as callbacks.
        terminator (optional, str): If a `terminator` is specified, it is added onto each call to stdout.write/print.
            This defaults to a newline which causes them to display on separate lines within an st.empty.write `dst.
            If using this with st.empty.code as `dst` it is recommended to set `terminator` to empty string.
    """
    with redirect(sys.stderr, dst, terminator):
        yield


class StreamlitLoggingHandler(logging.StreamHandler):
    """Extension of Stream Handler that passes the value of the stream IO buffer to a callback function on every log."""

    def set_callback(self, func: Callable):
        """Set the callback to be used on this record."""
        # pylint: disable=attribute-defined-outside-init
        self.callback = func

    def emit(self, record: logging.LogRecord):
        """Emit a record but also call a function on the full buffer."""
        super().emit(record)
        self.callback(self.stream.getvalue())


@extra
@contextmanager
def logcapture(
    dst: Callable,
    terminator: str = "\n",
    from_logger: logging.Logger | None = None,
    formatter: logging.Formatter | None = None,
):
    """
    Redirect logging to a streamlit function call `dst`.

    Args:
        dst (callable[str]): A function callable with a single string argument. The entire log contents will be
            passed to this function every time a log is written. It is designed to be compatible with st.empty().*
            functions as callbacks.
        terminator (optional, str): If a `terminator` is specified, it is added onto the end of each log.
            This defaults to a newline which causes them to display on separate lines within an st.empty.write `dst.
            If using this with st.empty.code as `dst` it is recommended to set `terminator` to empty string.
        from_logger (optional, logging.Logger or loguru.logger): The logger from which logs will be captured.
            Defaults to `logging.root`.
        formatter (optional, logging.Formatter): If specified, the specified formatter will be added to the logging
            handler to control how logs are displayed.
    """

    if not from_logger:
        from_logger = logging.getLogger()  # root logger

    # Special-case loguru
    using_loguru = (
        "loguru" in sys.modules and sys.modules["loguru"].logger is from_logger
    )

    with StringIO() as buffer:
        new_handler = StreamlitLoggingHandler(buffer)
        new_handler.set_callback(dst)
        new_handler.terminator = terminator
        if formatter:
            new_handler.setFormatter(formatter)
        elif using_loguru:
            pass
        else:
            new_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(levelname)s %(message)s",
                    datefmt="%m/%d/%Y %I:%M:%S %p",
                )
            )
        handler_id = None
        if using_loguru:
            handler_id = from_logger.add(new_handler)  # type: ignore
        else:
            from_logger.addHandler(new_handler)
        try:
            yield
        finally:
            if using_loguru:
                from_logger.remove(handler_id)  # type: ignore
            else:
                from_logger.removeHandler(new_handler)


# EXAMPLES ----------------------------------------------------------------------------------
def example_stdout():
    output = st.empty()
    with stdout(output.code, terminator=""):
        print("This is some captured stdout")
        print("How about that, Isn't it great?")
        if st.button("Click to print more"):
            print("You added another line!")


def example_stderr():
    output = st.empty()
    with stderr(output.code, terminator=""):
        print("This is some captured stderr", file=sys.stderr)
        print(
            "For this example, though, there aren't any problems...yet", file=sys.stderr
        )
        if st.button("Throw an error!"):
            print("ERROR: Task failed successfully", file=sys.stderr)
            print("Psst....stdout isn't captured here")


def example_logcapture():
    logger = logging.getLogger("examplelogger")
    logger.setLevel("DEBUG")
    with logcapture(st.empty().code, from_logger=logger):
        logger.error("Roses are red")
        logger.info("Violets are blue")
        logger.warning("This warning is yellow")
        logger.debug("Your code is broke, too")


# METADATA ------------------------------------------------------------------------------
__title__ = "Capture"
__desc__ = "Capture utility extensions for the standard streamlit library"
__icon__ = "🥅"
__examples__ = {
    example_stdout: [stdout],
    example_stderr: [stderr],
    example_logcapture: [logcapture],
}
__author__ = "Alexander Martin"
__experimental_playground__ = False


# TESTS ---------------------------------------------------------------------------------


# This patch makes the test _think_ it's running in stremalit
@mock.patch("streamlit_extras.capture.get_script_run_ctx", return_value="not none")
def test_st_stdout(_):
    fake_callback = mock.MagicMock()
    with stdout(fake_callback, terminator=""):
        print("Hello")
        fake_callback.assert_called_with("Hello\n")
        print("World")
        fake_callback.assert_called_with("Hello\nWorld\n")


# This patch makes the test _think_ it's running in stremalit
@mock.patch("streamlit_extras.capture.get_script_run_ctx", return_value="not none")
def test_st_stderr(_):
    fake_callback = mock.MagicMock()
    with stderr(fake_callback):
        print("olleH")
        sys.stderr.write("Hello")
        fake_callback.assert_called_with("Hello\n")
        sys.stderr.write("World")
        fake_callback.assert_called_with("Hello\nWorld\n")


def test_non_streamlit_no_patch():
    # When we're not mocking the current thread, these functions shouldn't patch anything.
    fake_callback = mock.MagicMock()
    original_stdout_write = sys.stdout.write
    original_stderr_write = sys.stderr.write
    with stderr(fake_callback):
        assert sys.stderr.write is original_stderr_write
    with stdout(fake_callback):
        assert sys.stdout.write is original_stdout_write


def test_st_logging():
    fake_callback = mock.MagicMock()

    # Test basic config
    with logcapture(fake_callback):
        logging.root.warning("test log")
        assert "WARNING test log\n" in fake_callback.call_args[0][0]

    # Test from_logger
    testlogger = logging.getLogger("test_logger")
    assert not testlogger.handlers
    fake_callback.reset_mock()
    with logcapture(fake_callback, from_logger=testlogger):
        logging.root.warning("don't show this")
        fake_callback.assert_not_called()
        testlogger.warning("but show this")
        assert "WARNING but show this\n" in fake_callback.call_args[0][0]

    # Test terminator
    with logcapture(fake_callback, terminator="foo"):
        logging.root.warning("test log")
        assert "WARNING test logfoo" in fake_callback.call_args[0][0]
        pass

    # Test formatter
    with logcapture(
        fake_callback, formatter=logging.Formatter("%(message)s %(levelname)s")
    ):
        logging.root.warning("test log")
        assert "test log WARNING" in fake_callback.call_args[0][0]

    # Test loguru
    sys.modules["loguru"] = mock_loguru = mock.MagicMock()
    mock_loguru.logger.add.return_value = 54
    with logcapture(fake_callback, from_logger=mock_loguru.logger):
        assert isinstance(mock_loguru.logger.add.call_args[0][0], logging.Handler)
    assert mock_loguru.logger.remove.call_args[0][0] == 54


__tests__ = [test_st_stdout, test_st_stderr, test_st_logging]
