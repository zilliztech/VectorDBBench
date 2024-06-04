from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, List, Sequence, Union

import streamlit as st
from streamlit.elements.image import AtomicImage
from streamlit.errors import StreamlitAPIException
from typing_extensions import Literal, Required, TypedDict

from streamlit_extras import streaming_write

from .. import extra

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator

SpecType = Union[int, Sequence[Union[int, float]]]


class ChatMessage(TypedDict):
    author: Required[str]
    avatar: Required[str | AtomicImage | None]
    content: Required[List[Any]]


def _active_dg():
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    ctx = get_script_run_ctx()
    if ctx and len(ctx.dg_stack) > 0:
        return ctx.dg_stack[-1]


def _display_message(
    name: str,
    *args: Any,
    avatar: str | AtomicImage | None = None,
) -> List[Any]:
    with st.chat_message(name, avatar=avatar):
        return streaming_write.write(*args)


@extra
def add_message(
    name: Literal["user", "assistant"] | str,
    *args: Any,
    avatar: str | AtomicImage | None = None,
):
    """
    Adds a chat message to the chat container.
    This command can only be used inside the `chat` container. The message
    will be displayed in the UI and added to the chat history so that the same
    message will be automatically displayed on reruns.

    Args:
        name (Literal["user", "assistant"] | str):
            The name of the message author. Can be “user” or “assistant” to
            enable preset styling and avatars.
            Currently, the name is not shown in the UI but is only set as an
            accessibility label. For accessibility reasons, you should not use
            an empty string.
        avatar (str | AtomicImage | None, optional):
            The avatar shown next to the message. Can be anything that is supported by
            the `avatar` parameter of `st.chat_message`. Defaults to None.
        *args (Any):
            The content of the message. This can be any number of elements that are supported by
            `st.write` as well as generator functions to stream content to the UI.
    """
    active_dg = _active_dg()

    if not hasattr(active_dg, "chat_history"):
        raise StreamlitAPIException(
            "The `add_message` command can only be used inside a `chat` container."
        )

    displayed_elements = _display_message(name, *args, avatar=avatar)
    active_dg.chat_history.append(
        ChatMessage(
            author=name,
            avatar=avatar,
            content=displayed_elements,
        )
    )


@extra
def chat(key: str = "chat_messages") -> "DeltaGenerator":
    """
    Insert a stateful chat container into your app.
    This chat container automatically keeps track of the chat history when you use
    the `add_message` command to add messages to the chat.

    Args:
        key (str, optional): The key that is used to keep track of the chat history in session state.
            Defaults to "chat_messages".

    Returns:
        DeltaGenerator: Chat Container
            The chat container that can be used together with `add_message` to automatically
            keep track of the chat history.
    """

    chat_container = st.container()

    if key not in st.session_state:
        st.session_state[key] = []
    chat_history: List[ChatMessage] = st.session_state[key]

    chat_container.chat_history = chat_history  # type: ignore

    with chat_container:
        for message in chat_history:
            _display_message(
                message["author"], *message["content"], avatar=message["avatar"]
            )

    return chat_container


def example():
    with chat(key="my_chat"):
        if prompt := st.chat_input():
            add_message("user", prompt, avatar="🧑‍💻")

            def stream_echo():
                for word in prompt.split():
                    yield word + " "
                    time.sleep(0.15)

            add_message("assistant", "Echo: ", stream_echo, avatar="🦜")


__title__ = "Stateful Chat"
__desc__ = "A chat container that automatically keeps track of the chat history."
__icon__ = "💬"
__examples__ = {
    example: [chat, add_message],
}
__author__ = "Lukas Masuch"
__experimental_playground__ = False
