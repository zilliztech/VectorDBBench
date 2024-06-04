# -*- coding: utf-8 -*-
import os
from typing import Any, Callable, Dict, List, Optional, Union

import streamlit.components.v1 as components

_RELEASE = True
COMPONENT_NAME = "streamlit_card"

if _RELEASE:  # use the build instead of development if release is true
    root_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(root_dir, "frontend/build")

    _streamlit_card = components.declare_component(
        COMPONENT_NAME, path=build_dir
    )
else:
    _streamlit_card = components.declare_component(
        COMPONENT_NAME, url="http://localhost:3000"
    )


def card(
    title: str,
    text: Union[str, List[str]],
    image: Optional[str] = None,
    url: Optional[str] = None,
    on_click: Callable[[Any], Any] = lambda: None,
    styles: Optional[Dict[str, Any]] = {"card": {}, "text": {}},
    key: Optional[str] = None,
) -> bool:
    """Creates a UI card like component.

    Args:
        title (str): The title of the card.
        text (str): The text of the card.
        image (str, optional): An optional background image. Defaults to None.
        url (str, optional): An optional url to open when the card is clicked. Defaults to None.
        on_click (Callable, optional): An optional function callback that
                                   will fire when clicked.
                                   Can only work if the url is None.
        key (str, optional): An optional key for the component. Defaults to None.
    """
    clicked = _streamlit_card(
        title=title,
        text=text,
        image=image,
        url=url,
        key=key,
        styles=styles,
        default=False,
    )
    if clicked:
        on_click()
    return clicked
