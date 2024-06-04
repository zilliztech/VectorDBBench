import base64
from io import BytesIO
from pathlib import Path
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components

# Tell streamlit that there is a component called camera_input_live,
# and that the code to display that component is in the "frontend" folder
frontend_dir = (Path(__file__).parent / "frontend").absolute()
_component_func = components.declare_component(
    "camera_input_live", path=str(frontend_dir)
)


def camera_input_live(
    debounce: int = 1000,
    height: int = 530,
    width: int = 704,
    key: Optional[str] = None,
    show_controls: bool = True,
    start_label: str = "Start capturing",
    stop_label: str = "Pause capturing",
) -> Optional[BytesIO]:
    """
    Add a descriptive docstring
    """
    b64_data: Optional[str] = _component_func(
        height=height,
        width=width,
        debounce=debounce,
        showControls=show_controls,
        startLabel=start_label,
        stopLabel=stop_label,
        key=key,
    )

    if b64_data is None:
        return None

    raw_data = b64_data.split(",")[1]  # Strip the data: type prefix

    component_value = BytesIO(base64.b64decode(raw_data))

    return component_value


def main():
    st.write("## Example")

    image = camera_input_live(show_controls=True)

    if image is not None:
        st.image(image)


if __name__ == "__main__":
    main()
