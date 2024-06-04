import streamlit as st
from camera_input_live import camera_input_live

from .. import extra

camera_input_live = extra(camera_input_live)


def example():
    st.write("# See a new image every second")
    controls = st.checkbox("Show controls")
    image = camera_input_live(show_controls=controls)
    if image is not None:
        st.image(image)


__title__ = "Camera input live"
__desc__ = "A camera input that updates a variable number of seconds"
__icon__ = "📸"
__author__ = "Zachary Blackwood"
__examples__ = [example]
__github_repo__ = "blackary/streamlit-camera-input-live"
__pypi_name__ = "streamlit-camera-input-live"
__package_name__ = "camera_input_live"
__experimental_playground__ = False
__forum_url__ = (
    "https://discuss.streamlit.io/t/new-component-streamlit-camera-live-input/31220"
)
__stlite__ = False
