import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from streamlit_extras import extra

streamlit_image_coordinates = extra(streamlit_image_coordinates)


def example():
    "# Click on the image"
    last_coordinates = streamlit_image_coordinates("https://placekitten.com/200/300")

    st.write(last_coordinates)


__title__ = "Streamlit Image Coordinates"
__desc__ = """
Allows you to add an image to your app, and get the coordinates of where the user last
clicked on the image."""
__icon__ = "🎯"
__examples__ = [example]
__author__ = "Zachary Blackwood"
__experimental_playground__ = False
__github_repo__ = "blackary/streamlit-image-coordinates"
__pypi_name__ = "streamlit-image-coordinates"
__package_name__ = "streamlit_image_coordinates"
__stlite__ = False
