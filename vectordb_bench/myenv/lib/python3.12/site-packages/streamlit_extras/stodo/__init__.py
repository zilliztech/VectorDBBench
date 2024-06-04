import streamlit as st

from .. import extra


@extra
def to_do(st_commands, checkbox_id):
    """Create a to_do item

    Args:
        st_commands (_type_): List of (cmd, args) where cmd is a
            streamlit command and args are the arguments of the command
        checkbox_id (str): Use as a key to the checkbox

    Returns:
        None: Prints the to do list
    """
    cols = st.columns((1, 20))
    done = cols[0].checkbox(" ", key=checkbox_id)
    if done:
        for (cmd, *args) in st_commands:
            with cols[1]:
                if cmd == st.write:
                    text = args[0]
                    cols[1].write(
                        "<s style='color: rgba(49, 51, 63, 0.4)'>" f" {text} </s>",
                        unsafe_allow_html=True,
                    )
                else:
                    if cmd in (
                        st.slider,
                        st.button,
                        st.checkbox,
                        st.time_input,
                        st.color_picker,
                        st.selectbox,
                        st.camera_input,
                        st.radio,
                        st.date_input,
                        st.multiselect,
                        st.text_area,
                        st.text_input,
                    ):
                        cmd(*args, disabled=True)
                    else:
                        cmd(*args)

    else:
        for (cmd, *args) in st_commands:
            with cols[1]:
                if cmd == st.write:
                    st.write(*args, unsafe_allow_html=True)
                else:

                    cmd(*args)
    st.write("")
    return done


def example():
    to_do(
        [(st.write, "☕ Take my coffee")],
        "coffee",
    )
    to_do(
        [(st.write, "🥞 Have a nice breakfast")],
        "pancakes",
    )
    to_do(
        [(st.write, ":train: Go to work!")],
        "work",
    )


__title__ = "To-do items"
__desc__ = "Simple Python function to create to-do items in Streamlit!"
__icon__ = "✔️"
__examples__ = [example]
__author__ = "Arnaud Miribel"
__github_repo__ = "arnaudmiribel/stodo"
__streamlit_cloud_url__ = "http://stodoo.streamlitapp.com"
