import os
import streamlit.components.v1 as components
import streamlit as st

_RELEASE = True

if not _RELEASE:
    _component_func = components.declare_component(
        "streamlit_toggle",
        url="http://localhost:3001",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component("streamlit_toggle", path=build_dir)

def st_toggle_switch(label=None, key=None, default_value=False, label_after = False, inactive_color = '#D3D3D3', active_color="#11567f", track_color="#29B5E8"):

    if label_after == True:
        label_end = label
        label_start = ''
        justify = 'flex-start'
    else:
        label_start = label
        label_end = ''
        justify = 'flex-end'

    toggle_value = _component_func(key=key, 
                                    default_value=default_value,
                                    label_after=label_after, 
                                    label_start = label_start, 
                                    label_end = label_end, 
                                    justify = justify,
                                    inactive_color=inactive_color,
                                    active_color=active_color,
                                    track_color=track_color,
                                    )
    return toggle_value if toggle_value != None else default_value

if not _RELEASE:
    
    st.header('Streamlit Toggle Switch')
    st.write('---')
    columns = st.columns(3)
    with columns[0]:
        st_toggle_switch(label="Question 1", key='c1',label_after=False)
        st_toggle_switch(label="Question 2", key='c2',label_after=False)
    with columns[1]:
        st_toggle_switch(label="Question 3", key='q2',label_after=True, default_value=True)
        st_toggle_switch(label="Question 4", key='q3',label_after=True,default_value=True)
    with columns[2]:
        range_slider_toggle = st_toggle_switch("Disable Filter", key='q1',label_after=False, default_value=True)
        range_slider = st.slider(label="Filter Range",min_value=0, max_value=100, disabled=range_slider_toggle)