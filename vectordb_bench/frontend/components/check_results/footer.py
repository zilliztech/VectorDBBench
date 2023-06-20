def footer(st):
    text = "* All test results are from community contributors. If there is any ambiguity, feel free to raise an issue or make amendments on our <a href='https://github.com/zilliztech/VectorDBBench'>GitHub page</a>."
    st.markdown(
        f"""
        <div style="margin-top: 16px; color: #aaa; font-size: 14px;">{text}</div
        """,
        unsafe_allow_html=True,
    )
