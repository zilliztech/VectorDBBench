def initStyle(st):
    st.markdown(
        """<style>
            /* expander - header */
            .main div[data-testid='stExpander'] p {font-size: 18px; font-weight: 600;}
            /* db icon */
            div[data-testid='stImage'] {margin: auto;}
            /* db column gap */
            div[data-testid='stHorizontalBlock'] {gap: 8px;}
            /* check box */
            .stCheckbox p { color: #000; font-size: 18px; font-weight: 600; }
            /* db selector - db_name should not wrap */
            div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stCheckbox"] div[data-testid="stWidgetLabel"] p { white-space: nowrap; }
        </style>""",
        unsafe_allow_html=True,
    )
