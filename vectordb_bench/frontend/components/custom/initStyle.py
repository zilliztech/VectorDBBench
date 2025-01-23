def initStyle(st):
    st.markdown(
        """<style>
            /* expander - header */
            .main div[data-testid='stExpander'] summary p {font-size: 20px; font-weight: 600;}
            /* 
            button {
                height: auto;
                padding-left: 8px !important;
                padding-right: 6px !important;
            }
            */
        </style>""",
        unsafe_allow_html=True,
    )
