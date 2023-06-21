def hideSidebar(st):
    st.markdown(
        """<style>
            div[data-testid='collapsedControl'] {display: none;} 
            .block-container { max-width: 1000px; }
        </style>""",
        unsafe_allow_html=True,
    )
