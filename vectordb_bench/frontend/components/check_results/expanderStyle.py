def initMainExpanderStyle(st):
    st.markdown(
        """<style>
            .main div[data-testid='stExpander'] p {font-size: 18px; font-weight: 600;}
            .main div[data-testid='stExpander'] {
                background-color: #F6F8FA;
                border: 1px solid #A9BDD140;
                border-radius: 8px;
            }
        </style>""",
        unsafe_allow_html=True,
    )


def initSidebarExanderStyle(st):
    st.markdown(
        """<style>
            section[data-testid='stSidebar']
                div[data-testid='stExpander']
                    div[data-testid='stVerticalBlock']
                        { gap: 0.2rem; }
            div[data-testid='stExpander']
                { background-color: #ffffff; }
            section[data-testid='stSidebar'] 
                .streamlit-expanderHeader 
                    p { font-size: 16px; font-weight: 600; }
            section[data-testid='stSidebar']
                div[data-testid='stExpander']
                    div[data-testid='stVerticalBlock'] 
                        button {
                            padding: 0 0.5rem;
                            margin-bottom: 8px;
                            float: right;
                        }
        <style>""",
        unsafe_allow_html=True,
    )
