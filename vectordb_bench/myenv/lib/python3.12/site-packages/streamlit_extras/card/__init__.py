from streamlit_card import card

from .. import extra

card = extra(card)


def example():
    card(
        title="Hello World!",
        text="Some description",
        image="http://placekitten.com/300/250",
        url="https://www.google.com",
    )


__title__ = "Card"  # title of your extra!
__desc__ = "Streamlit Component, for a UI card"  # description of your extra!
__icon__ = "💳️"  # give your extra an icon!
__examples__ = [example]  # create some examples to show how cool your extra is!
__author__ = "Gamliel Cohen <gamcoh>"
__pypi_name__ = "streamlit-card"
__package_name__ = "streamlit_card"
__github_repo__ = "gamcoh/st-card"  # Optional repo handle
