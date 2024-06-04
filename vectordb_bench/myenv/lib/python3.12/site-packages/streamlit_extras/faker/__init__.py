from streamlit_faker import get_streamlit_faker

from .. import extra

get_streamlit_faker = extra(get_streamlit_faker)


def example():
    fake = get_streamlit_faker(seed=42)
    fake.markdown()
    fake.info(icon="💡", body="You can also pass explicit parameters!")
    fake.selectbox()
    fake.slider()
    fake.metric()
    fake.altair_chart()


__title__ = "Streamlit Faker"
__desc__ = "Fake Streamlit commands at the speed of light! Great for prototyping apps."
__icon__ = "🥷"
__examples__ = [example]
__author__ = "Arnaud Miribel"
__github_repo__ = "arnaudmiribel/streamlit-faker"
__pypi_name__ = "streamlit-faker"
__package_name__ = "streamlit_faker"
__experimental_playground__ = False
__stlite__ = False
