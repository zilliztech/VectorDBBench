from faker import Faker

from .chart import StreamlitChartProvider
from .data_display import StreamlitDataDisplayProvider
from .input import StreamlitInputProvider
from .media import StreamlitMediaProvider
from .status import StreamlitStatusProvider
from .text import StreamlitTextProvider


def get_streamlit_faker(seed: int = None, locale: str = "en-US"):
    fake = Faker(locale=locale)
    fake.add_provider(StreamlitTextProvider)
    fake.add_provider(StreamlitChartProvider)
    fake.add_provider(StreamlitInputProvider)
    fake.add_provider(StreamlitDataDisplayProvider)
    fake.add_provider(StreamlitStatusProvider)
    fake.add_provider(StreamlitMediaProvider)

    _seed = seed if seed else fake.random_int(0, 1_000)
    Faker.seed(_seed)

    return fake


StreamlitFaker = get_streamlit_faker()

text_commands = list(set(dir(StreamlitTextProvider)) - set(dir(StreamlitChartProvider)))
chart_commands = list(
    set(dir(StreamlitChartProvider)) - set(dir(StreamlitTextProvider))
)
data_display_commands = list(
    set(dir(StreamlitDataDisplayProvider)) - set(dir(StreamlitTextProvider))
)
input_commands = list(
    set(dir(StreamlitInputProvider)) - set(dir(StreamlitTextProvider))
)
status_commands = list(
    set(dir(StreamlitStatusProvider)) - set(dir(StreamlitTextProvider))
)
media_commands = list(
    set(dir(StreamlitMediaProvider)) - set(dir(StreamlitTextProvider))
)

all_commands = (
    text_commands
    + chart_commands
    + input_commands
    + status_commands
    + data_display_commands
    + media_commands
)
