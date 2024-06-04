import streamlit as st
from faker import Faker
from faker.providers import BaseProvider, date_time

from .common import st_command_with_default

fake = Faker()
fake.add_provider(date_time)


class StreamlitMediaProvider(BaseProvider):
    def image(self, **kwargs):
        return st_command_with_default(
            st.image,
            {
                "image": f"http://placekitten.com/{self.random_int(118, 120)}/{self.random_int(118, 120)}"
            },
            **kwargs,
        )

    def video(self, **kwargs):
        raise NotImplementedError

    def audio(self, **kwargs):
        raise NotImplementedError
