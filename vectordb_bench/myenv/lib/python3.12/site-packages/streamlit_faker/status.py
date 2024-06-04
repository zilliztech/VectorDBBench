from multiprocessing import AuthenticationError

import streamlit as st
from faker import Faker
from faker.providers import BaseProvider

from .common import st_command_with_default

fake = Faker()


class StreamlitStatusProvider(BaseProvider):
    @staticmethod
    def error(**kwargs):
        return st_command_with_default(st.error, {"body": fake.sentence()}, **kwargs)

    @staticmethod
    def success(**kwargs):
        return st_command_with_default(st.success, {"body": fake.sentence()}, **kwargs)

    @staticmethod
    def warning(**kwargs):
        return st_command_with_default(st.warning, {"body": fake.sentence()}, **kwargs)

    @staticmethod
    def info(**kwargs):
        return st_command_with_default(st.info, {"body": fake.sentence()}, **kwargs)

    def exception(self, **kwargs):
        return st_command_with_default(
            st.exception,
            {
                "exception": self.random_element(
                    [AttributeError, KeyError, AuthenticationError, ImportError]
                )
            },
            **kwargs
        )

    @staticmethod
    def balloons():
        return st.balloons()

    @staticmethod
    def snow():
        return st.snow()
