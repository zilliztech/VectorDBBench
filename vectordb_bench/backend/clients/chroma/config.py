from pydantic import SecretStr

from ..api import DBConfig


class ChromaConfig(DBConfig):
    password: SecretStr
    host: SecretStr
    port: int

    def to_dict(self) -> dict:
        return {
            "host": self.host.get_secret_value(),
            "port": self.port,
            "password": self.password.get_secret_value(),
        }
