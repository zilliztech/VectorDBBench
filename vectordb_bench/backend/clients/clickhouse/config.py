from typing import TypedDict
from pydantic import BaseModel, SecretStr
from ..api import DBConfig, DBCaseConfig, MetricType, IndexType

class ClickhouseConfig(DBConfig):
    user_name: SecretStr = "default"
    password: SecretStr
    host: str = "127.0.0.1"
    port: int = 30193
    db_name: str = "default"

    def to_dict(self) -> dict:
        user_str = self.user_name.get_secret_value()
        pwd_str = self.password.get_secret_value()
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.db_name,
            "user": user_str,
            "password": pwd_str
        }
