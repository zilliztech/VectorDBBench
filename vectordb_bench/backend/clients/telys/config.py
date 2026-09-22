from typing import ClassVar

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, MetricType


class TelysConfig(DBConfig):
    """Connection to a running `telys serve` (TCP).

    Start the server first, e.g.:
        telys serve --path ./telys-vdbbench --host 0.0.0.0 --port 9099 \\
            --access-token "$TELYS_ACCESS_TOKEN"
    """

    host: str = "127.0.0.1"
    port: int = 9099
    access_token: SecretStr | None = None

    # access_token is optional (Unix-socket / trusted-network servers need none)
    _extra_empty_skip: ClassVar[frozenset[str]] = frozenset({"access_token"})

    def to_dict(self) -> dict:
        return {
            "host": self.host,
            "port": int(self.port),
            "access_token": self.access_token.get_secret_value() if self.access_token else None,
        }


class TelysIndexConfig(BaseModel, DBCaseConfig):
    """Telys is layout-first — the only knobs are the per-partition IVF build gate and the recall target.
    Partitions with >= `min_rows` rows get a per-partition IVF; smaller partitions stay exact."""

    metric_type: MetricType | None = None
    min_rows: int = 20000
    target_recall: float = 0.98

    def index_param(self) -> dict:
        return {"min_rows": self.min_rows, "target_recall": self.target_recall}

    def search_param(self) -> dict:
        return {}
