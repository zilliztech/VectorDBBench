from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB
from vectordb_bench.cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)

DBTYPE = DB.Cassandra


class CassandraTypeDict(CommonTypedDict):
    """CLI parameters for Cassandra vector database."""
    # Connection parameters for regular Cassandra
    host: Annotated[
        str | None,
        click.option("--host", type=str, help="Cassandra host (for regular Cassandra)", required=False),
    ]
    port: Annotated[
        int,
        click.option("--port", type=int, help="Cassandra port", default=9042),
    ]

    # Connection parameter for Astra DB
    secure_connect_bundle: Annotated[
        str | None,
        click.option(
            "--secure-connect-bundle",
            type=str,
            help="Path to Secure Connect Bundle zip file (for Astra DB)",
            required=False,
        ),
    ]

    # Authentication parameters
    username: Annotated[
        str | None,
        click.option("--username", type=str, help="Cassandra username", required=False),
    ]
    password: Annotated[
        str | None,
        click.option("--password", type=str, help="Cassandra password", required=False),
    ]
    token: Annotated[
        str | None,
        click.option("--token", type=str, help="Astra DB token", required=False),
    ]

    # Keyspace parameter
    keyspace: Annotated[
        str,
        click.option("--keyspace", type=str, help="Cassandra keyspace", default="vdb_bench"),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(CassandraTypeDict)
def Cassandra(**parameters: Unpack[CassandraTypeDict]):
    """Run VectorDB benchmark with Cassandra.

    Supports both regular Cassandra (use --host and --port) and
    DataStax Astra DB (use --secure-connect-bundle).
    """
    from .config import CassandraConfig, CassandraIndexConfig

    run(
        db=DBTYPE,
        db_config=CassandraConfig(
            host=parameters["host"],
            port=parameters["port"],
            secure_connect_bundle=parameters["secure_connect_bundle"],
            username=parameters["username"],
            password=SecretStr(parameters["password"]) if parameters["password"] else None,
            token=SecretStr(parameters["token"]) if parameters["token"] else None,
            keyspace=parameters["keyspace"],
        ),
        db_case_config=CassandraIndexConfig(),
        **parameters,
    )
