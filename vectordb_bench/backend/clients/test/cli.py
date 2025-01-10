from typing import Unpack

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)
from .. import DB
from ..test.config import TestConfig, TestIndexConfig


class TestTypedDict(CommonTypedDict): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(TestTypedDict)
def Test(**parameters: Unpack[TestTypedDict]):
    run(
        db=DB.Test,
        db_config=TestConfig(db_label=parameters["db_label"]),
        db_case_config=TestIndexConfig(),
        **parameters,
    )
