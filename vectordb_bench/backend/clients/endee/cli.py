import logging
import uuid
from typing import Annotated

import click

from vectordb_bench.cli.cli import (
    CommonTypedDict,
    benchmark_runner,
    click_parameter_decorators_from_typed_dict,
    get_custom_case_config,
    parse_task_stages,
)
from vectordb_bench.models import (
    CaseConfig,
    CaseType,
    ConcurrencySearchConfig,
    TaskConfig,
)

from .. import DB
from ..api import EmptyDBCaseConfig
from .config import EndeeConfig

log = logging.getLogger(__name__)


class EndeeTypedDict(CommonTypedDict):
    token: Annotated[str, click.option("--token", type=str, required=True, default=None, help="Endee API token")]
    region: Annotated[str, click.option("--region", type=str, default=None, help="Endee region", show_default=True)]
    base_url: Annotated[
        str,
        click.option(
            "--base-url", type=str, default="http://127.0.0.1:8080/api/v1", help="API server URL", show_default=True
        ),
    ]
    space_type: Annotated[
        str,
        click.option(
            "--space-type",
            type=click.Choice(["cosine", "l2", "ip"]),
            default="cosine",
            help="Distance metric",
            show_default=True,
        ),
    ]
    precision: Annotated[
        str,
        click.option(
            "--precision",
            type=click.Choice(["binary", "int8d", "int16d", "float16", "float32"]),
            default="int8d",
            help="Quant Level",
            show_default=True,
        ),
    ]
    version: Annotated[int, click.option("--version", type=int, default=None, help="Index version", show_default=True)]
    m: Annotated[int, click.option("--m", type=int, default=None, help="HNSW M parameter", show_default=True)]
    ef_con: Annotated[
        int, click.option("--ef-con", type=int, default=None, help="HNSW construction parameter", show_default=True)
    ]
    ef_search: Annotated[
        int, click.option("--ef-search", type=int, default=None, help="HNSW search parameter", show_default=True)
    ]
    index_name: Annotated[
        str,
        click.option(
            "--index-name",
            type=str,
            required=True,
            help="Endee index name (will use a random name if not provided)",
            show_default=True,
        ),
    ]


@click.command()
@click_parameter_decorators_from_typed_dict(EndeeTypedDict)
def Endee(**parameters):
    """
    Run VectorDBBench against Endee VectorDB.
    """
    stages = parse_task_stages(
        parameters["drop_old"],
        parameters["load"],
        parameters["search_serial"],
        parameters["search_concurrent"],
    )

    # Generate a random collection name if not provided
    collection_name = parameters["index_name"]
    if not collection_name:
        collection_name = f"endee_bench_{uuid.uuid4().hex[:8]}"

    # Filter out None values before creating config
    params_for_nd = {k: v for k, v in parameters.items() if v is not None}
    db_config = EndeeConfig(**params_for_nd)

    custom_case_config = get_custom_case_config(parameters)

    db_case_config = EmptyDBCaseConfig()

    task = TaskConfig(
        db=DB.Endee,
        db_config=db_config,
        db_case_config=db_case_config,
        case_config=CaseConfig(
            case_id=CaseType[parameters["case_type"]],
            k=parameters["k"],
            concurrency_search_config=ConcurrencySearchConfig(
                concurrency_duration=parameters["concurrency_duration"],
                num_concurrency=[int(s) for s in parameters["num_concurrency"]],
                concurrency_timeout=parameters["concurrency_timeout"],
            ),
            custom_case=custom_case_config,
        ),
        stages=stages,
    )

    # Use the run method of the benchmark_runner object
    if not parameters["dry_run"]:
        # Generate task label
        run_uuid = uuid.uuid4().hex
        base_name = parameters.get("task_label")
        if not base_name:
            base_name = parameters.get("index_name", "endee")

        final_label = f"{base_name}_{run_uuid}"

        # Run the benchmark
        benchmark_runner.run([task], final_label)

        # Wait for task to complete
        import time
        from concurrent.futures import wait

        from vectordb_bench.interface import global_result_future

        time.sleep(5)
        if global_result_future:
            wait([global_result_future])

        # Ensure CLI doesn't close while background processes are active
        while benchmark_runner.has_running():
            time.sleep(1)
