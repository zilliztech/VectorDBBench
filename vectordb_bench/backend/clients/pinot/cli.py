from typing import Annotated, TypedDict, Unpack

import click
from pydantic import SecretStr

from ....cli.cli import (
    CommonTypedDict,
    HNSWFlavor2,
    click_parameter_decorators_from_typed_dict,
    run,
)
from .. import DB


class PinotTypedDict(TypedDict):
    controller_host: Annotated[
        str,
        click.option("--controller-host", type=str, default="localhost", help="Pinot Controller host"),
    ]
    controller_port: Annotated[
        int,
        click.option("--controller-port", type=int, default=9000, help="Pinot Controller port"),
    ]
    broker_host: Annotated[
        str,
        click.option("--broker-host", type=str, default="localhost", help="Pinot Broker host"),
    ]
    broker_port: Annotated[
        int,
        click.option("--broker-port", type=int, default=8099, help="Pinot Broker port"),
    ]
    username: Annotated[
        str,
        click.option("--username", type=str, default=None, help="Pinot username (optional)"),
    ]
    password: Annotated[
        str,
        click.option("--password", type=str, default=None, help="Pinot password (optional)"),
    ]
    ingest_batch_size: Annotated[
        int,
        click.option(
            "--ingest-batch-size",
            type=int,
            default=100_000,
            show_default=True,
            help=(
                "Rows buffered before flushing one Pinot segment (one ingestFromFile call). "
                "Larger values mean fewer segments and better IVF training / query performance. "
                "Reduce if memory is constrained (100K x 768-dim float32 ~= 300 MB)."
            ),
        ),
    ]


def _pinot_db_config(parameters: dict):
    from .config import PinotConfig

    return PinotConfig(
        db_label=parameters["db_label"],
        controller_host=parameters["controller_host"],
        controller_port=parameters["controller_port"],
        broker_host=parameters["broker_host"],
        broker_port=parameters["broker_port"],
        username=parameters.get("username"),
        password=SecretStr(parameters["password"]) if parameters.get("password") else None,
        ingest_batch_size=parameters["ingest_batch_size"],
    )


@click.group()
def Pinot():
    """Apache Pinot vector search benchmarks."""


# ---------------------------------------------------------------------------
# HNSW
# ---------------------------------------------------------------------------


class PinotHNSWTypedDict(CommonTypedDict, PinotTypedDict, HNSWFlavor2): ...


@Pinot.command("hnsw")
@click_parameter_decorators_from_typed_dict(PinotHNSWTypedDict)
def pinot_hnsw(**parameters: Unpack[PinotHNSWTypedDict]):
    from .config import PinotHNSWConfig

    run(
        db=DB.Pinot,
        db_config=_pinot_db_config(parameters),
        db_case_config=PinotHNSWConfig(
            m=parameters["m"],
            ef_construction=parameters["ef_construction"],
            ef=parameters["ef_runtime"],
        ),
        **parameters,
    )


# ---------------------------------------------------------------------------
# IVF_FLAT
# ---------------------------------------------------------------------------


class PinotIVFFlatTypedDict(CommonTypedDict, PinotTypedDict):
    nlist: Annotated[
        int,
        click.option("--nlist", type=int, default=128, help="Number of Voronoi cells (IVF nlist)"),
    ]
    quantizer: Annotated[
        str,
        click.option(
            "--quantizer",
            type=click.Choice(["FLAT", "SQ8", "SQ4"]),
            default="FLAT",
            help="Quantizer type for IVF_FLAT",
        ),
    ]
    nprobe: Annotated[
        int,
        click.option("--nprobe", type=int, default=8, help="Number of cells to probe at query time"),
    ]
    train_sample_size: Annotated[
        int,
        click.option(
            "--train-sample-size",
            type=int,
            default=None,
            help="Training sample size (defaults to max(nlist*50, 1000) if not set)",
        ),
    ]


@Pinot.command("ivf-flat")
@click_parameter_decorators_from_typed_dict(PinotIVFFlatTypedDict)
def pinot_ivf_flat(**parameters: Unpack[PinotIVFFlatTypedDict]):
    from .config import PinotIVFFlatConfig

    run(
        db=DB.Pinot,
        db_config=_pinot_db_config(parameters),
        db_case_config=PinotIVFFlatConfig(
            nlist=parameters["nlist"],
            quantizer=parameters["quantizer"],
            nprobe=parameters["nprobe"],
            train_sample_size=parameters.get("train_sample_size"),
        ),
        **parameters,
    )


# ---------------------------------------------------------------------------
# IVF_PQ
# ---------------------------------------------------------------------------


class PinotIVFPQTypedDict(CommonTypedDict, PinotTypedDict):
    nlist: Annotated[
        int,
        click.option("--nlist", type=int, default=128, help="Number of Voronoi cells (IVF nlist)"),
    ]
    pq_m: Annotated[
        int,
        click.option("--pq-m", type=int, default=8, help="Number of PQ sub-quantizers (must divide dimension)"),
    ]
    pq_nbits: Annotated[
        int,
        click.option(
            "--pq-nbits",
            type=click.Choice(["4", "6", "8"]),
            default="8",
            help="Bits per PQ code (4, 6, or 8)",
        ),
    ]
    train_sample_size: Annotated[
        int,
        click.option("--train-sample-size", type=int, default=6400, help="Training sample size (must be >= nlist)"),
    ]
    nprobe: Annotated[
        int,
        click.option("--nprobe", type=int, default=8, help="Number of cells to probe at query time"),
    ]


@Pinot.command("ivf-pq")
@click_parameter_decorators_from_typed_dict(PinotIVFPQTypedDict)
def pinot_ivf_pq(**parameters: Unpack[PinotIVFPQTypedDict]):
    from .config import PinotIVFPQConfig

    run(
        db=DB.Pinot,
        db_config=_pinot_db_config(parameters),
        db_case_config=PinotIVFPQConfig(
            nlist=parameters["nlist"],
            pq_m=parameters["pq_m"],
            pq_nbits=int(parameters["pq_nbits"]),
            train_sample_size=parameters["train_sample_size"],
            nprobe=parameters["nprobe"],
        ),
        **parameters,
    )
