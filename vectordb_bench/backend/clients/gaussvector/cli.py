from typing import Annotated, Optional, TypedDict, Unpack

import click
import os
from pydantic import SecretStr

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    get_custom_case_config,
    run,
)
from vectordb_bench.backend.clients import DB


class GaussVectorTypedDict(CommonTypedDict):
    user_name: Annotated[
        str, click.option("--user-name", type=str, help="DB username", required=True)
    ]
    password: Annotated[
        str, click.option("--password", type=str, help="DB password"),
    ]
    host: Annotated[
        str, click.option("--host", type=str, help="DB host", required=True)
    ]
    port: Annotated[
        int, click.option("--port", type=int, help="DB port", required=True)
    ]
    db_name: Annotated[
        str, click.option("--db-name", type=str, help="DB name", required=True)
    ]


class GaussVectorDiskANNConfigTypedDict(TypedDict):
    queue_size: Annotated[
        Optional[int], click.option("--queue_size", type=int, help="queue_size, 64~1000", default=100)
    ]
    num_parallels: Annotated[
        Optional[int], click.option("--num_parallels", type=int, help="num_parallels, 1...64", default=16)
    ]
    enable_pq: Annotated[
        Optional[bool], click.option("--enable_pq", type=bool, help="enable_pq, T or F", default=None)
    ]
    subgraph_count: Annotated[
        Optional[int], click.option("--subgraph_count", type=int, help="subgraph_count, 0~32", default=1)
    ]
    enable_vector_copy: Annotated[
        Optional[bool], click.option("--enable_vector_copy", type=bool, help="T or F", default=False)
    ]
    build_with_quantized_vector: Annotated[
        Optional[bool], click.option("--build_with_quantized_vector", type=bool, help="T or F", default=False)
    ]
    graph_degree: Annotated[
        Optional[int], click.option("--graph_degree", type=int, help="graph_degree, 48~256", default=96)
    ]
    pq_nseg: Annotated[
        Optional[int], click.option("--pq_nseg", type=int, help="pq_nseg, 1~65535", default=1)
    ]
    pq_nclus: Annotated[
        Optional[int], click.option("--pq_nclus", type=int, help="pq_nclus, 2~256", default=16)
    ]
    quantization_type: Annotated[
        Optional[str], click.option("--quantization_type", type=str, help="pq/lvq", default="lvq")
    ]
    lvq_nclus: Annotated[
        Optional[int], click.option("--lvq_nclus", type=int, help="lvq_nclus, 2~128", default=128)
    ]
    maintenance_work_mem: Annotated[
        str, click.option("--maintenance_work_mem", type=str, help="maintenance_work_mem, e.g. 8GB", default="8GB")
    ]
    diskann_probe_ncandidates: Annotated[
        Optional[int], click.option("--diskann_probe_ncandidates", type=int, help="1~max", default=128)
    ]
    modify_vector_index_mode: Annotated[
        Optional[str], click.option("--modify_vector_index_mode", type=str, help="memory type, [1,2,3]", default=2)
    ]
    using_clustering_for_parallel: Annotated[
        Optional[str], click.option("--using_clustering_for_parallel", type=str, help="T or F", default=None)
    ]
    version: Annotated[
        str, click.option("--version", type=str, help="DB version, 102.1.0", default="102.1.0")
    ]


class GaussVectorDiskANNTypedDict(GaussVectorTypedDict, GaussVectorDiskANNConfigTypedDict):
    ...


@cli.command()
@click_parameter_decorators_from_typed_dict(GaussVectorDiskANNTypedDict)
def GaussVectorDiskANN(
    **parameters: Unpack[GaussVectorDiskANNTypedDict],
):
    from .config import GaussVectorConfig, GaussVectorDiskANNConfig

    parameters["custom_case"] = get_custom_case_config(parameters)
    quantization_type = parameters.get("quantization_type", "lvq")

    if quantization_type == "pq":
        parameters["lvq_nclus"] = None
    elif quantization_type == "lvq":
        parameters["pq_nseg"] = None
        parameters["pq_nclus"] = None

    run(
        db=DB.GaussVector,
        db_config=GaussVectorConfig(
            user_name=SecretStr(parameters["user_name"]),
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
            db_name=parameters["db_name"],
        ),
        db_case_config=GaussVectorDiskANNConfig(
            queue_size=parameters["queue_size"],
            num_parallels=parameters["num_parallels"],
            enable_pq=parameters["enable_pq"],
            subgraph_count=parameters["subgraph_count"],
            enable_vector_copy=parameters["enable_vector_copy"],
            build_with_quantized_vector=parameters["build_with_quantized_vector"],
            graph_degree=parameters["graph_degree"],
            pq_nseg=parameters["pq_nseg"],
            pq_nclus=parameters["pq_nclus"],
            quantization_type=parameters["quantization_type"],
            lvq_nclus=parameters["lvq_nclus"],
            maintenance_work_mem=parameters["maintenance_work_mem"],
            diskann_probe_ncandidates=parameters["diskann_probe_ncandidates"],
            modify_vector_index_mode=parameters["modify_vector_index_mode"],
            using_clustering_for_parallel=parameters["using_clustering_for_parallel"]
            version=parameters["version"]
        ),
        **parameters,
    )