"""
CLI command for Rust VectorDB in VectorDBBench

Usage:
    vectordbbench rustvectordb --help
    vectordbbench rustvectordb --dataset-name Cohere --dataset-scale 1M --probes 100
"""

from typing import Annotated, Unpack
import click

# Import VectorDBBench CLI utilities
try:
    from ....cli.cli import (
        CommonTypedDict,
        cli,
        click_parameter_decorators_from_typed_dict,
        run,
        get_custom_case_config,
    )
    from .. import DB
except ImportError as e:
    # Fallback for standalone testing
    print(f"Warning: VectorDBBench CLI utilities not found: {e}")
    print("Run this from within VectorDBBench.")
    import sys
    sys.exit(1)


class RustVectorDBTypedDict(CommonTypedDict):
    """CLI parameters for Rust VectorDB"""
    
    branching_factor: Annotated[
        int,
        click.option(
            "--branching-factor",
            type=int,
            help="Branching factor for hierarchical tree (50-200)",
            default=100,
        ),
    ]
    
    target_leaf_size: Annotated[
        int,
        click.option(
            "--target-leaf-size",
            type=int,
            help="Target vectors per leaf node (30-200)",
            default=100,
        ),
    ]
    
    probes: Annotated[
        int,
        click.option(
            "--probes",
            type=int,
            help="Number of clusters to probe during search (20-1000)",
            default=100,
        ),
    ]
    
    rerank_factor: Annotated[
        int,
        click.option(
            "--rerank-factor",
            type=int,
            help="Rerank factor for two-phase search (3-50)",
            default=10,
        ),
    ]
    
    temp_file_path: Annotated[
        str,
        click.option(
            "--temp-file-path",
            type=str,
            help="Path for temporary mmap storage",
            default="/tmp/vectordb_bench.bin",
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(RustVectorDBTypedDict)
def rustvectordb(**parameters: Unpack[RustVectorDBTypedDict]):
    """Benchmark Rust VectorDB with hierarchical clustering and RaBitQ quantization
    
    Rust VectorDB is a high-performance vector database implementation using:
    - Hierarchical clustering for fast search
    - RaBitQ quantization for memory efficiency
    - Two-phase search (quantized filtering + full precision reranking)
    - Memory-mapped storage for large datasets
    
    Examples:
        # Low latency configuration
        vectordbbench rustvectordb --probes 20 --rerank-factor 5 \\
            --dataset-name Cohere --dataset-scale 1M
        
        # Balanced configuration
        vectordbbench rustvectordb --probes 100 --rerank-factor 10 \\
            --dataset-name Cohere --dataset-scale 1M
        
        # High recall configuration
        vectordbbench rustvectordb --probes 200 --rerank-factor 25 \\
            --dataset-name Cohere --dataset-scale 1M
    """
    from .config import RustVectorDBConfig, RustVectorDBCaseConfig
    
    # Get custom case config if provided
    custom_case = get_custom_case_config(parameters)
    parameters["custom_case"] = custom_case
    
    # Create combined config (inherits from both DBConfig and DBCaseConfig)
    config = RustVectorDBCaseConfig(
        temp_file_path=parameters["temp_file_path"],
        branching_factor=parameters["branching_factor"],
        target_leaf_size=parameters["target_leaf_size"],
        probes=parameters["probes"],
        rerank_factor=parameters["rerank_factor"],
        metric_type=parameters.get("metric_type", "L2"),
    )
    
    run(
        db=DB.RustVectorDB,
        db_config=config,  # Has connection info (from RustVectorDBConfig)
        db_case_config=config,  # Has index/search params (from DBCaseConfig)
        **parameters,
    )
