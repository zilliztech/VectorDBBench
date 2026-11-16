"""Command line entry point.

python -m vectordb_bench.backend.filtered_dataset generate --config spec.yaml --out DIR
python -m vectordb_bench.backend.filtered_dataset verify DIR
"""

import json
import logging
import pathlib
import sys

import click
import yaml

from .generator import generate_dataset
from .spec import FilteredDatasetSpec
from .verify import verify_dataset


@click.group()
def cli():
    """Generate and verify distribution-aware filtered-search fixtures."""


@cli.command()
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    help="YAML or JSON spec; omitted fields use the defaults.",
)
@click.option("--out", required=True, type=click.Path(file_okay=False, path_type=pathlib.Path))
@click.option("--seed", type=int, help="Override the spec's seed.")
@click.option(
    "--overwrite",
    is_flag=True,
    help="Allow writing into a non-empty --out directory; existing files may be overwritten.",
)
def generate(config: pathlib.Path | None, out: pathlib.Path, seed: int | None, overwrite: bool):
    """Write vectors, metadata, filter cases, exact ground truth and manifest.json to --out."""
    raw = (yaml.safe_load(config.read_text()) if config else None) or {}
    if seed is not None:
        raw["seed"] = seed
    spec = FilteredDatasetSpec(**raw)
    manifest = generate_dataset(spec, out, overwrite=overwrite)
    for case in manifest["cases"]:
        click.echo(
            f"{case['case_id']:<28} {case['expr']:<40} target {case['target_selectivity']:<7g} "
            f"actual {case['actual_selectivity']:.4f}  no unfiltered match: "
            f"{case['queries_with_no_unfiltered_match']} queries "
            f"(independent: {case['expected_queries_with_no_unfiltered_match_if_independent']:.1f})"
        )
    click.echo(f"Wrote {out}")


@cli.command()
@click.argument("data_dir", type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path))
@click.option("--sample", type=int, help="Check the distances of only this many random queries.")
def verify(data_dir: pathlib.Path, sample: int | None):
    """Check DATA_DIR's ground truth against an independent exact reference."""
    problems = verify_dataset(data_dir, sample_queries=sample)
    for problem in problems:
        click.echo(problem, err=True)
    if problems:
        sys.exit(1)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    click.echo(f"OK: {len(manifest['cases'])} cases verified in {data_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cli()
