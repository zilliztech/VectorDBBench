"""Check a generated fixture against an independent exact reference.

Filters are re-evaluated from the manifest's conditions and the written metadata (not from the
generator's masks), and distances are recomputed directly (differences rather than the expanded
L2 form used by the generator).
"""

import json
import pathlib

import numpy as np
import pyarrow.parquet as pq

from ..clients.api import MetricType
from .generator import ID_FIELD, MANIFEST_FILE, NEIGHBORS_FIELD, VECTOR_FIELD
from .spec import FORMAT_VERSION

DISTANCE_RTOL = 1e-9
DISTANCE_ATOL = 1e-9


def _vectors(path: pathlib.Path) -> np.ndarray:
    table = pq.read_table(path, columns=[ID_FIELD, VECTOR_FIELD])
    ids = table.column(ID_FIELD).to_numpy()
    if not np.array_equal(ids, np.arange(len(ids))):
        msg = f"{path.name}: ids must be 0..N-1 in row order"
        raise ValueError(msg)
    emb = table.column(VECTOR_FIELD).combine_chunks()
    return emb.flatten().to_numpy().reshape(len(ids), -1).astype(np.float64)


def _neighbors(path: pathlib.Path) -> list[np.ndarray]:
    table = pq.read_table(path)
    return [np.asarray(row, dtype=np.int64) for row in table.column(NEIGHBORS_FIELD).to_pylist()]


def evaluate_condition(condition: dict, metadata: dict[str, np.ndarray]) -> np.ndarray:
    values = metadata[condition["field"]]
    if condition["op"] == "==":
        return values == condition["value"]
    if condition["op"] == "range":
        return (values >= condition["low"]) & (values < condition["high_exclusive"])
    msg = f"unknown filter op {condition['op']!r}"
    raise ValueError(msg)


def _reference_distances(query: np.ndarray, rows: np.ndarray, metric_type: MetricType) -> np.ndarray:
    if metric_type == MetricType.L2:
        diff = rows - query
        return np.einsum("ij,ij->i", diff, diff)
    if metric_type == MetricType.COSINE:
        rows = rows / np.linalg.norm(rows, axis=1, keepdims=True)
        query = query / np.linalg.norm(query)
    return -(rows @ query)


def _check_groundtruth(
    name: str,
    groundtruth: list[np.ndarray],
    mask: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    k: int,
    metric_type: MetricType,
    sample: np.ndarray,
) -> list[str]:
    problems = []
    if len(groundtruth) != len(test):
        return [f"{name}: {len(groundtruth)} ground-truth rows for {len(test)} queries"]
    matching = np.flatnonzero(mask)
    expected_width = min(k, matching.size)
    for qid, row in enumerate(groundtruth):
        if len(row) != expected_width:
            problems.append(f"{name}: query {qid} has {len(row)} neighbors, expected {expected_width}")
        elif len(np.unique(row)) != len(row):
            problems.append(f"{name}: query {qid} repeats a neighbor id")
        elif len(row) and not mask[row].all():
            problems.append(f"{name}: query {qid} returns ids that do not satisfy the filter")
    if problems or matching.size == 0:
        return problems
    for qid in sample:
        dist = _reference_distances(test[qid], train[matching], metric_type)
        order = np.lexsort((matching, dist))[:expected_width]
        reference = dist[order]
        given = _reference_distances(test[qid], train[groundtruth[qid]], metric_type)
        # Compare distances rather than ids: equal-distance neighbors are interchangeable answers.
        if not np.allclose(given, reference, rtol=DISTANCE_RTOL, atol=DISTANCE_ATOL):
            problems.append(f"{name}: query {qid} does not match the exact reference top-{expected_width}")
    return problems


def verify_dataset(data_dir: str | pathlib.Path, sample_queries: int | None = None, seed: int = 0) -> list[str]:
    """Return a list of problems; an empty list means the fixture checks out."""
    root = pathlib.Path(data_dir)
    manifest = json.loads((root / MANIFEST_FILE).read_text())
    if manifest.get("format_version") != FORMAT_VERSION:
        return [f"unsupported format_version {manifest.get('format_version')!r}, expected {FORMAT_VERSION}"]
    spec = manifest["spec"]
    metric_type = MetricType(spec["metric_type"])
    k = spec["k"]
    files = manifest["files"]
    train = _vectors(root / files["train"])
    test = _vectors(root / files["test"])

    labels_table = pq.read_table(root / files["scalar_labels"])
    if not np.array_equal(labels_table.column(ID_FIELD).to_numpy(), np.arange(len(train))):
        return ["scalar_labels.parquet: row i must describe train id i"]
    metadata = {
        name: labels_table.column(name).to_numpy(zero_copy_only=False)
        for name in labels_table.column_names
        if name != ID_FIELD
    }

    rng = np.random.default_rng(seed)
    if sample_queries is None or sample_queries >= len(test):
        sample = np.arange(len(test))
    else:
        sample = np.sort(rng.choice(len(test), size=sample_queries, replace=False))

    problems = _check_groundtruth(
        files["groundtruth"],
        _neighbors(root / files["groundtruth"]),
        np.ones(len(train), dtype=bool),
        train,
        test,
        k,
        metric_type,
        sample,
    )
    for case in manifest["cases"]:
        mask = evaluate_condition(case["condition"], metadata)
        matching = int(mask.sum())
        if matching != case["matching_rows"]:
            problems.append(f"{case['case_id']}: manifest says {case['matching_rows']} matching rows, found {matching}")
        if not np.isclose(case["actual_selectivity"], matching / len(train)):
            problems.append(f"{case['case_id']}: recorded selectivity does not match the data")
        groundtruth = _neighbors(root / case["groundtruth_file"])
        problems += _check_groundtruth(case["case_id"], groundtruth, mask, train, test, k, metric_type, sample)
    return problems
