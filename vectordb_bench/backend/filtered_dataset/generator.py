"""Generate a filtered-search fixture: vectors, metadata, filter cases and exact ground truth.

Output layout. train, test, scalar_labels and neighbors.parquet load through CustomDataset as they
are; the per-case ground truth needs runner support for these filters (catalog V02), including
rows shorter than k when fewer rows match.

    train.parquet          id (0..N-1), emb
    test.parquet           id, emb
    scalar_labels.parquet  id, labels, int_attr   (row i describes train id i)
    neighbors.parquet      id, neighbors_id       (unfiltered top-k)
    neighbors_<case>.parquet                      (top-k over the rows matching each case)
    manifest.json          spec, per-case actual selectivity and ground-truth statistics
"""

import json
import logging
import math
import pathlib
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .groundtruth import exact_topk, prepare_vectors
from .spec import FORMAT_VERSION, CaseType, Distribution, FilteredDatasetSpec, IntSpec, LabelSpec

log = logging.getLogger(__name__)

LABEL_FIELD = "labels"
INT_FIELD = "int_attr"
ID_FIELD = "id"
VECTOR_FIELD = "emb"
NEIGHBORS_FIELD = "neighbors_id"
MANIFEST_FILE = "manifest.json"


@dataclass(frozen=True)
class FilterCase:
    case_id: str
    type: CaseType
    target_selectivity: float
    expr: str
    condition: dict  # machine-readable form of expr

    @property
    def groundtruth_file(self) -> str:
        return f"neighbors_{self.case_id}.parquet"


def _streams(seed: int) -> dict[str, np.random.Generator]:
    """Independent RNG per component, so changing one part of a spec leaves the others unchanged."""
    names = ("centers", "train", "test", "labels", "int_attr")
    children = np.random.SeedSequence(seed).spawn(len(names))
    return {name: np.random.default_rng(child) for name, child in zip(names, children, strict=True)}


def _gaussian_mixture(rng: np.random.Generator, centers: np.ndarray, n: int, std: float) -> np.ndarray:
    assignment = rng.integers(0, len(centers), size=n)
    noise = rng.standard_normal((n, centers.shape[1]))
    return (centers[assignment] + std * noise).astype(np.float32)


def _probabilities(num_values: int, distribution: Distribution, zipf_s: float) -> np.ndarray:
    if distribution == Distribution.UNIFORM:
        return np.full(num_values, 1.0 / num_values)
    weights = 1.0 / np.arange(1, num_values + 1, dtype=np.float64) ** zipf_s
    return weights / weights.sum()


def _exact_counts(probabilities: np.ndarray, total: int) -> np.ndarray:
    """Largest-remainder rounding: counts sum to `total` and follow `probabilities` as closely as possible."""
    raw = probabilities * total
    counts = np.floor(raw).astype(np.int64)
    remainder = total - counts.sum()
    # Ties in the fractional part go to the lower index, keeping the result deterministic.
    order = np.argsort(-(raw - counts), kind="stable")
    counts[order[:remainder]] += 1
    return counts


def _latent_order(rng: np.random.Generator, vectors: np.ndarray, correlation: float) -> np.ndarray:
    """Row order from a latent score mixing a vector projection with noise.

    Metadata is assigned by walking rows in this order, so the marginal distribution is exact
    while `correlation` controls how much it follows vector geometry: 0 gives independent
    metadata, 1 gives metadata that is a monotone function of the projection.
    """
    direction = rng.standard_normal(vectors.shape[1])
    direction /= np.linalg.norm(direction)
    projection = vectors.astype(np.float64) @ direction
    std = projection.std()
    z = (projection - projection.mean()) / (std if std > 0 else 1.0)
    noise = rng.standard_normal(len(vectors))
    latent = correlation * z + math.sqrt(max(0.0, 1.0 - correlation**2)) * noise
    return np.argsort(latent, kind="stable")


def _label_names(num_values: int) -> list[str]:
    width = len(str(num_values - 1))
    return [f"label_{i:0{width}d}" for i in range(num_values)]


def generate_labels(rng: np.random.Generator, vectors: np.ndarray, spec: LabelSpec) -> tuple[np.ndarray, list[str]]:
    """Label index per row (0 is the most frequent value) and the value names."""
    counts = _exact_counts(_probabilities(spec.num_values, spec.distribution, spec.zipf_s), len(vectors))
    order = _latent_order(rng, vectors, spec.correlation)
    labels = np.empty(len(vectors), dtype=np.int64)
    labels[order] = np.repeat(np.arange(spec.num_values), counts)
    return labels, _label_names(spec.num_values)


def generate_int_attr(rng: np.random.Generator, vectors: np.ndarray, spec: IntSpec) -> np.ndarray:
    domain = np.arange(spec.low, spec.high + 1, dtype=np.int64)
    probabilities = _probabilities(len(domain), spec.distribution, spec.zipf_s)
    values = np.sort(rng.choice(domain, size=len(vectors), p=probabilities))
    order = _latent_order(rng, vectors, spec.correlation)
    out = np.empty(len(vectors), dtype=np.int64)
    out[order] = values
    return out


def build_cases(spec: FilteredDatasetSpec, labels: np.ndarray, label_names: list[str], int_attr: np.ndarray):
    """Turn target selectivities into concrete filters, choosing the closest achievable one."""
    n = len(labels)
    label_counts = np.bincount(labels, minlength=len(label_names))
    sorted_int = np.sort(int_attr)
    cases: list[tuple[FilterCase, np.ndarray]] = []
    for index, case_spec in enumerate(spec.cases):
        target = case_spec.target_selectivity
        case_id = f"{index:02d}_{case_spec.type.value}_{target:g}"
        if case_spec.type == CaseType.LABEL_EQUAL:
            nonempty = np.flatnonzero(label_counts)
            # argmin returns the first minimum, so ties pick the more frequent (lower-index) value.
            best = int(nonempty[np.argmin(np.abs(label_counts[nonempty] / n - target))])
            value = label_names[best]
            mask = labels == best
            case = FilterCase(
                case_id,
                case_spec.type,
                target,
                expr=f'{LABEL_FIELD} == "{value}"',
                condition={"field": LABEL_FIELD, "op": "==", "value": value},
            )
        else:
            start = int(case_spec.start_quantile * n)
            low = int(sorted_int[start])
            end = start + round(target * n)
            if end >= n:
                high = int(sorted_int[-1]) + 1
            else:
                # Ties at the cut can make the natural bound overshoot or undershoot; keep the closer one,
                # but never an empty range (a heavy value at `low` can make "nothing" the closer count).
                candidates = {int(sorted_int[end]), int(sorted_int[end]) + 1}
                counts = {h: int(np.count_nonzero((int_attr >= low) & (int_attr < h))) for h in candidates}
                nonempty = [h for h in candidates if counts[h] > 0]
                high = min(nonempty or candidates, key=lambda h: (abs(counts[h] - target * n), h))
            mask = (int_attr >= low) & (int_attr < high)
            case = FilterCase(
                case_id,
                case_spec.type,
                target,
                expr=f"{INT_FIELD} >= {low} and {INT_FIELD} < {high}",
                condition={"field": INT_FIELD, "op": "range", "low": low, "high_exclusive": high},
            )
        cases.append((case, mask))
    return cases


def _vector_table(vectors: np.ndarray) -> pa.Table:
    n, dim = vectors.shape
    values = pa.array(vectors.reshape(-1), type=pa.float32())
    offsets = pa.array(np.arange(0, (n + 1) * dim, dim, dtype=np.int32))
    emb = pa.ListArray.from_arrays(offsets, values)
    return pa.table({ID_FIELD: pa.array(np.arange(n, dtype=np.int64)), VECTOR_FIELD: emb})


def _neighbors_table(ids: list[np.ndarray]) -> pa.Table:
    neighbors = pa.array([row.tolist() for row in ids], type=pa.list_(pa.int64()))
    return pa.table({ID_FIELD: pa.array(np.arange(len(ids), dtype=np.int64)), NEIGHBORS_FIELD: neighbors})


def generate_dataset(spec: FilteredDatasetSpec, out_dir: str | pathlib.Path, overwrite: bool = False) -> dict:
    """Write the fixture to `out_dir` and return the manifest."""
    out = pathlib.Path(out_dir)
    if out.exists() and any(out.iterdir()) and not overwrite:
        msg = f"{out} is not empty; pass overwrite=True to write into it anyway"
        raise FileExistsError(msg)
    out.mkdir(parents=True, exist_ok=True)

    rngs = _streams(spec.seed)
    centers = rngs["centers"].standard_normal((spec.num_clusters, spec.dim))
    train = _gaussian_mixture(rngs["train"], centers, spec.num_train, spec.cluster_std)
    test = _gaussian_mixture(rngs["test"], centers, spec.num_test, spec.cluster_std)
    labels, label_names = generate_labels(rngs["labels"], train, spec.labels)
    int_attr = generate_int_attr(rngs["int_attr"], train, spec.int_attr)
    cases = build_cases(spec, labels, label_names, int_attr)

    pq.write_table(_vector_table(train), out / "train.parquet")
    pq.write_table(_vector_table(test), out / "test.parquet")
    pq.write_table(
        pa.table(
            {
                ID_FIELD: pa.array(np.arange(spec.num_train, dtype=np.int64)),
                LABEL_FIELD: pa.array([label_names[i] for i in labels], type=pa.string()),
                INT_FIELD: pa.array(int_attr),
            }
        ),
        out / "scalar_labels.parquet",
    )

    train64 = prepare_vectors(train, spec.metric_type)
    test64 = prepare_vectors(test, spec.metric_type)
    unfiltered = exact_topk(test64, train64, np.ones(spec.num_train, dtype=bool), spec.k, spec.metric_type)
    pq.write_table(_neighbors_table(unfiltered.ids), out / "neighbors.parquet")

    case_reports = []
    seen_conditions: dict[str, str] = {}
    for case, mask in cases:
        log.info("ground truth for %s (%s)", case.case_id, case.expr)
        topk = exact_topk(test64, train64, mask, spec.k, spec.metric_type)
        pq.write_table(_neighbors_table(topk.ids), out / case.groundtruth_file)
        matching = int(mask.sum())
        widths = [len(row) for row in topk.ids]
        pass_rates = np.array([float(mask[row].mean()) if len(row) else 0.0 for row in unfiltered.ids])
        selectivity = matching / spec.num_train
        unfiltered_k = min(spec.k, spec.num_train)
        case_reports.append(
            {
                "case_id": case.case_id,
                "type": case.type.value,
                "expr": case.expr,
                "condition": case.condition,
                "groundtruth_file": case.groundtruth_file,
                "target_selectivity": case.target_selectivity,
                "actual_selectivity": selectivity,
                "filter_rate": 1.0 - selectivity,
                "matching_rows": matching,
                "empty": matching == 0,
                "k": spec.k,
                "min_groundtruth_width": min(widths),
                "queries_with_fewer_than_k": sum(w < spec.k for w in widths),
                "queries_with_boundary_ties": topk.boundary_ties,
                # Share of each query's unfiltered top-k that passes the filter, averaged over queries.
                "unfiltered_topk_pass_rate": float(pass_rates.mean()),
                # Queries none of whose unfiltered neighbors pass: the filter cuts against similarity, the
                # hard case for filtered ANN. Compare with the count expected if metadata ignored the vectors;
                # a much larger count means the metadata is correlated with vector position.
                "queries_with_no_unfiltered_match": int(np.count_nonzero(pass_rates == 0)),
                "expected_queries_with_no_unfiltered_match_if_independent": spec.num_test
                * (1.0 - selectivity) ** unfiltered_k,
                # Set when the closest achievable filter is the same as an earlier case's.
                "same_condition_as": seen_conditions.get(json.dumps(case.condition, sort_keys=True)),
            }
        )
        seen_conditions.setdefault(json.dumps(case.condition, sort_keys=True), case.case_id)

    label_counts = np.bincount(labels, minlength=len(label_names))
    manifest = {
        "format_version": FORMAT_VERSION,
        "generator": "vectordb_bench.backend.filtered_dataset",
        # Byte-identical output is only promised for the same library versions and platform.
        "library_versions": {"numpy": np.__version__, "pyarrow": pa.__version__},
        "spec": spec.model_dump(mode="json"),
        "files": {
            "train": "train.parquet",
            "test": "test.parquet",
            "scalar_labels": "scalar_labels.parquet",
            "groundtruth": "neighbors.parquet",
        },
        "fields": {
            "id": ID_FIELD,
            "vector": VECTOR_FIELD,
            "neighbors": NEIGHBORS_FIELD,
            "label": LABEL_FIELD,
            "int": INT_FIELD,
        },
        "metadata": {
            "label_cardinality": int(np.count_nonzero(label_counts)),
            "label_counts": {name: int(c) for name, c in zip(label_names, label_counts, strict=True)},
            "int_distinct_values": int(np.unique(int_attr).size),
            "int_min": int(int_attr.min()),
            "int_max": int(int_attr.max()),
        },
        "unfiltered_queries_with_boundary_ties": unfiltered.boundary_ties,
        "cases": case_reports,
    }
    (out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
