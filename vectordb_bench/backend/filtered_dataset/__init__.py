"""Distribution-aware filtered-search fixtures with exact ground truth (Stanford catalog V01)."""

from .generator import FilterCase, generate_dataset
from .spec import CaseType, Distribution, FilterCaseSpec, FilteredDatasetSpec, IntSpec, LabelSpec
from .verify import verify_dataset

__all__ = [
    "CaseType",
    "Distribution",
    "FilterCase",
    "FilterCaseSpec",
    "FilteredDatasetSpec",
    "IntSpec",
    "LabelSpec",
    "generate_dataset",
    "verify_dataset",
]
