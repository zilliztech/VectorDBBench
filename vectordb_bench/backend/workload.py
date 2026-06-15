from enum import StrEnum


class WorkloadKind(StrEnum):
    VECTOR = "vector"
    FULL_TEXT = "full_text"
    HYBRID_DENSE_BM25 = "hybrid_dense_bm25"
