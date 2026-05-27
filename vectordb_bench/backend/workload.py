from enum import StrEnum


class WorkloadKind(StrEnum):
    VECTOR = "vector"
    FULL_TEXT_BM25 = "full_text_bm25"
    HYBRID_DENSE_BM25 = "hybrid_dense_bm25"
