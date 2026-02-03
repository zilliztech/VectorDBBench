from enum import Enum


class WorkloadKind(str, Enum):
    VECTOR = "vector"
    FULL_TEXT_BM25 = "full_text_bm25"
