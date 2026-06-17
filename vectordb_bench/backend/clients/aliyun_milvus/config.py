from pydantic import field_validator

from ..api import DBCaseConfig, IndexType
from ..milvus.config import DISKANNConfig


class AliyunMilvusDISKANNConfig(DISKANNConfig):
    """Milvus DISKANN plus three opt-in Aliyun search-time params.

    Identical to the upstream Milvus DISKANN for index build/load. The only
    difference is three extra search params that are injected into the per-query
    search params **only when explicitly specified**. ``None`` means "not
    specified" (the param is omitted and the server keeps its own default).

    Note ``0`` is a meaningful value (e.g. ``rerank_topk_multiplier=0`` disables
    rerank reads), so "unset" must be ``None``, not ``0``. The web UI passes a
    negative number / ``"DEFAULT"`` sentinel which is normalized to ``None``.
    """

    rerank_topk_multiplier: int | None = None
    early_termination_threshold: int | None = None
    cross_segment_rerank: bool | None = None

    @field_validator("rerank_topk_multiplier", "early_termination_threshold", mode="before")
    @classmethod
    def _normalize_optional_int(cls, v: object) -> int | None:
        if v is None or v == "" or v == "DEFAULT":
            return None
        iv = int(v)
        # negative value is the UI "unset" sentinel; 0 stays a real value
        return None if iv < 0 else iv

    @field_validator("cross_segment_rerank", mode="before")
    @classmethod
    def _normalize_optional_bool(cls, v: object) -> bool | None:
        if v is None or v == "" or v == "DEFAULT":
            return None
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes", "on")
        return bool(v)

    def search_param(self) -> dict:
        # Reuse the base DISKANN search params (metric_type + search_list) and
        # only add the three knobs that were explicitly specified.
        sp = super().search_param()
        if self.rerank_topk_multiplier is not None:
            sp["params"]["rerank_topk_multiplier"] = self.rerank_topk_multiplier
        if self.early_termination_threshold is not None:
            sp["params"]["early_termination_threshold"] = self.early_termination_threshold
        if self.cross_segment_rerank is not None:
            sp["params"]["cross_segment_rerank"] = self.cross_segment_rerank
        return sp


_aliyun_milvus_case_config: dict[IndexType, type[DBCaseConfig]] = {
    IndexType.DISKANN: AliyunMilvusDISKANNConfig,
}
