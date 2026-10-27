from enum import StrEnum

FLOAT32_BYTES_PER_DIM = 4


class PayloadProfile(StrEnum):
    IDS_ONLY = "ids_only"
    VECTOR = "vector"
    SCALAR_LABEL = "scalar_label"
    TEXT = "text"

    def estimated_bytes_per_query(self, *, k: int, dim: int, vector_bytes_per_dim: int = FLOAT32_BYTES_PER_DIM) -> int:
        # Approximate payload size used for cloud leaderboard cost expansion.
        # ID + distance is about 20 bytes per hit; vectors default to float32 width.
        id_distance_bytes = 20
        scalar_label_bytes = 16
        if self == PayloadProfile.IDS_ONLY:
            return k * id_distance_bytes
        if self == PayloadProfile.VECTOR:
            return k * (id_distance_bytes + dim * vector_bytes_per_dim)
        if self == PayloadProfile.SCALAR_LABEL:
            return k * (id_distance_bytes + scalar_label_bytes)
        if self == PayloadProfile.TEXT:
            # Approximate returned document text. FTS metrics still use IDs only.
            return k * (id_distance_bytes + 512)
        msg = f"Unsupported payload profile: {self}"
        raise ValueError(msg)
