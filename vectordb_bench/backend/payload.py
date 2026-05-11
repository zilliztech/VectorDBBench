from enum import StrEnum


class PayloadProfile(StrEnum):
    IDS_ONLY = "ids_only"
    VECTOR = "vector"
    SCALAR_LABEL = "scalar_label"

    def estimated_bytes_per_query(self, *, k: int, dim: int) -> int:
        # Approximate payload size used for cloud leaderboard cost expansion.
        # ID + distance is about 20 bytes per hit; vector is float32.
        id_distance_bytes = 20
        scalar_label_bytes = 16
        if self == PayloadProfile.IDS_ONLY:
            return k * id_distance_bytes
        if self == PayloadProfile.VECTOR:
            return k * (id_distance_bytes + dim * 4)
        if self == PayloadProfile.SCALAR_LABEL:
            return k * (id_distance_bytes + scalar_label_bytes)
        msg = f"Unsupported payload profile: {self}"
        raise ValueError(msg)
