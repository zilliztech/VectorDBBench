from enum import StrEnum


class PayloadProfile(StrEnum):
    IDS_ONLY = "ids_only"
    VECTOR = "vector"

    def estimated_bytes_per_query(self, *, k: int, dim: int) -> int:
        # Approximate payload size used for cloud leaderboard cost expansion.
        # ID + distance is about 20 bytes per hit; vector is float32.
        id_distance_bytes = 20
        if self == PayloadProfile.IDS_ONLY:
            return k * id_distance_bytes
        if self == PayloadProfile.VECTOR:
            return k * (id_distance_bytes + dim * 4)
        raise ValueError(f"Unsupported payload profile: {self}")
