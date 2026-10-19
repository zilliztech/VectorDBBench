def build_fts_index_param(bm25_k1: float | None, bm25_b: float | None) -> dict:
    text_mapping = {"type": "text"}
    if bm25_k1 is not None or bm25_b is not None:
        text_mapping["similarity"] = "vdbbench_bm25"
    return {
        "properties": {
            "doc_id": {"type": "keyword"},
            "filter_id": {"type": "long"},
            "text": text_mapping,
        },
    }


def build_bm25_similarity_settings(bm25_k1: float | None, bm25_b: float | None) -> dict:
    if bm25_k1 is None and bm25_b is None:
        return {}

    bm25_settings = {"type": "BM25"}
    if bm25_k1 is not None:
        bm25_settings["k1"] = bm25_k1
    if bm25_b is not None:
        bm25_settings["b"] = bm25_b
    return {"similarity": {"vdbbench_bm25": bm25_settings}}
