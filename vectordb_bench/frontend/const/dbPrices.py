from vectordb_bench.backend.clients import DB


DB_DBLABEL_TO_PRICE = {
    DB.Milvus.value: {},
    DB.ZillizCloud.value: {
        "1cu-perf": 0.159,
        "8cu-perf": 1.272,
        "1cu-cap": 0.159,
        "2cu-cap": 0.318,
    },
    DB.WeaviateCloud.value: {
        # "sandox": 0,
        "standard": 10.10,
        "bus_crit": 32.60,
    },
    DB.ElasticCloud.value: {
        "upTo2.5c8g": 0.4793,
    },
    DB.QdrantCloud.value: {
        "0.5c4g-1node": 0.052,
        "2c8g-1node": 0.166,
        "4c16g-5node": 1.426,
    },
    DB.Pinecone.value: {
        "s1.x1": 0.0973,
        "s1.x2": 0.194,
        "p1.x1": 0.0973,
        "p2.x1": 0.146,
        "p2.x1-8node": 1.168,
        "p1.x1-8node": 0.779,
        "s1.x1-2node": 0.195,
    },
}
