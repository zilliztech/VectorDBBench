from vectordb_bench.models import DB

# style const
DB_SELECTOR_COLUMNS = 6
DB_CONFIG_SETTING_COLUMNS = 3
CASE_CONFIG_SETTING_COLUMNS = 4
CHECKBOX_INDENT = 30
TASK_LABEL_INPUT_COLUMNS = 2
CHECKBOX_MAX_COLUMNS = 4
DB_CONFIG_INPUT_MAX_COLUMNS = 2
CASE_CONFIG_INPUT_MAX_COLUMNS = 3
DB_CONFIG_INPUT_WIDTH_RADIO = 2
CASE_CONFIG_INPUT_WIDTH_RADIO = 0.98
CASE_INTRO_RATIO = 3
SIDEBAR_CONTROL_COLUMNS = 3
LEGEND_RECT_WIDTH = 24
LEGEND_RECT_HEIGHT = 16
LEGEND_TEXT_FONT_SIZE = 14

PATTERN_SHAPES = ["", "+", "\\", "x", ".", "|", "/", "-"]


def getPatternShape(i):
    return PATTERN_SHAPES[i % len(PATTERN_SHAPES)]


# run_test page auto-refresh config
MAX_AUTO_REFRESH_COUNT = 999999
MAX_AUTO_REFRESH_INTERVAL = 5000  # 5s

PAGE_TITLE = "VectorDB Benchmark"
FAVICON = "https://assets.zilliz.com/favicon_f7f922fe27.png"
HEADER_ICON = "https://assets.zilliz.com/vdb_benchmark_db790b5387.png"

# RedisCloud icon: https://assets.zilliz.com/Redis_Cloud_74b8bfef39.png
# Elasticsearch icon: https://assets.zilliz.com/elasticsearch_beffeadc29.png
# Chroma icon: https://assets.zilliz.com/chroma_ceb3f06ed7.png
DB_TO_ICON = {
    DB.Milvus: "https://assets.zilliz.com/milvus_c30b0d1994.png",
    DB.ZillizCloud: "https://assets.zilliz.com/zilliz_5f4cc9b050.png",
    DB.ElasticCloud: "https://assets.zilliz.com/Elatic_Cloud_dad8d6a3a3.png",
    DB.Pinecone: "https://assets.zilliz.com/pinecone_94d8154979.png",
    DB.QdrantCloud: "https://assets.zilliz.com/qdrant_b691674fcd.png",
    DB.WeaviateCloud: "https://assets.zilliz.com/weaviate_4f6f171ebe.png",
    DB.PgVector: "https://assets.zilliz.com/PG_Vector_d464f2ef5f.png",
    DB.PgVectoRS: "https://assets.zilliz.com/PG_Vector_d464f2ef5f.png",
    DB.Redis: "https://assets.zilliz.com/Redis_Cloud_74b8bfef39.png",
    DB.Chroma: "https://assets.zilliz.com/chroma_ceb3f06ed7.png",
    DB.Alloy: "data:image/png;base64,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"
}

# RedisCloud color: #0D6EFD
# Chroma color: #FFC107
COLOR_MAP = {
    DB.Milvus.value: "#0DCAF0",
    DB.ZillizCloud.value: "#0D6EFD",
    DB.ElasticCloud.value: "#04D6C8",
    DB.Pinecone.value: "#6610F2",
    DB.QdrantCloud.value: "#D91AD9",
    DB.WeaviateCloud.value: "#20C997",
    DB.PgVector.value: "#4C779A",
    DB.Redis.value: "#0D6EFD",
    DB.Alloy.value: "#0D6EFC",
}
