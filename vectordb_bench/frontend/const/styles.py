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
    DB.TencentVDB: "data:image/png;base64,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",
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
    DB.TencentVDB.value: "#bbbbbb"
}
