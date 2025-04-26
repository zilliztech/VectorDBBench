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
    DB.AWSOpenSearch: "https://assets.zilliz.com/opensearch_1eee37584e.jpeg",
    DB.TiDB: "https://img2.pingcap.com/forms/3/d/3d7fd5f9767323d6f037795704211ac44b4923d6.png",
    DB.Vespa: "https://vespa.ai/vespa-content/uploads/2025/01/Vespa-symbol-green-rgb.png.webp",
    DB.LanceDB: "https://raw.githubusercontent.com/lancedb/lancedb/main/docs/src/assets/logo.png",
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
    DB.AWSOpenSearch.value: "#0DCAF0",
    DB.TiDB.value: "#0D6EFD",
    DB.Vespa.value: "#61d790",
}
