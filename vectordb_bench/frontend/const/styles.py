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

DB_TO_ICON = {
    DB.Milvus: "https://assets.zilliz.com/milvus_c30b0d1994.png",
    DB.ZillizCloud: "https://assets.zilliz.com/zilliz_5f4cc9b050.png",
    DB.ElasticCloud: "https://assets.zilliz.com/elasticsearch_beffeadc29.png",
    DB.Pinecone: "https://assets.zilliz.com/pinecone_94d8154979.png",
    DB.QdrantCloud: "https://assets.zilliz.com/qdrant_b691674fcd.png",
    DB.WeaviateCloud: "https://assets.zilliz.com/weaviate_4f6f171ebe.png",
}


COLOR_MAP = {
    DB.Milvus.value: "#0DCAF0",
    DB.ZillizCloud.value: "#0D6EFD",
    DB.ElasticCloud.value: "#fdc613",
    DB.Pinecone.value: "#6610F2",
    DB.QdrantCloud.value: "#D91AD9",
    DB.WeaviateCloud.value: "#20C997",
}
