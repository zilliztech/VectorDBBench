import tcvectordb
from tcvectordb.model.enum import FieldType, IndexType, MetricType, EmbeddingModel
from tcvectordb.model.index import Index, VectorIndex, FilterIndex, HNSWParams
from tcvectordb.model.enum import FieldType, IndexType, MetricType, ReadConsistency
from tcvectordb.model.document import Document, SearchParams, Filter

url = "http://"
user = "root"
key = ""


client = tcvectordb.VectorDBClient(
    url=url,
    username=user,
    key=key,
    read_consistency=ReadConsistency.EVENTUAL_CONSISTENCY,
    timeout=30,
)

current_dbs = client.list_databases()
for db in current_dbs:
    db.drop_database()

client.create_database("db-test")
db = client.database("db-test")

index = Index(
    FilterIndex(
        name="id", field_type=FieldType.String, index_type=IndexType.PRIMARY_KEY
    ),
    FilterIndex(
        name="author", field_type=FieldType.String, index_type=IndexType.FILTER
    ),
    FilterIndex(
        name="bookName", field_type=FieldType.String, index_type=IndexType.FILTER
    ),
    VectorIndex(
        name="vector",
        dimension=3,
        index_type=IndexType.HNSW,
        metric_type=MetricType.L2,
        params=HNSWParams(m=16, efconstruction=200),
    ),
)

coll = db.create_collection(
    name="book-test",
    shard=1,
    replicas=0,
    description="this is a collection of test embedding",
    index=index,
)


document_list = [
    Document(
        id="0001",
        vector=[0.2123, 0.21, 0.213],
        bookName="西游记",
        author="吴承恩",
        page=21,
        segment="富贵功名，前缘分定，为人切莫欺心。",
    ),
    Document(
        id="0002",
        vector=[0.2123, 0.22, 0.213],
        bookName="西游记",
        author="吴承恩",
        page=22,
        segment="正大光明，忠良善果弥深。些些狂妄天加谴，眼前不遇待时临。",
    ),
    Document(
        id="0003",
        vector=[0.2123, 0.23, 0.213],
        bookName="三国演义",
        author="罗贯中",
        page=23,
        segment="细作探知这个消息，飞报吕布。",
    ),
    Document(
        id="0004",
        vector=[0.2123, 0.24, 0.213],
        bookName="三国演义",
        author="罗贯中",
        page=24,
        segment="布大惊，与陈宫商议。宫曰：“闻刘玄德新领徐州，可往投之。”布从其言，竟投徐州来。有人报知玄德。",
    ),
    Document(
        id="0005",
        vector=[0.2123, 0.25, 0.213],
        bookName="三国演义",
        author="罗贯中",
        page=25,
        segment="玄德曰：“布乃当今英勇之士，可出迎之。”糜竺曰：“吕布乃虎狼之徒，不可收留；收则伤人矣。",
    ),
]
coll.upsert(documents=document_list)


res = coll.search(
    vectors=[[0.3123, 0.43, 0.213], [0.233, 0.12, 0.97]],  # 指定检索向量，最多指定20个
    params=SearchParams(ef=200),  # 若使用HNSW索引，则需要指定参数ef，ef越大，召回率越高，但也会影响检索速度
    retrieve_vector=False,  # 是否需要返回向量字段，False：不返回，True：返回
    limit=2,  # 指定 Top K 的 K 值
    output_fields=["id"]
)
# 输出相似性检索结果，检索结果为二维数组，每一位为一组返回结果，分别对应search时指定的多个向量
# print(res)
print(res[0][0])
