from transwarp_hippo_api.hippo_client import HippoClient, HippoField
from transwarp_hippo_api.hippo_type import HippoType, IndexType, MetricType
import numpy as np

ip = ""
port = ""
username = ""
pwd = ""

dim = 128
n_train = 10000
n_test = 100

# connect
hc = HippoClient([f"{ip}:{port}"], username=username, pwd=pwd)

# create database
database_name = "default"
# db = hc.create_database(database_name)

# create table
table_name = "vdbbench_table"
# table_check = hc.check_table_exists(table_name, database_name=database_name)
# if table_check:
#     hc.delete_table(table_name, database_name=database_name)
#     hc.delete_table_in_trash(table_name, database_name=database_name)
vector_field_name = "vector"
int_field_name = "label"
pk_field_name = "pk"
fields = [
    HippoField(pk_field_name, True, HippoType.INT64),
    HippoField(int_field_name, False, HippoType.INT64),
    HippoField(vector_field_name, False, HippoType.FLOAT_VECTOR,
               type_params={"dimension": dim}),
]
client = hc.create_table(name=table_name, fields=fields,
                         database_name=database_name, number_of_shards=1, number_of_replicas=1)


# get table
client = hc.get_table(table_name, database_name=database_name)


# create index
index_name = "vector_index"
M = 30  # [4,96]
ef_construction = 360  # [8, 512]
ef_search = 100  # [topk, 32768]
client.create_index(field_name=vector_field_name, index_name=index_name,
                    index_type=IndexType.HNSW, metric_type=MetricType.L2,
                    M=M, ef_construction=ef_construction, ef_search=ef_search)


# # load?
# index_loaded = client.load_index(index_name)

# insert
pk_data = np.arange(n_train)
int_data = np.random.randint(0, 100, n_train)
vector_data = np.random.rand(n_train, dim)
batch_size = 100
for offset in range(0, n_train, batch_size):
    start = offset
    end = offset + batch_size
    print(f"insert {start}-{end}")
    data = [
        pk_data[start:end].tolist(), int_data[start:end].tolist(
        ), vector_data[start:end].tolist(),
    ]
    client.insert_rows(data)

# need activate - like milvus load
client.activate_index(index_name, wait_for_completion=True, timeout="25h")

# ann search
query_vectors = np.random.rand(n_test, dim)
output_fields = [pk_field_name, int_field_name]
k = 10
dsl = f"{int_field_name} >= 90"
result = client.query(vector_field_name, query_vectors.tolist(),
                      output_fields, topk=k, dsl=dsl)
print(result[0])

result = client.query(vector_field_name, query_vectors.tolist(),
                      output_fields, topk=100)
print(result[0])

# delete table
hc.delete_table(table_name, database_name=database_name)
hc.delete_table_in_trash(table_name, database_name=database_name)

# # delete database
# hc.delete_database(database_name)
