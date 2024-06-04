import vectordb_bench.backend.clients.alloydb.alloy as alloy
import vectordb_bench.backend.clients.alloydb.config as config
from pydantic import BaseModel
from vectordb_bench.backend.clients.api import DBCaseConfig, DBConfig, IndexType, MetricType

user = 'postgres'
password = 'mysecretpassword'
host = 'localhost'
port = '5432'
dbname = 'test_db'

dim = 768

configDict : config.PgVectorConfigDict = {
	'user' : user, 
	'password' : password, 
	'host': host, 
	'port': port, 
	'dbname': dbname
}



configIndex = config.PgVectorHNSWConfig()

my_db = alloy.alloyDB(dim, configDict, configIndex)



my_db._create_table(dim)
my_db._create_index()
my_db.insert_embeddings()


my_db._drop_table()
my_db._drop_index()






