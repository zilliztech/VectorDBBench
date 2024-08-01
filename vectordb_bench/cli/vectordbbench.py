from ..backend.clients.pgvector.cli import PgVectorHNSW
from ..backend.clients.pgvecto_rs.cli import PgVectoRSHNSW, PgVectoRSIVFFlat
from ..backend.clients.redis.cli import Redis
from ..backend.clients.memorydb.cli import MemoryDB
from ..backend.clients.test.cli import Test
from ..backend.clients.weaviate_cloud.cli import Weaviate
from ..backend.clients.zilliz_cloud.cli import ZillizAutoIndex
from ..backend.clients.milvus.cli import MilvusAutoIndex
from ..backend.clients.aws_opensearch.cli import AWSOpenSearch


from .cli import cli

cli.add_command(PgVectorHNSW)
cli.add_command(PgVectoRSHNSW)
cli.add_command(PgVectoRSIVFFlat)
cli.add_command(Redis)
cli.add_command(MemoryDB)
cli.add_command(Weaviate)
cli.add_command(Test)
cli.add_command(ZillizAutoIndex)
cli.add_command(MilvusAutoIndex)
cli.add_command(AWSOpenSearch)


if __name__ == "__main__":
    cli()
