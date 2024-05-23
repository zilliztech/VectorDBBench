from ..backend.clients.pgvector.cli import PgVectorHNSW
from ..backend.clients.redis.cli import Redis
from ..backend.clients.test.cli import Test
from ..backend.clients.weaviate_cloud.cli import Weaviate
from ..backend.clients.zilliz_cloud.cli import Zilliz
from ..backend.clients.milvus.cli import MilvusAutoIndex



from .cli import cli

cli.add_command(PgVectorHNSW)
cli.add_command(Redis)
cli.add_command(Weaviate)
cli.add_command(Test)
cli.add_command(Zilliz)
cli.add_command(MilvusAutoIndex)


if __name__ == "__main__":
    cli()
