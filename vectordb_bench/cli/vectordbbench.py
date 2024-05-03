from ..backend.clients.pgvector.cli import PgVectorHNSW
from ..backend.clients.redis.cli import Redis
from ..backend.clients.test.cli import Test
from ..backend.clients.weaviate_cloud.cli import Weaviate
from .cli import cli

cli.add_command(PgVectorHNSW)
cli.add_command(Redis)
cli.add_command(Weaviate)
cli.add_command(Test)

if __name__ == "__main__":
    cli()
