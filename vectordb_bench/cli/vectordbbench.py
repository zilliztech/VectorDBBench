from ..backend.clients.alloydb.cli import AlloyDBScaNN
from ..backend.clients.aws_opensearch.cli import AWSOpenSearch
from ..backend.clients.clickhouse.cli import Clickhouse
from ..backend.clients.lancedb.cli import LanceDB
from ..backend.clients.mariadb.cli import MariaDBHNSW
from ..backend.clients.memorydb.cli import MemoryDB
from ..backend.clients.milvus.cli import MilvusAutoIndex
from ..backend.clients.pgdiskann.cli import PgDiskAnn
from ..backend.clients.pgvecto_rs.cli import PgVectoRSHNSW, PgVectoRSIVFFlat
from ..backend.clients.pgvector.cli import PgVectorHNSW
from ..backend.clients.pgvectorscale.cli import PgVectorScaleDiskAnn
from ..backend.clients.redis.cli import Redis
from ..backend.clients.test.cli import Test
from ..backend.clients.tidb.cli import TiDB
from ..backend.clients.vespa.cli import Vespa
from ..backend.clients.weaviate_cloud.cli import Weaviate
from ..backend.clients.zilliz_cloud.cli import ZillizAutoIndex
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
cli.add_command(PgVectorScaleDiskAnn)
cli.add_command(PgDiskAnn)
cli.add_command(AlloyDBScaNN)
cli.add_command(MariaDBHNSW)
cli.add_command(TiDB)
cli.add_command(Clickhouse)
cli.add_command(Vespa)
cli.add_command(LanceDB)


if __name__ == "__main__":
    cli()
