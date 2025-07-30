"""Wrapper around the pg_diskann vector database over VectorDB"""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import Connection, Cursor, sql

from ..api import VectorDB
from .config import PgDiskANNConfigDict, PgDiskANNIndexConfig

log = logging.getLogger(__name__)


class PgDiskANN(VectorDB):
    """Use psycopg instructions"""

    conn: psycopg.Connection[Any] | None = None
    coursor: psycopg.Cursor[Any] | None = None

    _filtered_search: sql.Composed
    _unfiltered_search: sql.Composed

    def __init__(
        self,
        dim: int,
        db_config: PgDiskANNConfigDict,
        db_case_config: PgDiskANNIndexConfig,
        collection_name: str = "pg_diskann_collection",
        drop_old: bool = False,
        **kwargs,
    ):
        self.name = "PgDiskANN"
        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = collection_name
        self.dim = dim

        self._index_name = "pgdiskann_index"
        self._primary_field = "id"
        self._vector_field = "embedding"

        self.conn, self.cursor = self._create_connection(**self.db_config)

        log.info(f"{self.name} config values: {self.db_config}\n{self.case_config}")
        if not any(
            (
                self.case_config.create_index_before_load,
                self.case_config.create_index_after_load,
            ),
        ):
            msg = (
                f"{self.name} config must create an index using create_index_before_load or create_index_after_load"
                f"{self.name} config values: {self.db_config}\n{self.case_config}"
            )
            log.error(msg)
            raise RuntimeError(msg)

        if drop_old:
            self._drop_index()
            self._drop_table()
            self._create_table(dim)
            if self.case_config.create_index_before_load:
                self._create_index()

        self.cursor.close()
        self.conn.close()
        self.cursor = None
        self.conn = None

    @staticmethod
    def _create_connection(**kwargs) -> tuple[Connection, Cursor]:
        conn = psycopg.connect(**kwargs)
        cursor = conn.cursor()
        
        # Enable extensions - Azure Database for PostgreSQL has pre-installed extensions
        # We need to enable them, not create them
        try:
            # Check which extensions are already enabled
            cursor.execute("SELECT extname FROM pg_extension WHERE extname IN ('pg_diskann', 'vector', 'citus');")
            enabled_extensions = [row[0] for row in cursor.fetchall()]
            
            # Enable pg_diskann extension if not already enabled
            if 'pg_diskann' not in enabled_extensions:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_diskann CASCADE")
                log.info("PgDiskANN extension enabled successfully")
            
            # Enable vector extension if not already enabled
            if 'vector' not in enabled_extensions:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector CASCADE")
                log.info("Vector extension enabled successfully")
                
            # Enable citus extension if not already enabled (for Azure Citus)
            if 'citus' not in enabled_extensions:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS citus CASCADE")
                log.info("Citus extension enabled successfully")
                
            conn.commit()
            
        except Exception as e:
            log.warning(f"Extension setup warning: {e}")
            # Try to continue anyway, extensions might already be available
            conn.rollback()
        
        register_vector(conn)
        conn.autocommit = False
        cursor = conn.cursor()

        assert conn is not None, "Connection is not initialized"
        assert cursor is not None, "Cursor is not initialized"

        return conn, cursor

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        self.conn, self.cursor = self._create_connection(**self.db_config)

        # index configuration may have commands defined that we should set during each client session
        session_options: dict[str, Any] = self.case_config.session_param()

        if len(session_options) > 0:
            for setting_name, setting_val in session_options.items():
                command = sql.SQL("SET {setting_name} " + "= {setting_val};").format(
                    setting_name=sql.Identifier(setting_name),
                    setting_val=sql.Identifier(str(setting_val)),
                )
                log.debug(command.as_string(self.cursor))
                self.cursor.execute(command)
            self.conn.commit()

        self._filtered_search = sql.Composed(
            [
                sql.SQL(
                    "SELECT id FROM public.{table_name} WHERE id >= %s ORDER BY embedding ",
                ).format(table_name=sql.Identifier(self.table_name)),
                sql.SQL(self.case_config.search_param()["metric_fun_op"]),
                sql.SQL(" %s::vector LIMIT %s::int"),
            ],
        )

        self._unfiltered_search = sql.Composed(
            [
                sql.SQL("SELECT id FROM public.{} ORDER BY embedding ").format(
                    sql.Identifier(self.table_name),
                ),
                sql.SQL(self.case_config.search_param()["metric_fun_op"]),
                sql.SQL(" %s::vector LIMIT %s::int"),
            ],
        )

        try:
            yield
        finally:
            self.cursor.close()
            self.conn.close()
            self.cursor = None
            self.conn = None

    def _drop_table(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        log.info(f"{self.name} client drop table : {self.table_name}")

        self.cursor.execute(
            sql.SQL("DROP TABLE IF EXISTS public.{table_name}").format(
                table_name=sql.Identifier(self.table_name),
            ),
        )
        self.conn.commit()

    def optimize(self, data_size: int | None = None):
        self._post_insert()

    def _post_insert(self):
        log.info(f"{self.name} post insert before optimize")
        if self.case_config.create_index_after_load:
            self._drop_index()
            self._create_index()

    def _drop_index(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        log.info(f"{self.name} client drop index : {self._index_name}")

        drop_index_sql = sql.SQL("DROP INDEX IF EXISTS {index_name}").format(
            index_name=sql.Identifier(self._index_name),
        )
        log.debug(drop_index_sql.as_string(self.cursor))
        self.cursor.execute(drop_index_sql)
        self.conn.commit()

    def _set_parallel_index_build_param(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        index_param = self.case_config.index_param()

        if index_param["maintenance_work_mem"] is not None:
            self.cursor.execute(
                sql.SQL("SET maintenance_work_mem TO {};").format(
                    index_param["maintenance_work_mem"],
                ),
            )
            self.cursor.execute(
                sql.SQL("ALTER USER {} SET maintenance_work_mem TO {};").format(
                    sql.Identifier(self.db_config["user"]),
                    index_param["maintenance_work_mem"],
                ),
            )
            self.conn.commit()

        if index_param["max_parallel_workers"] is not None:
            self.cursor.execute(
                sql.SQL("SET max_parallel_maintenance_workers TO '{}';").format(
                    index_param["max_parallel_workers"],
                ),
            )
            self.cursor.execute(
                sql.SQL("ALTER USER {} SET max_parallel_maintenance_workers TO '{}';").format(
                    sql.Identifier(self.db_config["user"]),
                    index_param["max_parallel_workers"],
                ),
            )
            self.cursor.execute(
                sql.SQL("SET max_parallel_workers TO '{}';").format(
                    index_param["max_parallel_workers"],
                ),
            )
            self.cursor.execute(
                sql.SQL("ALTER USER {} SET max_parallel_workers TO '{}';").format(
                    sql.Identifier(self.db_config["user"]),
                    index_param["max_parallel_workers"],
                ),
            )
            self.cursor.execute(
                sql.SQL("ALTER TABLE {} SET (parallel_workers = {});").format(
                    sql.Identifier(self.table_name),
                    index_param["max_parallel_workers"],
                ),
            )
            self.conn.commit()

        results = self.cursor.execute(sql.SQL("SHOW max_parallel_maintenance_workers;")).fetchall()
        results.extend(self.cursor.execute(sql.SQL("SHOW max_parallel_workers;")).fetchall())
        results.extend(self.cursor.execute(sql.SQL("SHOW maintenance_work_mem;")).fetchall())
        log.info(f"{self.name} parallel index creation parameters: {results}")

    def _create_index(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        log.info(f"{self.name} client create index : {self._index_name}")

        index_param: dict[str, Any] = self.case_config.index_param()
        self._set_parallel_index_build_param()

        options = []
        for option_name, option_val in index_param["options"].items():
            if option_val is not None:
                options.append(
                    sql.SQL("{option_name} = {val}").format(
                        option_name=sql.Identifier(option_name),
                        val=sql.Identifier(str(option_val)),
                    ),
                )

        with_clause = sql.SQL("WITH ({});").format(sql.SQL(", ").join(options)) if any(options) else sql.Composed(())

        index_create_sql = sql.SQL(
            """
            CREATE INDEX IF NOT EXISTS {index_name} ON public.{table_name}
            USING {index_type} (embedding {embedding_metric})
            """,
        ).format(
            index_name=sql.Identifier(self._index_name),
            table_name=sql.Identifier(self.table_name),
            index_type=sql.Identifier(index_param["index_type"].lower()),
            embedding_metric=sql.Identifier(index_param["metric"]),
        )
        index_create_sql_with_with_clause = (index_create_sql + with_clause).join(" ")
        log.debug(index_create_sql_with_with_clause.as_string(self.cursor))
        self.cursor.execute(index_create_sql_with_with_clause)
        self.conn.commit()

    def _create_table(self, dim: int):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        try:
            log.info(f"{self.name} client create table : {self.table_name}")

            # Create the table first
            self.cursor.execute(
                sql.SQL(
                    "CREATE TABLE IF NOT EXISTS public.{table_name} (id BIGINT PRIMARY KEY, embedding vector({dim}));",
                ).format(table_name=sql.Identifier(self.table_name), dim=dim),
            )
            self.conn.commit()

            # Check if Citus extension is available and create distributed table
            self._create_distributed_table()
            
        except Exception as e:
            log.warning(f"Failed to create pgdiskann table: {self.table_name} error: {e}")
            raise e from None

    def _create_distributed_table(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        # Check if Citus distribution is enabled in config
        if not getattr(self.case_config, 'enable_citus_distribution', True):
            log.info(f"{self.name} Citus distribution disabled in config")
            return

        try:
            # Check if Citus extension is enabled (Azure Citus should have it pre-installed)
            self.cursor.execute(
                "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'citus');"
            )
            citus_available = self.cursor.fetchone()[0]
            
            if citus_available:
                log.info(f"{self.name} Citus extension found, creating distributed table")
                
                # Create distributed table with hash distribution on id column
                self.cursor.execute(
                    sql.SQL("SELECT create_distributed_table({table_name}, 'id');").format(
                        table_name=sql.Literal(self.table_name)
                    )
                )
                self.conn.commit()
                
                # Set shard count from config (optimized for 500K rows)
                shard_count = getattr(self.case_config, 'shard_count', 8)
                log.info(f"{self.name} Setting shard count to {shard_count}")
                
                # Get distribution info
                self.cursor.execute(
                    sql.SQL("SELECT * FROM pg_dist_partition WHERE logicalrelid = {table_name}::regclass;").format(
                        table_name=sql.Literal(self.table_name)
                    )
                )
                dist_info = self.cursor.fetchall()
                log.info(f"{self.name} Successfully created distributed table with {len(dist_info)} partition(s)")
                log.info(f"{self.name} Distribution info: {dist_info}")
                
                # Get shard distribution across workers
                try:
                    # Get basic shard information
                    self.cursor.execute(
                        sql.SQL("SELECT count(*) FROM pg_dist_shard WHERE logicalrelid = {table_name}::regclass;").format(
                            table_name=sql.Literal(self.table_name)
                        )
                    )
                    actual_shard_count = self.cursor.fetchone()[0]
                    log.info(f"{self.name} Actual shard count created: {actual_shard_count}")
                    
                    # Get shard placement information
                    self.cursor.execute(
                        sql.SQL("SELECT sp.shardid, sp.shardstate, sp.nodename, sp.nodeport FROM pg_dist_shard_placement sp JOIN pg_dist_shard s ON sp.shardid = s.shardid WHERE s.logicalrelid = {table_name}::regclass ORDER BY sp.shardid;").format(
                            table_name=sql.Literal(self.table_name)
                        )
                    )
                    shard_info = self.cursor.fetchall()
                    log.info(f"{self.name} Shard distribution: {shard_info}")
                    
                    # Get worker node information
                    self.cursor.execute("SELECT nodename, nodeport FROM pg_dist_node WHERE isactive = true;")
                    worker_info = self.cursor.fetchall()
                    log.info(f"{self.name} Active worker nodes: {worker_info}")
                    
                except Exception as e:
                    log.info(f"{self.name} Could not get shard distribution info: {e}")
                
            else:
                log.warning(f"{self.name} Citus extension not found, creating regular table")
                
        except Exception as e:
            log.warning(f"Failed to create distributed table: {e}")
            log.info(f"{self.name} Falling back to regular table creation")

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> tuple[int, Exception | None]:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        try:
            metadata_arr = np.array(metadata)
            embeddings_arr = np.array(embeddings)

            with self.cursor.copy(
                sql.SQL("COPY public.{table_name} FROM STDIN (FORMAT BINARY)").format(
                    table_name=sql.Identifier(self.table_name),
                ),
            ) as copy:
                copy.set_types(["bigint", "vector"])
                for i, row in enumerate(metadata_arr):
                    copy.write_row((row, embeddings_arr[i]))
            self.conn.commit()

            if kwargs.get("last_batch"):
                self._post_insert()

            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert data into table ({self.table_name}), error: {e}")
            return 0, e

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
    ) -> list[int]:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        q = np.asarray(query)
        if filters:
            gt = filters.get("id")
            result = self.cursor.execute(
                self._filtered_search,
                (gt, q, k),
                prepare=True,
                binary=True,
            )
        else:
            result = self.cursor.execute(self._unfiltered_search, (q, k), prepare=True, binary=True)

        return [int(i[0]) for i in result.fetchall()]

    def collect_post_benchmark_config(self) -> dict:
        """
        Collect comprehensive database configuration metrics after benchmark completion.
        This runs while data is still loaded to capture actual runtime state.
                
        Returns:
            dict: Comprehensive configuration metrics collected from the live database
        """
        try:
            # Re-establish connection for post-benchmark analysis if needed
            with psycopg.connect(**self.db_config) as conn:
                with conn.cursor() as cursor:
                    
                    log.info("🔍 POST_BENCHMARK_ANALYSIS_START")
                    
                    config_metrics = {
                        'collection_timestamp': datetime.now().isoformat(),
                        'data_still_loaded': True,
                        'analysis_phase': 'post_benchmark'
                    }
                    
                    
                    # 1. Current GUC parameters (live state) - ACTIVE
                    config_metrics['current_guc_parameters'] = self._collect_live_guc_parameters(cursor)
                    
                    # 2. Current Citus state (if enabled) - ACTIVE
                    #if self.case_config.enable_citus_distribution:
                    #    config_metrics['citus_runtime_state'] = self._collect_citus_runtime_state(cursor)
                    
                    # COMMENTED OUT - Not needed for current use case
                    # config_metrics['table_statistics'] = self._collect_current_table_stats(cursor)
                    # config_metrics['session_state'] = self._collect_session_state(cursor)
                    # config_metrics['index_statistics'] = self._collect_index_statistics(cursor)
                    # config_metrics['memory_usage'] = self._collect_memory_usage(cursor)
                    # config_metrics['query_performance_settings'] = self._collect_query_performance_settings(cursor)
                    
                    log.info("🔍 POST_BENCHMARK_ANALYSIS_END")
                    
                    return config_metrics
                    
        except Exception as e:
            log.error(f"Error collecting post-benchmark config: {e}")
            return {'error': str(e)}
    
    
    def _collect_live_guc_parameters(self, cursor) -> dict:
        """Collect current GUC parameter values in live system."""
        try:
            guc_params = {}
            
            # Define all the Citus parameters to collect
            citus_parameters = [
                'citus.all_modifications_commutative',
                'citus.background_task_queue_interval',
                'citus.cluster_name',
                'citus.coordinator_aggregation_strategy',
                'citus.count_distinct_error_rate',
                'citus.cpu_priority',
                'citus.cpu_priority_for_logical_replication_senders',
                'citus.defer_drop_after_shard_move',
                'citus.defer_drop_after_shard_split',
                'citus.defer_shard_delete_interval',
                'citus.desired_percent_disk_available_after_move',
                'citus.distributed_deadlock_detection_factor',
                'citus.enable_binary_protocol',
                'citus.enable_change_data_capture',
                'citus.enable_create_role_propagation',
                'citus.enable_deadlock_prevention',
                'citus.enable_local_execution',
                'citus.enable_local_reference_table_foreign_keys',
                'citus.enable_repartition_joins',
                'citus.enable_schema_based_sharding',
                'citus.enable_statistics_collection',
                'citus.explain_all_tasks',
                'citus.explain_analyze_sort_method',
                'citus.limit_clause_row_fetch_count',
                'citus.local_hostname',
                'citus.local_shared_pool_size',
                'citus.local_table_join_policy',
                'citus.log_remote_commands',
                'citus.max_adaptive_executor_pool_size',
                'citus.max_background_task_executors',
                'citus.max_background_task_executors_per_node',
                'citus.max_cached_connection_lifetime',
                'citus.max_cached_conns_per_worker',
                'citus.max_client_connections',
                'citus.max_high_priority_background_processes',
                'citus.max_intermediate_result_size',
                'citus.max_matview_size_to_auto_recreate',
                'citus.max_shared_pool_size',
                'citus.max_worker_nodes_tracked',
                'citus.multi_shard_modify_mode',
                'citus.multi_task_query_log_level',
                'citus.node_connection_timeout',
                'citus.node_conninfo',
                'citus.propagate_set_commands',
                'citus.recover_2pc_interval',
                'citus.remote_task_check_interval',
                'citus.shard_count',
                'citus.shard_replication_factor',
                'citus.show_shards_for_app_name_prefixes',
                'citus.skip_constraint_validation',
                'citus.skip_jsonb_validation_in_copy',
                'citus.stat_statements_track',
                'citus.stat_tenants_limit',
                'citus.stat_tenants_log_level',
                'citus.stat_tenants_period',
                'citus.stat_tenants_track',
                'citus.stat_tenants_untracked_sample_rate',
                'citus.task_assignment_policy',
                'citus.task_executor_type',
                'citus.use_citus_managed_tables',
                'citus.use_secondary_nodes',
                'citus.values_materialization_threshold',
                'citus.version',
                'citus.worker_min_messages',
                'citus.writable_standby_coordinator'
            ]
            
            # Use simple SHOW commands for each parameter
            successful_params = 0
            failed_params = 0
            
            for param in citus_parameters:
                try:
                    cursor.execute(f"SHOW {param}")
                    result = cursor.fetchone()
                    guc_params[param] = result[0] if result else 'NOT_AVAILABLE'
                    successful_params += 1
                except Exception as e:
                    guc_params[param] = f'ERROR: {str(e)}'
                    failed_params += 1
            
            # Add collection metadata
            guc_params['_collection_info'] = {
                'timestamp': datetime.now().isoformat(),
                'total_parameters_requested': len(citus_parameters),
                'successful_parameters': successful_params,
                'failed_parameters': failed_params,
                'method': 'SHOW_commands',
                'success_rate': f"{(successful_params/len(citus_parameters)*100):.1f}%"
            }
            
            log.info(f"Collected {successful_params}/{len(citus_parameters)} Citus GUC parameters using SHOW commands")
            return guc_params
            
        except Exception as e:
            log.error(f"Error collecting Citus GUC parameters: {e}")
            return {'error': str(e)}
    
    def _collect_citus_runtime_state(self, cursor) -> dict:
        """Collect current Citus runtime state."""
        try:
            citus_state = {}
            
            # 1. Get basic shard information
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as total_shards,
                    MIN(shardid) as min_shard_id,
                    MAX(shardid) as max_shard_id
                FROM pg_dist_shard 
                WHERE logicalrelid = '{self.table_name}'::regclass
            """)
            shard_summary = cursor.fetchone()
            if shard_summary:
                citus_state['shard_summary'] = {
                    'total_shards': shard_summary[0],
                    'min_shard_id': shard_summary[1],
                    'max_shard_id': shard_summary[2]
                }
            
            # 2. Get detailed shard distribution across workers
            cursor.execute(f"""
                SELECT 
                    s.shardid,
                    s.shardminvalue,
                    s.shardmaxvalue,
                    n.nodename,
                    n.nodeport,
                    p.shardstate,
                    CASE 
                        WHEN p.shardstate = 1 THEN 'ACTIVE'
                        WHEN p.shardstate = 3 THEN 'INACTIVE'
                        ELSE 'UNKNOWN'
                    END as state_description
                FROM pg_dist_shard s
                JOIN pg_dist_placement p ON s.shardid = p.shardid
                JOIN pg_dist_node n ON p.groupid = n.groupid
                WHERE s.logicalrelid = '{self.table_name}'::regclass
                ORDER BY s.shardid
            """)
            shard_details = cursor.fetchall()
            
            citus_state['shard_distribution'] = []
            for shard in shard_details:
                citus_state['shard_distribution'].append({
                    'shard_id': shard[0],
                    'min_hash_value': shard[1],
                    'max_hash_value': shard[2],
                    'worker_node': shard[3],
                    'worker_port': shard[4],
                    'shard_state': shard[5],
                    'state_description': shard[6]
                })
            
            # 3. Get worker node information
            cursor.execute("""
                SELECT 
                    nodename,
                    nodeport,
                    isactive,
                    noderole,
                    shouldhaveshards
                FROM pg_dist_node
                WHERE noderole = 'primary'
                ORDER BY nodename
            """)
            worker_nodes = cursor.fetchall()
            
            citus_state['worker_nodes'] = []
            for worker in worker_nodes:
                citus_state['worker_nodes'].append({
                    'node_name': worker[0],
                    'node_port': worker[1],
                    'is_active': worker[2],
                    'node_role': worker[3],
                    'should_have_shards': worker[4]
                })
            
            # 4. Calculate shards per worker
            cursor.execute(f"""
                SELECT 
                    n.nodename,
                    n.nodeport,
                    COUNT(*) as shard_count
                FROM pg_dist_shard s
                JOIN pg_dist_placement p ON s.shardid = p.shardid
                JOIN pg_dist_node n ON p.groupid = n.groupid
                WHERE s.logicalrelid = '{self.table_name}'::regclass
                AND p.shardstate = 1
                GROUP BY n.nodename, n.nodeport
                ORDER BY n.nodename
            """)
            shards_per_worker_results = cursor.fetchall()
            
            citus_state['shards_per_worker'] = []
            for worker_shard in shards_per_worker_results:
                citus_state['shards_per_worker'].append({
                    'worker_node': worker_shard[0],
                    'worker_port': worker_shard[1],
                    'shard_count': worker_shard[2]
                })
            
            # 5. Get row count per shard (entries per shard)
            # This requires querying each shard individually
            entries_per_shard = []
            total_entries = 0
            
            try:
                for shard_info in citus_state['shard_distribution']:
                    shard_id = shard_info['shard_id']
                    try:
                        # Query the specific shard table
                        cursor.execute(f"""
                            SELECT COUNT(*) 
                            FROM {self.table_name}_{shard_id}
                        """)
                        row_count = cursor.fetchone()[0]
                        entries_per_shard.append({
                            'shard_id': shard_id,
                            'worker_node': shard_info['worker_node'],
                            'entry_count': row_count
                        })
                        total_entries += row_count
                    except Exception as e:
                        # If direct shard query fails, try alternative method
                        entries_per_shard.append({
                            'shard_id': shard_id,
                            'worker_node': shard_info['worker_node'],
                            'entry_count': f'ERROR: {str(e)}'
                        })
                
                citus_state['entries_per_shard'] = entries_per_shard
                citus_state['total_entries_across_shards'] = total_entries
                
            except Exception as e:
                log.warning(f"Could not get per-shard row counts: {e}")
                
                # Alternative: Get total table row count
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                    total_rows = cursor.fetchone()[0]
                    citus_state['total_table_rows'] = total_rows
                    
                    # Estimate entries per shard
                    if citus_state['shard_summary']['total_shards'] > 0:
                        avg_entries_per_shard = total_rows / citus_state['shard_summary']['total_shards']
                        citus_state['estimated_avg_entries_per_shard'] = round(avg_entries_per_shard, 2)
                    
                except Exception as e2:
                    citus_state['row_count_error'] = str(e2)
            
            # 6. Get distribution quality metrics
            if citus_state.get('shards_per_worker'):
                shard_counts = [worker['shard_count'] for worker in citus_state['shards_per_worker']]
                if shard_counts:
                    min_shards = min(shard_counts)
                    max_shards = max(shard_counts)
                    avg_shards = sum(shard_counts) / len(shard_counts)
                    
                    citus_state['distribution_quality'] = {
                        'min_shards_per_worker': min_shards,
                        'max_shards_per_worker': max_shards,
                        'avg_shards_per_worker': round(avg_shards, 2),
                        'distribution_variance': round(max_shards - min_shards, 2),
                        'is_balanced': max_shards - min_shards <= 1  # Within 1 shard difference
                    }
            
            # 7. Add collection metadata
            citus_state['_collection_info'] = {
                'timestamp': datetime.now().isoformat(),
                'table_name': self.table_name,
                'total_workers': len(citus_state.get('worker_nodes', [])),
                'active_workers': len([w for w in citus_state.get('worker_nodes', []) if w['is_active']]),
                'method': 'citus_metadata_queries'
            }
            
            log.info(f"Collected Citus runtime state: {citus_state['shard_summary']['total_shards']} shards across {citus_state['_collection_info']['active_workers']} workers")
            return citus_state
            
        except Exception as e:
            log.error(f"Error collecting Citus runtime state: {e}")
            return {'error': str(e)}
    
    # COMMENTED OUT SKELETON METHODS - Not needed for current use case
    # Uncomment and implement when needed in the future
    
    # def _collect_current_table_stats(self, cursor) -> dict:
    #     """Collect current table statistics while data is loaded."""
    #     # TODO: Implement table size, row count, etc. queries here
    #     # Example: cursor.execute(f"SELECT pg_size_pretty(pg_total_relation_size('{self.table_name}'))")
    #     return {'placeholder': 'implement_table_stats_queries_here'}
    
    # def _collect_session_state(self, cursor) -> dict:
    #     """Collect current session and connection state."""
    #     # TODO: Implement session info queries here
    #     # Example: cursor.execute("SELECT current_database(), current_user, version()")
    #     return {'placeholder': 'implement_session_queries_here'}
    
    # def _collect_index_statistics(self, cursor) -> dict:
    #     """Collect current index usage statistics."""
    #     # TODO: Implement index stats queries here
    #     # Example: cursor.execute("SELECT * FROM pg_stat_user_indexes WHERE tablename = %s", (self.table_name,))
    #     return {'placeholder': 'implement_index_stats_queries_here'}
    
    # def _collect_memory_usage(self, cursor) -> dict:
    #     """Collect current memory usage information."""
    #     # TODO: Implement memory usage queries here
    #     # Example: cursor.execute("SELECT name, setting FROM pg_settings WHERE name LIKE '%mem%'")
    #     return {'placeholder': 'implement_memory_queries_here'}
    
    # def _collect_query_performance_settings(self, cursor) -> dict:
    #     """Collect current query performance related settings."""
    #     # TODO: Implement query performance settings queries here
    #     # Example: cursor.execute("SELECT name, setting FROM pg_settings WHERE name LIKE 'enable_%'")
    #     return {'placeholder': 'implement_performance_queries_here'}
