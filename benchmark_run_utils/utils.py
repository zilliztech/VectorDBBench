#utils.py
import json
import time
from typing import List, Optional
import psycopg2
import os
import logging
from psycopg2 import sql
from contextlib import redirect_stdout
from itertools import product

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

os.environ["LOG_LEVEL"] = "DEBUG"


def load_config(json_file):
    with open(json_file, 'r') as file:
        config = json.load(file)
    return config


def setup_database(config):
    try:
        conn = psycopg2.connect(
            dbname='postgres',
            user=config['database']['username'],
            password=config['database']['password'],
            host=config['database']['host']
        )
        conn.autocommit = True
        cursor = conn.cursor()
        # Create the database if it doesn't exist
        cursor.execute(sql.SQL("SELECT 1 FROM pg_database WHERE datname = %s"), [config['database']['db-name']])
        if not cursor.fetchone():
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(config['database']['db-name'])))
        conn.close()

        # Connect to the new database to create the extension
        conn = psycopg2.connect(
            dbname=config['database']['db-name'],
            user=config['database']['username'],
            password=config['database']['password'],
            host=config['database']['host']
        )
        cursor = conn.cursor()
        print("Creating required extensions")
        for ext in ["pg_buffercache", "pg_prewarm", "vector", "pg_diskann", "vectorscale"]:
            try:
                cursor.execute(f"CREATE EXTENSION IF NOT EXISTS {ext};")
                conn.commit()
                print(f"Extension {ext} installed in database [{config['database']['db-name']}]")
            except Exception as e:
                logger.error(f"Installing {ext} extension failed: {e}")
        conn.close()
    except Exception as e:
        print(f"Setup failed: {e}")


def get_stats(config):
    with open('queries.json', 'r') as file:
        queries = json.load(file)
    try:
        conn = psycopg2.connect(
            dbname=config['db-name'],
            user=config['username'],
            password=config['password'],
            host=config['host']
        )
        cur = conn.cursor()
        for item in queries:
            query = item['query']
            description = item['description']
            print(f"\nRunning query: {description}")
            try:
                cur.execute(query)
                rows = cur.fetchall()
                headers = [desc[0] for desc in cur.description]
                print(f"{' | '.join(headers)}")
                for row in rows:
                    print(f"{' | '.join(map(str, row))}")
            except Exception as e:
                print(f"Failed to run query: {e}")
        conn.close()
    except Exception as e:
        print(f"Setup failed: {e}")
    finally:
        conn.close()


def run_pre_warm(config: dict, case: dict):
    print(f"Running pre warm for database:{config['db-name']}")
    indexes = {
        "pgvectorhnsw": "public.pgvector_index",
        "pgdiskann": "public.pgdiskann_index",
    }
    index_name = indexes.get(case["vdb-command"], "public.pgvector_index")
    try:
        conn = psycopg2.connect(
                dbname=config['db-name'],
                user=config['username'],
                password=config['password'],
                host=config['host'],
        )
        cursor = conn.cursor()
        cursor.execute(f"SELECT pg_prewarm('{index_name}') as block_loaded")
        conn.commit()

        result = cursor.fetchone()
        print(f"Pre-warm blocks loaded: {result[0]}")
        conn.close()
    except Exception as e:
        print(f"Failed to pre-warm the database: {e}")


def teardown_database(config):
    # Optionally drop the database after the test
    pass


def query_configurations(config):
    # List of configuration parameters to query
    config_queries = [
        "SHOW checkpoint_timeout;",
        "SHOW effective_cache_size;",
        "SHOW jit;",
        "SHOW maintenance_work_mem;",
        "SHOW max_parallel_maintenance_workers;",
        "SHOW max_parallel_workers;",
        "SHOW max_parallel_workers_per_gather;",
        "SHOW max_wal_size;",
        "SHOW max_worker_processes;",
        "SHOW shared_buffers;",
        "SHOW wal_compression;",
        "SHOW work_mem;"
    ]

    try:
        conn = psycopg2.connect(
            dbname=config['db-name'],
            user=config['username'],
            password=config['password'],
            host=config['host']
        )
        cursor = conn.cursor()
        results = []

        # Execute each query and collect the result
        for query in config_queries:
            cursor.execute(query)
            result = cursor.fetchone()
            results.append(result[0] if result else None)

        # Print the raw output to debug
        print("Raw query results:", results)

        config_dict = {
            "checkpoint_timeout": results[0],
            "effective_cache_size": results[1],
            "jit": results[2],
            "maintenance_work_mem": results[3],
            "max_parallel_maintenance_workers": results[4],
            "max_parallel_workers": results[5],
            "max_parallel_workers_per_gather": results[6],
            "max_wal_size": results[7],
            "max_worker_processes": results[8],
            "shared_buffers": results[9],
            "wal_compression": results[10],
            "work_mem": results[11]
        }

        conn.close()
        return config_dict
    except Exception as e:
        print(f"Failed to query configurations: {e}")
        return {}


def get_base_command(case: dict, db_config: dict) -> list:
    base_command = [
        "vectordbbench", case["vdb-command"],
        "--user-name", db_config["username"],
        "--password", db_config["password"],
        "--host", db_config["host"],
        "--db-name", db_config["db-name"],
        "--case-type", case["case-type"],
        "--num-concurrency", case["num-concurrency"],
        "--concurrency-duration", str(case["concurrency-duration"]),
        "--k", str(case["k"]),
    ]

    # Handle initial flags (no skip for the first ef_search)
    if case.get("drop-old", True):
        base_command.append("--drop-old")
    else:
        base_command.append("--skip-drop-old")

    if case.get("load", True):
        base_command.append("--load")
    else:
        base_command.append("--skip-load")

    if case.get("search-serial", True):
        base_command.append("--search-serial")
    else:
        base_command.append("--skip-search-serial")

    if case.get("search-concurrent", True):
        base_command.append("--search-concurrent")
    else:
        base_command.append("--skip-search-concurrent")
    
    if "reranking" in case:
        if case.get("reranking", True):
            base_command.append("--reranking")
        else:
            base_command.append("--skip-reranking")
    
    for key, value in case["index-params"].items():
        base_command.extend([f"--{key}", str(value)])

    return base_command

def handle_drop_old_load_flags(command) -> list[str]:
    """If --drop-old or --load flags are present, remove them and add skip flags"""
    command = [arg for arg in command if arg not in ["--drop-old", "--load"]]
    if "--skip-drop-old" not in command:
        command.append("--skip-drop-old")
    if "--skip-load" not in command:
        command.append("--skip-load")
    return command

def get_extension_version(db_config: dict):
    try:
        conn = psycopg2.connect(
            dbname=db_config['db-name'],
            user=db_config['username'],
            password=db_config['password'],
            host=db_config['host']
        )
        cursor = conn.cursor()
        cursor.execute("SELECT extname, extversion FROM pg_extension WHERE extname LIKE '%vec%' OR extname LIKE '%ann%' OR extname = 'scann';")
        extensions = cursor.fetchall()
        conn.close()
        extensions = {ext[0]: ext[1] for ext in extensions}
        return extensions
    except Exception as e:
        print(f"Failed to get extension versions: {e}")
        return {}

def get_postgres_version(db_config: dict):
    try:
        conn = psycopg2.connect(
            dbname=db_config['db-name'],
            user=db_config['username'],
            password=db_config['password'],
            host=db_config['host']
        )
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        pgversion = cursor.fetchone()
        conn.close()
        return pgversion[0]
    except Exception as e:
        print(f"Failed to get extension versions: {e}")
        return ""
    
def get_output_dir_path(
    case: dict,
    benchmark_info: dict,
    search_params: Optional[List[str | int]],
    run: Optional[int],
    db_config: dict,
    base_dir: bool = False,
) -> str:
    ext_version = get_extension_version(db_config)
    output_dir = f"results/{case['vector-ext']}-{ext_version.get(case['vector-ext'], '')}/{case['index-type']}/{case['db-label']}/{benchmark_info['provider']}/{benchmark_info['instance-service']}/{benchmark_info['instance-service']}/{case['case-type']}/"
    if base_dir:
        return output_dir

    for key, value in case["index-params"].items():
        if key not in ["maintenance-work-mem", "max-parallel-workers"]:
            output_dir += f"{value}-"
    for val in search_params:
        if val.isdigit():
            output_dir += f"{val}-"
            if case["index-type"] == "hnsw-bq" and "reranking" in case:
                if case.get("half-quantized-fetch-limit", False):
                    output_dir += f"{int(int(val) / 2)}-"
                else:
                    output_dir += f"{val}-"
    output_dir += f"{run}-{int(time.time())}"
    return output_dir

def print_configuration(
    case: dict,
    benchmark_info: dict,
    db_config: dict,
    command: list,
    output_file,
):
    with redirect_stdout(output_file):
        print("Benchmark Information:")
        for key, value in benchmark_info.items():
            print(f"{key}: {value}")
        output_file.flush()
        
        print("Benchmark Test Run Information:")
        for key, value in case.items():
            print(f"{key}: {value}")
        output_file.flush()
        
        print(f"Postgres Database Configuration")
        current_configs = query_configurations(db_config)
        for key, value in current_configs.items():
            print(f"{key}: {value}")
        output_file.flush()
        
        print(f"Get Buffer Hit Ratio Stats")
        get_stats(db_config)
        output_file.flush()

        print(f"Running command: {' '.join(command)}")
        output_file.flush()
        logger.info(f"Running command: {' '.join(command)}")

def generate_combinations(config_dict: dict) -> list:
    keys = []
    values = []
    for key, value in config_dict.items():
        keys.append(f"--{key}")
        if isinstance(value, list):
            values.append(value)
        else:
            values.append([value])

    combinations = []
    for combo in product(*values):
        combination = []
        for k, v in zip(keys, combo):
            combination.append((k, str(v)))
        combinations.append([item for pair in combination for item in pair])
    
    logger.info(f"Total combinations generated: {len(combinations)}")
    return combinations

def generate_benchmark_metadata(
    metadata: dict,
    start_time: str,
    end_time: str,
    output_dir: str,
):
    metadata["benchmark-info"]["extension-versions"] = get_extension_version(metadata['database'])
    metadata["benchmark-info"]["postgres_version"] = get_postgres_version(metadata['database'])
    metadata["benchmark-info"]["start_time"] = start_time
    metadata["benchmark-info"]["end_time"] = end_time
    del metadata["database"]

    output_filename = f"{output_dir}benchmark_metadata.json"
    with open(output_filename, "w") as f:
        json.dump(metadata, f, indent=4)
        logger.info(f"Benchmark metadata saved to {output_filename}")
