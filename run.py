#run.py
import argparse
import time
import subprocess
import os
import logging
from contextlib import redirect_stdout
from benchmark_run_utils import *

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

os.environ["LOG_LEVEL"] = "DEBUG"

def main():
    parser = argparse.ArgumentParser(description="Run HNSW benchmark")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and output directory without executing")
    args = parser.parse_args()

    config = load_config("config.json")
    benchmark_info = config["benchmark-info"]
    start_time = time.time()
    start_timeh = time.strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"Benchmark run start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    for case in config['cases']:
        print(f"Running case: {case['db-label']}")
        setup_database(config)
        run_benchmark(case, config['database'], config["benchmark-info"], args.dry_run)
        teardown_database(config)
    end_timeh = time.strftime('%Y-%m-%d %H:%M:%S')
    output_dir = get_output_dir_path(case, benchmark_info, [], 0, db_config=config['database'], base_dir=True)
    if not args.dry_run:
        generate_benchmark_metadata(config, start_timeh, end_timeh, output_dir)

    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Benchmark run end time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"COMPLETED ALL EXECUTIONS. total_duration={execution_time}")


def run_benchmark(case, db_config, benchmark_info, dry_run=False):
    base_command = get_base_command(case, db_config)
    run_count = case.get("run-count", 1)  # Default to 1 if not specified
    for run in range(run_count):
        print(f"Starting run {run + 1} of {run_count} for case: {case['db-label']}")
        for i, search_params in enumerate(generate_combinations(case["search-params"])):
            command = base_command + search_params
            if case["index-type"] == "hnsw-bq" and "reranking" in case:
                if case.get("half-quantized-fetch-limit", False):
                    command += ["--quantized-fetch-limit", str(int(int(search_params[1]) / 2))]
                else:
                    command += ["--quantized-fetch-limit", search_params[1]]

            if i > 0 or run > 0:
                command = handle_drop_old_load_flags(command)

            if dry_run:
                logger.info(f"Command: {' '.join(command)}")
                logger.info(f"Output Dir: {get_output_dir_path(case, benchmark_info, search_params, run, db_config)}")
                logger.info(f"Extra Information: {get_extension_version(db_config)} \n")
            else:
                try:
                    output_dir = get_output_dir_path(case, benchmark_info, search_params, run, db_config)
                    os.environ["RESULTS_LOCAL_DIR"] = output_dir
                    os.makedirs(output_dir, exist_ok=True)

                    with open(f"{output_dir}/log.txt", 'w') as f:
                        print_configuration(case, benchmark_info, db_config, command, f)
                        run_pre_warm(db_config, case)
                        f.flush()

                        logger.info("***********START***********")
                        start_time = time.time()
                        # Capture both stdout and stderr and write them to the log file
                        subprocess.run(command, check=True, stdout=f, stderr=f)
                        end_time = time.time()
                        execution_time = end_time - start_time
                        logger.info(f"total_duration={execution_time}")
                        logger.info("***********END***********")

                        with redirect_stdout(f):
                            get_stats(db_config)
                            f.flush()
                        f.flush()

                    # Print output directory at the end with clickable link
                    logger.info(f"Results saved to:")
                    logger.info(f"   file://{os.path.abspath(output_dir)}")
                    logger.info("=" * 40)

                except subprocess.CalledProcessError as e:
                    logger.error(f"Benchmark Failed: {e}")
                logger.info("Sleeping for 1 min")
                time.sleep(60)

if __name__ == "__main__":
    main()

