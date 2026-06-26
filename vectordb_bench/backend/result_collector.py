import argparse
import logging
import pathlib
import sys
from datetime import date
from typing import Literal

from vectordb_bench.models import TestResult

log = logging.getLogger(__name__)

ResultGroupBy = Literal["run_id", "db"]


class ResultCollector:
    @staticmethod
    def _group_key(file_result: TestResult, group_by: ResultGroupBy) -> str:
        if group_by == "run_id":
            return file_result.run_id

        if not file_result.results:
            return file_result.run_id
        return file_result.results[0].task_config.db.value

    @classmethod
    def collect(
        cls,
        result_dir: pathlib.Path,
        group_by: ResultGroupBy = "run_id",
        trans_unit: bool = True,
    ) -> list[TestResult]:
        reg = "result_*.json"
        results_d = {}
        if group_by not in {"run_id", "db"}:
            msg = f"Unsupported result grouping: {group_by}"
            raise ValueError(msg)

        if not result_dir.exists() or len(list(result_dir.rglob(reg))) == 0:
            return []

        for json_file in sorted(result_dir.rglob(reg)):
            file_result = TestResult.read_file(json_file, trans_unit=trans_unit)
            key = cls._group_key(file_result, group_by)

            # Default behavior groups files from the same run. FTS publish artifacts can
            # opt into DB grouping so one backend owns one consolidated result file.
            if key in results_d:
                results_d[key].results.extend(file_result.results)
            else:
                results_d[key] = file_result

        return list(results_d.values())

    @classmethod
    def merge_by_db(
        cls,
        result_dir: pathlib.Path,
        task_label: str = "fts_standard",
        replace: bool = False,
        dry_run: bool = False,
    ) -> list[pathlib.Path]:
        source_files = sorted(result_dir.rglob("result_*.json"))
        merged_results = cls.collect(result_dir, group_by="db", trans_unit=False)
        output_files = []

        for result in merged_results:
            if not result.results:
                continue

            db = result.results[0].task_config.db.value
            db_lower = db.lower()
            merged = TestResult(
                run_id=f"{task_label}_{db_lower}",
                task_label=task_label,
                results=result.results,
                timestamp=result.timestamp,
            )
            file_name = merged.file_fmt.format(date.today().strftime("%Y%m%d"), task_label, db_lower)
            output_file = result_dir.joinpath(db, file_name)
            output_files.append(output_file)

            if dry_run:
                log.info("Would write %s with %s case results", output_file, len(result.results))
            else:
                merged.write_db_file(result_dir.joinpath(db), merged, db_lower)

        if replace and not dry_run:
            output_file_set = set(output_files)
            for source_file in source_files:
                if source_file not in output_file_set and source_file.exists():
                    source_file.unlink()

        return output_files


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect or consolidate VDBBench result JSON files.")
    parser.add_argument("result_dir", type=pathlib.Path)
    parser.add_argument("--merge-by-db", action="store_true", help="write one consolidated result file per backend")
    parser.add_argument(
        "--replace", action="store_true", help="remove source result files after merged files are written"
    )
    parser.add_argument("--task-label", default="fts_standard", help="task label used for merged result files")
    parser.add_argument("--dry-run", action="store_true", help="show planned output files without writing")
    return parser.parse_args()


def main():
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    if args.merge_by_db:
        output_files = ResultCollector.merge_by_db(
            args.result_dir,
            task_label=args.task_label,
            replace=args.replace,
            dry_run=args.dry_run,
        )
        for output_file in output_files:
            sys.stdout.write(f"{output_file}\n")
        return

    for result in ResultCollector.collect(args.result_dir):
        sys.stdout.write(f"{result.run_id}\t{len(result.results)}\n")


if __name__ == "__main__":
    main()
