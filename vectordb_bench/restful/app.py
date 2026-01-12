from flask import Flask, jsonify, request

from vectordb_bench.backend.clients import DB
from vectordb_bench.interface import benchmark_runner
from vectordb_bench.models import ALL_TASK_STAGES, CaseConfig, TaskConfig, TaskStage
from vectordb_bench.restful.format_res import format_results

app = Flask(__name__)


def res_wrapper(code: int = 0, message: str = "", data: any = None):  # noqa: RUF013
    return jsonify({"code": code, "message": message, "data": data}), 200


def success_res(data: any = None, message: str = "success"):  # noqa: RUF013
    return res_wrapper(code=0, message=message, data=data)


def failed_res(data: any = None, message: str = "failed"):  # noqa: RUF013
    return res_wrapper(code=1, message=message, data=data)


@app.route("/get_res", methods=["GET"])
def get_res():
    """task label -> res"""
    task_label = request.args.get("task_label", "standard")
    all_results = benchmark_runner.get_results()
    res = format_results(all_results, task_label=task_label)

    return success_res(res)


@app.route("/get_status", methods=["GET"])
def get_status():
    "running 5/18, not running"
    is_running = benchmark_runner.has_running()
    tasks_count = benchmark_runner.get_tasks_count()
    if is_running:
        tasks_count = benchmark_runner.get_tasks_count()
        cur_task_idx = benchmark_runner.get_current_task_id()
        return success_res(
            data={
                "is_running": is_running,
                "tasks_count": tasks_count,
                "cur_task_idx": cur_task_idx,
            }
        )
    return success_res(data={"is_running": is_running})


@app.route("/stop", methods=["GET"])
def stop():
    benchmark_runner.stop_running()
    return success_res(message="stopped")


@app.route("/run", methods=["post"])
def run():
    if benchmark_runner.has_running():
        return failed_res(message="There are already running tasks.")
    data = request.get_json()
    task_label = data.get("task_label", "test")
    use_aliyun = data.get("use_aliyun", False)
    task_configs: list[TaskConfig] = []
    try:
        tasks = data.get("tasks", [])
        if len(tasks) == 0:
            return failed_res(message="empty tasks")
        for task in tasks:
            db = DB(task["db"])
            db_config = db.config_cls(**task["db_config"])
            case_config = CaseConfig(**task["case_config"])
            print(case_config)  # noqa: T201
            db_case_config = db.case_config_cls(index_type=task["db_case_config"].get("index", None))(
                **task["db_case_config"]
            )
            stages = [TaskStage(stage) for stage in task.get("stages", ALL_TASK_STAGES)]
            print(stages)  # noqa: T201
            task_config = TaskConfig(
                db=db,
                db_config=db_config,
                case_config=case_config,
                db_case_config=db_case_config,
                stages=stages,
            )
            task_configs.append(task_config)
    except Exception as e:
        return failed_res(message=f"invalid tasks: {e}")

    benchmark_runner.set_download_address(use_aliyun)
    benchmark_runner.run(task_configs, task_label)

    return success_res(message="start")


def main():
    app.run(host="0.0.0.0", port=5000, debug=False)  # noqa: S104


if __name__ == "__main__":
    main()
