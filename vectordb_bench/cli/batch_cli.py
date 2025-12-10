import logging
from collections.abc import MutableMapping
from concurrent.futures import wait
from pathlib import Path
from typing import Annotated, Any, TypedDict

import click
from click.testing import CliRunner
from yaml import Loader, load

from .. import config
from ..cli.cli import (
    cli,
    click_parameter_decorators_from_typed_dict,
)
from ..models import TaskConfig

log = logging.getLogger(__name__)


def click_get_defaults_from_file(ctx, param, value):  # noqa: ANN001, ARG001
    if not value:
        raise click.MissingParameter
    path = Path(value)
    input_file = path if path.exists() else Path(config.CONFIG_LOCAL_DIR, path)
    try:
        with input_file.open() as f:
            _config: dict[str, list[dict[str, Any]]] = load(f.read(), Loader=Loader)  # noqa: S506
            ctx.default_map = _config
    except Exception as e:
        msg = f"Failed to load batch config file: {e}"
        raise click.BadParameter(msg) from e
    return value


class BatchCliTypedDict(TypedDict):
    batch_config_file: Annotated[
        bool,
        click.option(
            "--batch-config-file",
            type=click.Path(),
            callback=click_get_defaults_from_file,
            is_eager=True,
            expose_value=False,
            help="Read batch configuration from yaml file",
        ),
    ]


def build_sub_cmd_args(batch_config: MutableMapping[str, Any] | None):
    bool_options = {
        "drop_old": True,
        "load": True,
        "search_serial": True,
        "search_concurrent": True,
        "dry_run": False,
        "custom_dataset_use_shuffled": True,
        "custom_dataset_with_gt": True,
    }

    def format_option(key: str, value: Any):
        opt_name = key.replace("_", "-")

        if key in bool_options:
            return format_bool_option(opt_name, value, skip=False)

        if key.startswith("skip_"):
            raw_key = key[5:]
            raw_opt = raw_key.replace("_", "-")
            return format_bool_option(raw_opt, value, skip=True, raw_key=raw_key)

        return [f"--{opt_name}", str(value)]

    def format_bool_option(opt_name: str, value: Any, skip: bool = False, raw_key: str | None = None):
        if isinstance(value, bool):
            if skip:
                if bool_options.get(raw_key, False):
                    return [f"--skip-{opt_name}"] if value else [f"--{opt_name}"]
                return [f"--{opt_name}", str(value)]
            if value:
                return [f"--{opt_name}"]
            if bool_options.get(opt_name.replace("-", "_"), False):
                return [f"--skip-{opt_name}"]
            return []
        return [f"--{opt_name}", str(value)]

    args_arr = []
    for sub_cmd_key, sub_cmd_config_list in batch_config.items():
        for sub_cmd_args in sub_cmd_config_list:
            args = [sub_cmd_key]
            for k, v in sub_cmd_args.items():
                args.extend(format_option(k, v))
            args_arr.append(args)

    return args_arr


def build_task_from_config(cmd_name: str, config_dict: dict[str, Any]) -> TaskConfig | None:

    collected_tasks = []
    original_run = None

    try:
        from ..interface import benchmark_runner

        original_run = benchmark_runner.run

        def collect_task_wrapper(tasks: list[TaskConfig], task_label: str | None = None):  # noqa: ARG001
            collected_tasks.extend(tasks)
            return True

        benchmark_runner.run = collect_task_wrapper

        # build CLI parameters
        args = [cmd_name]
        bool_options = {
            "drop_old": True,
            "load": True,
            "search_serial": True,
            "search_concurrent": True,
            "dry_run": False,
            "custom_dataset_use_shuffled": True,
            "custom_dataset_with_gt": True,
        }

        def format_option(key: str, value: Any):
            opt_name = key.replace("_", "-")

            if key in bool_options:
                return format_bool_option(opt_name, value, skip=False)

            if key.startswith("skip_"):
                raw_key = key[5:]
                raw_opt = raw_key.replace("_", "-")
                return format_bool_option(raw_opt, value, skip=True, raw_key=raw_key)

            return [f"--{opt_name}", str(value)]

        def format_bool_option(opt_name: str, value: Any, skip: bool = False, raw_key: str | None = None):
            if isinstance(value, bool):
                if skip:
                    if bool_options.get(raw_key, False):
                        return [f"--skip-{opt_name}"] if value else [f"--{opt_name}"]
                    return [f"--{opt_name}", str(value)]
                if value:
                    return [f"--{opt_name}"]
                if bool_options.get(opt_name.replace("-", "_"), False):
                    return [f"--skip-{opt_name}"]
                return []
            return [f"--{opt_name}", str(value)]

        for k, v in config_dict.items():
            args.extend(format_option(k, v))

        # call CLI command (this will trigger collect_task_wrapper)
        runner = CliRunner()
        result = runner.invoke(cli, args, catch_exceptions=False)

        if result.exception:
            log.error(f"Failed to build task for {cmd_name}: {result.exception}")
            return None

        if collected_tasks:
            return collected_tasks[0]
        return None  # noqa: TRY300

    except Exception:
        log.exception("Error building task from config")
        return None
    finally:
        if original_run is not None:
            from ..interface import benchmark_runner

            benchmark_runner.run = original_run


@cli.command()
@click_parameter_decorators_from_typed_dict(BatchCliTypedDict)
def BatchCli():
    ctx = click.get_current_context()
    batch_config = ctx.default_map

    from ..interface import benchmark_runner, global_result_future

    # collect all tasks
    all_tasks: list[TaskConfig] = []
    task_labels: set[str] = set()

    for cmd_name, cmd_config_list in batch_config.items():
        for config_dict in cmd_config_list:
            log.info(f"Building task for {cmd_name} with config: {config_dict.get('task_label', 'N/A')}")

            # collect task_label from config
            if "task_label" in config_dict:
                task_labels.add(config_dict["task_label"])

            # TaskConfig
            task = build_task_from_config(cmd_name, config_dict)
            if task:
                all_tasks.append(task)
                log.info(f"Successfully built task: {task.db.value} - {task.case_config.case_id.name}")
            else:
                log.warning(f"Failed to build task for {cmd_name}")

    if not all_tasks:
        log.error("No tasks were built from the batch config file")
        return

    if len(task_labels) == 1:
        task_label = task_labels.pop()
        log.info(f"Using shared task_label from config: {task_label}")
    elif len(task_labels) > 1:
        task_label = next(iter(task_labels))
        log.warning(f"Multiple task_labels found in config, using the first one: {task_label}")
    else:
        from datetime import datetime

        task_label = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log.info(f"No task_label found in config, using generated one: {task_label}")

    log.info(f"Running {len(all_tasks)} tasks with shared task_label: {task_label}")

    benchmark_runner.run(all_tasks, task_label)

    if global_result_future:
        log.info("Waiting for all tasks to complete...")
        wait([global_result_future])
        log.info("All tasks completed successfully")
    else:
        log.warning("No global_result_future found, tasks may be running in background")
