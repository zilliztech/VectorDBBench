import logging
import time
from concurrent.futures import wait
from datetime import datetime
from pprint import pformat
from typing import (
    Annotated,
    Callable,
    List,
    Optional,
    Type,
    TypedDict,
    Unpack,
    get_origin,
    get_type_hints,
    Dict,
    Any,
)
import click

from vectordb_bench.backend.clients.api import MetricType
from .. import config
from ..backend.clients import DB
from ..interface import benchMarkRunner, global_result_future
from ..models import (
    CaseConfig,
    CaseType,
    ConcurrencySearchConfig,
    DBCaseConfig,
    DBConfig,
    TaskConfig,
    TaskStage,
)
import os
from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def click_get_defaults_from_file(ctx, param, value):
    if value:
        if os.path.exists(value):
            input_file = value
        else:
            input_file = os.path.join(config.CONFIG_LOCAL_DIR, value)
        try:
            with open(input_file, 'r') as f:
                _config: Dict[str, Dict[str, Any]] = load(f.read(), Loader=Loader)
                ctx.default_map = _config.get(ctx.command.name, {})
        except Exception as e:
            raise click.BadParameter(f"Failed to load config file: {e}")
    return value


def click_parameter_decorators_from_typed_dict(
    typed_dict: Type,
) -> Callable[[click.decorators.FC], click.decorators.FC]:
    """A convenience method decorator that will read in a TypedDict with parameters defined by Annotated types.
    from .models import CaseConfig, CaseType, DBCaseConfig, DBConfig, TaskConfig, TaskStage
    The click.options will be collected and re-composed as a single decorator to apply to the click.command.

    Args:
        typed_dict (TypedDict) with Annotated[..., click.option()] keys

    Returns:
        a fully decorated method


    For clarity, the key names of the TypedDict will be used to determine the type hints for the input parameters.
    The actual function parameters are controlled by the click.option definitions. You must manually ensure these are aligned in a sensible way!

    Example:
    ```
    class CommonTypedDict(TypedDict):
        z: Annotated[int, click.option("--z/--no-z", is_flag=True, type=bool, help="help z", default=True, show_default=True)]
        name: Annotated[str, click.argument("name", required=False, default="Jeff")]

    class FooTypedDict(CommonTypedDict):
        x: Annotated[int, click.option("--x", type=int, help="help x", default=1, show_default=True)]
        y: Annotated[str, click.option("--y", type=str, help="help y", default="foo", show_default=True)]

    @cli.command()
    @click_parameter_decorators_from_typed_dict(FooTypedDict)
    def foo(**parameters: Unpack[FooTypedDict]):
        "Foo docstring"
        print(f"input parameters: {parameters["x"]}")
    ```
    """
    decorators = []
    for _, t in get_type_hints(typed_dict, include_extras=True).items():
        assert get_origin(t) is Annotated
        if (
            len(t.__metadata__) == 1
            and t.__metadata__[0].__module__ == "click.decorators"
        ):
            # happy path -- only accept Annotated[..., Union[click.option,click.argument,...]] with no additional metadata defined (len=1)
            decorators.append(t.__metadata__[0])
        else:
            raise RuntimeError(
                "Click-TypedDict decorator parsing must only contain root type and a click decorator like click.option. See docstring"
            )

    def deco(f):
        for dec in reversed(decorators):
            f = dec(f)
        return f

    return deco


def click_arg_split(ctx: click.Context, param: click.core.Option, value):
    """Will split a comma-separated list input into an actual list.

    Args:
        ctx (...): unused click arg
        param (...): unused click arg
        value (str): input comma-separated list

    Returns:
        value (List[str]): list of original
    """
    # split columns by ',' and remove whitespace
    if value is None:
        return []
    return [c.strip() for c in value.split(",") if c.strip()]


def parse_task_stages(
    drop_old: bool,
    load: bool,
    search_serial: bool,
    search_concurrent: bool,
) -> List[TaskStage]:
    stages = []
    if load and not drop_old:
        raise RuntimeError("Dropping old data cannot be skipped if loading data")
    elif drop_old and not load:
        raise RuntimeError("Load cannot be skipped if dropping old data")
    if drop_old:
        stages.append(TaskStage.DROP_OLD)
    if load:
        stages.append(TaskStage.LOAD)
    if search_serial:
        stages.append(TaskStage.SEARCH_SERIAL)
    if search_concurrent:
        stages.append(TaskStage.SEARCH_CONCURRENT)
    return stages


def check_custom_case_parameters(ctx, param, value):
    if ctx.params.get("case_type") == "PerformanceCustomDataset":
        if value is None:
            raise click.BadParameter("Custom case parameters\
                                     \n--custom-case-name\n--custom-dataset-name\n--custom-dataset-dir\n--custom-dataset-size \
                                     \n--custom-dataset-dim\n--custom-dataset-file-count\n are required")
    return value


def get_custom_case_config(parameters: dict) -> dict:
    custom_case_config = {}
    if parameters["case_type"] == "PerformanceCustomDataset":
        custom_case_config = {
            "name": parameters["custom_case_name"],
            "description": parameters["custom_case_description"],
            "load_timeout": parameters["custom_case_load_timeout"],
            "optimize_timeout": parameters["custom_case_optimize_timeout"],
            "dataset_config": {
                "name": parameters["custom_dataset_name"],
                "dir": parameters["custom_dataset_dir"],
                "size": parameters["custom_dataset_size"],
                "dim": parameters["custom_dataset_dim"],
                "metric_type": parameters["custom_dataset_metric_type"],
                "file_count": parameters["custom_dataset_file_count"],
                "use_shuffled": parameters["custom_dataset_use_shuffled"],
                "with_gt": parameters["custom_dataset_with_gt"],
            }
        }
    return custom_case_config


log = logging.getLogger(__name__)


class CommonTypedDict(TypedDict):
    config_file: Annotated[
        bool,
        click.option('--config-file',
                     type=click.Path(),
                     callback=click_get_defaults_from_file,
                     is_eager=True,
                     expose_value=False,
                     help='Read configuration from yaml file'),
    ]
    drop_old: Annotated[
        bool,
        click.option(
            "--drop-old/--skip-drop-old",
            type=bool,
            default=True,
            help="Drop old or skip",
            show_default=True,
        ),
    ]
    load: Annotated[
        bool,
        click.option(
            "--load/--skip-load",
            type=bool,
            default=True,
            help="Load or skip",
            show_default=True,
        ),
    ]
    search_serial: Annotated[
        bool,
        click.option(
            "--search-serial/--skip-search-serial",
            type=bool,
            default=True,
            help="Search serial or skip",
            show_default=True,
        ),
    ]
    search_concurrent: Annotated[
        bool,
        click.option(
            "--search-concurrent/--skip-search-concurrent",
            type=bool,
            default=True,
            help="Search concurrent or skip",
            show_default=True,
        ),
    ]
    case_type: Annotated[
        str,
        click.option(
            "--case-type",
            type=click.Choice([ct.name for ct in CaseType if ct.name != "Custom"]),
            is_eager=True,
            default="Performance1536D50K",
            help="Case type",
        ),
    ]
    db_label: Annotated[
        str,
        click.option(
            "--db-label", type=str, help="Db label, default: date in ISO format",
            show_default=True,
            default=datetime.now().isoformat()
        ),
    ]
    dry_run: Annotated[
        bool,
        click.option(
            "--dry-run",
            type=bool,
            default=False,
            is_flag=True,
            help="Print just the configuration and exit without running the tasks",
        ),
    ]
    k: Annotated[
        int,
        click.option(
            "--k",
            type=int,
            default=config.K_DEFAULT,
            show_default=True,
            help="K value for number of nearest neighbors to search",
        ),
    ]
    concurrency_duration: Annotated[
        int,
        click.option(
            "--concurrency-duration",
            type=int,
            default=config.CONCURRENCY_DURATION,
            show_default=True,
            help="Adjusts the duration in seconds of each concurrency search",
        ),
    ]
    num_concurrency: Annotated[
        List[str],
        click.option(
            "--num-concurrency",
            type=str,
            help="Comma-separated list of concurrency values to test during concurrent search",
            show_default=True,
            default=",".join(map(str, config.NUM_CONCURRENCY)),
            callback=lambda *args: list(map(int, click_arg_split(*args))),
        ),
    ]
    custom_case_name: Annotated[
        str,
        click.option(
            "--custom-case-name",
            help="Custom dataset case name",
            callback=check_custom_case_parameters,
        )
    ]
    custom_case_description: Annotated[
        str,
        click.option(
            "--custom-case-description",
            help="Custom dataset case description",
            default="This is a customized dataset.",
            show_default=True,
        )
    ]
    custom_case_load_timeout: Annotated[
        int,
        click.option(
            "--custom-case-load-timeout",
            help="Custom dataset case load timeout",
            default=36000,
            show_default=True,
        )
    ]
    custom_case_optimize_timeout: Annotated[
        int,
        click.option(
            "--custom-case-optimize-timeout",
            help="Custom dataset case optimize timeout",
            default=36000,
            show_default=True,
        )
    ]
    custom_dataset_name: Annotated[
        str,
        click.option(
            "--custom-dataset-name",
            help="Custom dataset name",
            callback=check_custom_case_parameters,
        ),
    ]
    custom_dataset_dir: Annotated[
        str,
        click.option(
            "--custom-dataset-dir",
            help="Custom dataset directory",
            callback=check_custom_case_parameters,
        ),
    ]
    custom_dataset_size: Annotated[
        int,
        click.option(
            "--custom-dataset-size",
            help="Custom dataset size",
            callback=check_custom_case_parameters,
        ),
    ]
    custom_dataset_dim: Annotated[
        int,
        click.option(
            "--custom-dataset-dim",
            help="Custom dataset dimension",
            callback=check_custom_case_parameters,
        ),
    ]
    custom_dataset_metric_type: Annotated[
        str,
        click.option(
            "--custom-dataset-metric-type",
            help="Custom dataset metric type",
            default=MetricType.COSINE.name,
            show_default=True,
        ),
    ]
    custom_dataset_file_count: Annotated[
        int,
        click.option(
            "--custom-dataset-file-count",
            help="Custom dataset file count",
            callback=check_custom_case_parameters,
        ),
    ]
    custom_dataset_use_shuffled: Annotated[
        bool,
        click.option(
            "--custom-dataset-use-shuffled/--skip-custom-dataset-use-shuffled",
            help="Custom dataset use shuffled",
            default=False,
            show_default=True,
        ),
    ]
    custom_dataset_with_gt: Annotated[
        bool,
        click.option(
            "--custom-dataset-with-gt/--skip-custom-dataset-with-gt",
            help="Custom dataset with ground truth",
            default=True,
            show_default=True,
        ),
    ]


class HNSWBaseTypedDict(TypedDict):
    m: Annotated[Optional[int], click.option("--m", type=int, help="hnsw m")]
    ef_construction: Annotated[
        Optional[int],
        click.option("--ef-construction", type=int, help="hnsw ef-construction"),
    ]


class HNSWBaseRequiredTypedDict(TypedDict):
    m: Annotated[Optional[int], click.option("--m", type=int, help="hnsw m", required=True)]
    ef_construction: Annotated[
        Optional[int],
        click.option("--ef-construction", type=int, help="hnsw ef-construction", required=True),
    ]


class HNSWFlavor1(HNSWBaseTypedDict):
    ef_search: Annotated[
        Optional[int], click.option("--ef-search", type=int, help="hnsw ef-search", is_eager=True)
    ]


class HNSWFlavor2(HNSWBaseTypedDict):
    ef_runtime: Annotated[
        Optional[int], click.option("--ef-runtime", type=int, help="hnsw ef-runtime")
    ]


class HNSWFlavor3(HNSWBaseRequiredTypedDict):
    ef_search: Annotated[
        Optional[int], click.option("--ef-search", type=int, help="hnsw ef-search", required=True)
    ]


class IVFFlatTypedDict(TypedDict):
    lists: Annotated[
        Optional[int], click.option("--lists", type=int, help="ivfflat lists")
    ]
    probes: Annotated[
        Optional[int], click.option("--probes", type=int, help="ivfflat probes")
    ]


class IVFFlatTypedDictN(TypedDict):
    nlist: Annotated[
        Optional[int], click.option("--lists", "nlist", type=int, help="ivfflat lists", required=True)
    ]
    nprobe: Annotated[
        Optional[int], click.option("--probes", "nprobe", type=int, help="ivfflat probes", required=True)
    ]


@click.group()
def cli():
    ...


def run(
    db: DB,
    db_config: DBConfig,
    db_case_config: DBCaseConfig,
    **parameters: Unpack[CommonTypedDict],
):
    """Builds a single VectorDBBench Task and runs it, awaiting the task until finished.

    Args:
        db (DB)
        db_config (DBConfig)
        db_case_config (DBCaseConfig)
        **parameters: expects keys from CommonTypedDict
    """

    task = TaskConfig(
        db=db,
        db_config=db_config,
        db_case_config=db_case_config,
        case_config=CaseConfig(
            case_id=CaseType[parameters["case_type"]],
            k=parameters["k"],
            concurrency_search_config=ConcurrencySearchConfig(
                concurrency_duration=parameters["concurrency_duration"],
                num_concurrency=[int(s) for s in parameters["num_concurrency"]],
            ),
            custom_case=get_custom_case_config(parameters),
        ),
        stages=parse_task_stages(
            (
                False if not parameters["load"] else parameters["drop_old"]
            ),  # only drop old data if loading new data
            parameters["load"],
            parameters["search_serial"],
            parameters["search_concurrent"],
        ),
    )

    log.info(f"Task:\n{pformat(task)}\n")
    if not parameters["dry_run"]:
        benchMarkRunner.run([task])
        time.sleep(5)
        if global_result_future:
            wait([global_result_future])
