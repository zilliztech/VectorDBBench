from functools import partial
from typing import Callable


def st_command_with_default(st_command: Callable, kwarg_defaults: dict, **kwargs):
    for kwarg, default in kwarg_defaults.items():
        kwargs[kwarg] = kwargs.get(kwarg, default)
    return partial(st_command, **kwargs)()
