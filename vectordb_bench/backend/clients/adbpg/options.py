from typing import Any

import click


def _normalize_config_items(values: tuple[str, ...]) -> list[str]:
    """Normalize repeatable CLI/YAML values without changing value contents."""
    return [item for value in values if (item := str(value).strip())]


def parse_key_values(_ctx: Any, _param: Any, values: tuple[str, ...]) -> dict[str, str]:
    """Parse repeatable ``name=value`` settings; the last value wins."""
    parsed: dict[str, str] = {}
    for item in _normalize_config_items(values):
        if "=" not in item:
            message = f"Expected name=value, got: {item}"
            raise click.BadParameter(message)
        name, value = (part.strip() for part in item.split("=", 1))
        if not name:
            message = f"Empty setting name in: {item}"
            raise click.BadParameter(message)
        parsed[name] = value
    return parsed


def parse_reloptions(_ctx: Any, _param: Any, values: tuple[str, ...]) -> dict[str, str | None]:
    """Parse reloptions; a bare name means ``ALTER INDEX ... RESET``."""
    parsed: dict[str, str | None] = {}
    for item in _normalize_config_items(values):
        name, separator, value = item.partition("=")
        name = name.strip()
        if not name:
            message = f"Empty reloption name in: {item}"
            raise click.BadParameter(message)
        parsed[name] = value.strip() if separator else None
    return parsed
