import logging
from logging import config
from pathlib import Path


def init(log_level: str):
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s | %(levelname)s |%(message)s (%(filename)s:%(lineno)s)",
            },
            "colorful_console": {
                "format": "%(asctime)s | %(levelname)s: %(message)s (%(filename)s:%(lineno)s) (%(process)s)",
                "()": ColorfulFormatter,
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "colorful_console",
            },
            "no_color_console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "default",
                "filename": "logs/vectordb_bench.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf8",
            },
        },
        "loggers": {
            "vectordb_bench": {
                "handlers": ["console", "file"],
                "level": log_level,
                "propagate": False,
            },
            "no_color": {
                "handlers": ["no_color_console", "file"],
                "level": log_level,
                "propagate": False,
            },
        },
        "propagate": False,
    }

    config.dictConfig(log_config)


class colors:
    HEADER = "\033[95m"
    INFO = "\033[92m"
    DEBUG = "\033[94m"
    WARNING = "\033[93m"
    ERROR = "\033[95m"
    CRITICAL = "\033[91m"
    ENDC = "\033[0m"


COLORS = {
    "INFO": colors.INFO,
    "INFOM": colors.INFO,
    "DEBUG": colors.DEBUG,
    "DEBUGM": colors.DEBUG,
    "WARNING": colors.WARNING,
    "WARNINGM": colors.WARNING,
    "CRITICAL": colors.CRITICAL,
    "CRITICALM": colors.CRITICAL,
    "ERROR": colors.ERROR,
    "ERRORM": colors.ERROR,
    "ENDC": colors.ENDC,
}


class ColorFulFormatColMixin:
    def format_col(self, message: str, level_name: str):
        if level_name in COLORS:
            message = COLORS[level_name] + message + COLORS["ENDC"]
        return message


class ColorfulLogRecordProxy(logging.LogRecord):
    def __init__(self, record: any):
        self._record = record
        msg_level = record.levelname + "M"
        self.msg = f"{COLORS[msg_level]}{record.msg}{COLORS['ENDC']}"
        self.filename = record.filename
        self.lineno = f"{record.lineno}"
        self.process = f"{record.process}"
        self.levelname = f"{COLORS[record.levelname]}{record.levelname}{COLORS['ENDC']}"

    def __getattr__(self, attr: any):
        if attr not in self.__dict__:
            return getattr(self._record, attr)
        return getattr(self, attr)


class ColorfulFormatter(ColorFulFormatColMixin, logging.Formatter):
    def format(self, record: any):
        proxy = ColorfulLogRecordProxy(record)
        return super().format(proxy)
