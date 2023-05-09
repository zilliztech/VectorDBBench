import logging
from logging import config, Filter
import os
import datetime
from pytz import timezone


def init(log_level, log_path, name, tz='UTC'):
    def build_log_file(level, log_path, name, tz):
        utc_now = datetime.datetime.utcnow()
        utc_tz = timezone('UTC')
        local_tz = timezone(tz)
        tznow = utc_now.replace(tzinfo=utc_tz).astimezone(local_tz)
        return '{}-{}-{}.log'.format(os.path.join(log_path, name), tznow.strftime("%m-%d-%Y-%H:%M:%S"),
                                     level)

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    LOGGING = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'default': {
                'format': '%(asctime)s | %(levelname)s | %(name)s | %(threadName)s: %(message)s (%(filename)s:%(lineno)s)',
            },
            'colorful_console': {
                'format': '%(asctime)s | %(levelname)s: %(message)s (%(filename)s:%(lineno)s) (%(threadName)s)',
                '()': ColorfulFormatter,
            },
        },
        'filters': {
            'InfoFilter': {
                '()': InfoFilter,
            },
            'DebugFilter': {
                '()': DebugFilter,
            },
            'WarnFilter': {
                '()': WarnFilter,
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'colorful_console',
            },
            'debug_file': {
                'level': 'DEBUG',
                'filters': ['DebugFilter'],
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'default',
                'filename': build_log_file('debug', log_path, name, tz)
            },
            'info_file': {
                'level': 'INFO',
                'filters': ['InfoFilter'],
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'default',
                'filename': build_log_file('info', log_path, name, tz)
            },
            'warn_file': {
                'level': 'WARN',
                'filters': ['WarnFilter'],
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'default',
                'filename': build_log_file('warn', log_path, name, tz)
            },
        },
        'loggers': {
            '': {
                #  'handlers': ['console', 'info_file', 'debug_file', 'warn_file'],
                'handlers': ['console'],
                'level': log_level,
                'propagate': False
            },
        },
        'propagate': False,
    }

    config.dictConfig(LOGGING)

class colors:
    HEADER= '\033[95m'
    INFO= '\033[92m'
    DEBUG= '\033[94m'
    WARNING= '\033[93m'
    ERROR= '\033[95m'
    CRITICAL= '\033[91m'
    ENDC= '\033[0m'



COLORS = {
    'INFO': colors.INFO,
    'INFOM': colors.INFO,
    'DEBUG': colors.DEBUG,
    'DEBUGM': colors.DEBUG,
    'WARNING': colors.WARNING,
    'WARNINGM': colors.WARNING,
    'ENDC': colors.ENDC,
}


class ColorFulFormatColMixin:
    def format_col(self, message_str, level_name):
        if level_name in COLORS.keys():
            message_str = COLORS[level_name] + message_str + COLORS['ENDC']
        return message_str

    def formatTime(self, record, datefmt=None):
        ret = super().formatTime(record, datefmt)
        #  ret = COLORS['ASCTIME'] + ret + COLORS['ENDC']
        return ret


class ColorfulLogRecordProxy(logging.LogRecord):
    def __init__(self, record):
        self._record = record
        msg_level = record.levelname + 'M'
        self.msg = f"{COLORS[msg_level]}{record.msg}{COLORS['ENDC']}"
        self.filename = record.filename
        self.lineno = f'{record.lineno}'
        self.threadName = f'{record.threadName}'
        self.levelname = f"{COLORS[record.levelname]}{record.levelname}{COLORS['ENDC']}"
        #  self.filename = COLORS['FILENAME'] + record.filename + COLORS['ENDC']
        #  self.lineno = '{}{}{}'.format(COLORS['LINENO'], record.lineno, COLORS['ENDC'])
        #  self.threadName = '{}{}{}'.format(COLORS['THREAD'], record.threadName, COLORS['ENDC'])
        #  self.levelname = COLORS[record.levelname] + record.levelname + COLORS['ENDC']

    def __getattr__(self, attr):
        if attr not in self.__dict__:
            return getattr(self._record, attr)
        return getattr(self, attr)


class ColorfulFormatter(ColorFulFormatColMixin, logging.Formatter):
    def format(self, record):
        proxy = ColorfulLogRecordProxy(record)
        message_str = super().format(proxy)

        return message_str

class InfoFilter(Filter):
    def filter(self, rec):
        return rec.levelno == logging.INFO


class DebugFilter(Filter):
    def filter(self, rec):
        return rec.levelno == logging.DEBUG


class WarnFilter(Filter):
    def filter(self, rec):
        return rec.levelno == logging.WARN
