import logging
import subprocess
import os

from . import config

LOG_LEVEL = "DEBUG"
LOG_PATH = '/tmp/falcon_mark'
LOG_NAME = 'logfile'
TIMEZONE = 'UTC'

config.init(LOG_LEVEL, LOG_PATH, LOG_NAME, TIMEZONE)

log = logging.getLogger("main")

# TODO: logging configs before start
def main():
    log.debug("Debug log message")
    log.info("Info log message")
    log.warning("Warining log message")

    run_streamlit()

def run_streamlit():
    # TODO: add log level
    cmd = ['streamlit', 'run', f'{os.path.dirname(__file__)}/frontend/run_test.py']
    log.debug(f"cmd: {cmd}")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        log.info("exit streamlit...")
    except BaseException as e:
        log.info(f"exit, err={e.__class__}\nstack trace={e.with_traceback()}")


if __name__ == "__main__":
    main()
