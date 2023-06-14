import traceback
import logging
import subprocess
import os
from . import config

log = logging.getLogger("vectordb_bench")


def main():
    log.info(f"all configs: {config().display()}")
    run_streamlit()


def run_streamlit():
    cmd = [
        "streamlit",
        "run",
        f"{os.path.dirname(__file__)}/frontend/vdb_benchmark.py",
        "--logger.level",
        "info",
        "--theme.base",
        "light",
        "--theme.primaryColor",
        "#3670F2",
        "--theme.secondaryBackgroundColor",
        "#F0F2F6",
    ]
    log.debug(f"cmd: {cmd}")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        log.info("exit streamlit...")
    except Exception as e:
        log.warning(f"exit, err={e}\nstack trace={traceback.format_exc(chain=True)}")


if __name__ == "__main__":
    main()
