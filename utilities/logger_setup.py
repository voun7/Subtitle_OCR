import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def get_console_handler() -> logging.handlers:
    """
    The console sends only messages by default no need for formatter.
    """
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    return console_handler


def get_file_handler(log_dir: Path, log_format: logging.Formatter) -> logging.handlers:
    log_file = log_dir / "runtime.log"
    file_handler = RotatingFileHandler(log_file, backupCount=7, encoding='utf-8', delay=True)
    file_handler.namer = my_namer
    if log_file.exists():
        file_handler.doRollover()
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_format)
    return file_handler


def reset_handlers(logger) -> None:
    """
    Remove all handlers from the root logger.
    This helps prevent duplicate logs from other handlers created by imported modules.
    """
    for handler in logger.handlers:
        logger.removeHandler(handler)


def setup_logging(name: str = "") -> None:
    """
    Use the following to add logger to other modules.
    import logging
    logger = logging.getLogger(__name__)

    The following suppress log messages. It will not log messages of given module unless they are at least warnings.
    logging.getLogger("module_name").setLevel(logging.WARNING)
    """
    # Create folder for file logs.
    log_dir = Path(__file__).parent.parent / f"logs{f'/{name}' if name else name}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create a custom logger.
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Better to have too much log than not enough.
    reset_handlers(logger)

    # Create formatters and add it to handlers.
    # logfmt_str1 = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logfmt_str2 = "%(asctime)s %(levelname)-8s pid:%(process)d %(name)s:%(lineno)03d:%(funcName)s %(message)s"
    log_format = logging.Formatter(logfmt_str2)

    # Add handlers to the logger.
    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler(log_dir, log_format))


def my_namer(default_name: str) -> str:
    """
    This will be called when doing the log rotation
    default_name is the default filename that would be assigned, e.g. Rotate_Test.txt.YYYY-MM-DD
    Do any manipulations to that name here, for example this function changes the name to Rotate_Test.YYYY-MM-DD.txt
    """
    base_filename, ext, date = default_name.split(".")
    return f"{base_filename}.{date}.{ext}"
