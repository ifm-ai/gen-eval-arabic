"""Colored logger setup for the evaluation scripts."""
import logging
import sys

from configs import LOGGER_LEVEL

def red(text: str) -> str: return f"\033[1;31m{text}\033[0m"
def green(text: str) -> str: return f"\033[1;32m{text}\033[0m"
def yellow(text: str) -> str: return f"\033[1;33m{text}\033[0m"
def blue(text: str) -> str: return f"\033[1;34m{text}\033[0m"
def white(text: str) -> str: return f"\033[1;37m{text}\033[0m"


LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(funcName)s:%(lineno)d | %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


class _ColorFormatter(logging.Formatter):
    _COLORS = {
        logging.DEBUG: white,
        logging.INFO: blue,
        logging.WARNING: yellow,
        logging.ERROR: red,
        logging.CRITICAL: red,
    }

    def format(self, record):
        message = super().format(record)
        colorize = self._COLORS.get(record.levelno)
        return colorize(message) if colorize else message


def _setup_logger(name: str = "jais3_evals") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_ColorFormatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
        log.addHandler(handler)
        log.setLevel(getattr(logging, LOGGER_LEVEL.upper(), logging.DEBUG))
        log.propagate = False
    return log


logger = _setup_logger()
