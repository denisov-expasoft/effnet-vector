__all__ = [
    'turn_logging_on',
    'turn_logging_off',
]

import logging.config
from pathlib import Path

_DIR = Path(__file__).parent
_ACTIVE_LOGGING_CONFIG = _DIR.joinpath('data', 'logging.conf').as_posix()
_MUTE_LOGGING_CONFIG = _DIR.joinpath('data', 'logging_silent.conf').as_posix()


def turn_logging_on():
    logging.config.fileConfig(_ACTIVE_LOGGING_CONFIG)


def turn_logging_off():
    logging.config.fileConfig(_MUTE_LOGGING_CONFIG)
