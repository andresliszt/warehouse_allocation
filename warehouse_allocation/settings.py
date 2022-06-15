# -*- coding: utf-8 -*-
"""Settings del proyecto."""
import logging
from enum import Enum
from enum import IntEnum
from pathlib import Path
from typing import Optional

from dotenv import find_dotenv
from dotenv import load_dotenv
from pydantic import BaseSettings


def init_dotenv():
    """Localiza y carga el archivo `.env`."""

    candidate = find_dotenv(usecwd=True)

    if not candidate:
        # raise IOError("No se encuentra el archivo `.env`.")
        return

    load_dotenv(candidate)


class LogLevel(IntEnum):
    """Logs levels."""

    CRITICAL = logging.CRITICAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    TRACE = 1 + logging.NOTSET
    NOTSET = logging.NOTSET


class LogDest(Enum):
    """Destinos de los logs."""

    CONSOLE = "CONSOLE"
    """Log en consola"""

    FILE = "FILE"
    """Log en archivo"""


class LogFormatter(Enum):
    """Formatos de los logs."""

    JSON = "JSON"
    """JSONs, para máquinas por ejemplo."""

    COLOR = "COLOR"
    """pprinted, colored, para humanos por ejemplo (o perritos)"""


class Settings(BaseSettings):
    """Settings comunes."""

    PACKAGE_PATH = Path(__file__).parent

    PROJECT_PATH = PACKAGE_PATH.parent

    LOG_PATH: Optional[Path]

    LOG_FORMAT: LogFormatter = LogFormatter.JSON.value

    LOG_LEVEL: LogLevel = LogLevel.INFO.value

    LOG_DESTINATION: LogDest = LogDest.CONSOLE.value

    class Config:
        """Configuración interna."""

        env_prefix = "warehouse_allocation_"
        use_enum_values = True


def init_project_settings():
    """Retorna settings y carga env vars."""
    init_dotenv()
    return Settings()
