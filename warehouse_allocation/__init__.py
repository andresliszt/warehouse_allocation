# -*- coding: utf-8 -*-
"""Inicialización del paquete."""


from pymoo.util.function_loader import is_compiled

from warehouse_allocation._logging import configure_logging
from warehouse_allocation.settings import init_project_settings

SETTINGS = init_project_settings()

logger = configure_logging("warehouse_allocation", SETTINGS, kidnap_loggers=True)


if not is_compiled():
    logger.warning(
        "Versión no compilada de ``pymoo``. La performance se verá afectada",
        see_also="https://pymoo.org/installation.html",
    )

__version__ = "1.0.0"
