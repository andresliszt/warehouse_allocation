# -*- coding: utf-8 -*-
"""Conftest models."""

import pickle
from pathlib import Path

import pytest

from warehouse_allocation import SETTINGS

ALGORITM_DATA = []

for file in Path(
    SETTINGS.PROJECT_PATH, "tests", "algorithms", "data"
).iterdir():

    with open(file, "rb") as inp:
        ALGORITM_DATA.append(pickle.load(inp))


@pytest.fixture(scope="module", params=ALGORITM_DATA)
def algorithm_data(request):
    yield request.param
