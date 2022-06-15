from collections import namedtuple


import pytest
import numpy as np

from warehouse_allocation.models.chung import ChungProblemStrictlyWeighted


InputMatingTest = namedtuple("InputMatingTest", ["problem", "X_pop"])
"""Inputs para los operadores mating"""


PROBLEMS = [
    InputMatingTest(
        problem=ChungProblemStrictlyWeighted(
            D=np.array([1, 1]),
            Z=np.array([2, 2]),
            OCM=np.array([[0, 1], [1, 0]]),
            W=np.array([1, 1]),
            WT=[(0, 1), (0, 1)],
        ),
        X_pop=np.array(
            [
                [1, 0, 0, 1],
                [0, 1, 1, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
            ]
        ),
    ),
    InputMatingTest(
        problem=ChungProblemStrictlyWeighted(
            D=np.array([1, 1, 1]),
            Z=np.array([3, 3, 3]),
            OCM=np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
            W=np.array([1, 1, 1]),
            WT=[(0, 1), (0, 1), (0, 1)],
        ),
        X_pop=np.array(
            [
                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1, 1],
            ]
        ),
    ),
]


@pytest.fixture(scope="module")
def weighted_problems():
    return PROBLEMS
