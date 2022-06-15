# -*- coding: utf-8 -*-
"""Unit test para problemas de asignación lineal."""


import pytest
import numpy as np

from warehouse_allocation.operators.lap import LAPWithCostFlow
from warehouse_allocation.operators.lap import (
    LAPWithCostFlowRestrictedClusters,
)
from warehouse_allocation.operators.lap import (
    LAPWithCostFlowRestrictedClustersPlusDivision,
)


LAP_PROBLEMS_TEST = [
    (
        LAPWithCostFlow(
            Cost=-np.array([[1, 2, 3], [4, 5, 6]]), Z=np.array([2, 2])
        ),
        np.array(
            [
                [True, False, False, False, True, True],
                [False, True, False, True, False, True],
                [False, False, True, True, True, False],
            ]
        ),
    ),
    (
        LAPWithCostFlow(
            Cost=-np.array([[1, 2, 3, 4], [5, 6, 9, 10]]), Z=np.array([2, 2])
        ),
        np.array([True, True, False, False, False, False, True, True]),
    ),
    (
        LAPWithCostFlow(
            Cost=-np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
            Z=np.array([2, 2, 2]),
        ),
        np.array(
            [
                [
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                ],
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                    True,
                    False,
                    False,
                ],
            ],
        ),
    ),
    (
        LAPWithCostFlowRestrictedClusters(
            Cost=-np.array([[1, 2, 3], [4, 5, 6]]),
            Z=np.array([2, 2]),
            allowed_clusters_by_sku={0: [0, 1], 1: [1], 2: [1]},
        ),
        np.array([True, False, False, False, True, True]),
    ),
    (
        LAPWithCostFlowRestrictedClusters(
            Cost=-np.array([[1, 2, 3, 4], [5, 6, 7, -8]]),
            Z=np.array([3, 3]),
            allowed_clusters_by_sku={0: [0], 1: [0, 1], 2: [1], 3: [0, 1]},
        ),
        np.array([True, False, False, True, False, True, True, False]),
    ),
    (
        LAPWithCostFlowRestrictedClusters(
            Cost=-np.array([[-1, 2, 3, 4], [5, 6, -7, 8]]),
            Z=np.array([2, 2]),
            allowed_clusters_by_sku={
                0: [0, 1],
                1: [0, 1],
                2: [0, 1],
                3: [0, 1],
            },
        ),
        np.array(
            [
                [False, False, True, True, True, True, False, False],
                [False, True, True, False, True, False, False, True],
            ]
        ),
    ),
    (
        LAPWithCostFlowRestrictedClusters(
            Cost=-np.array([[100, 2, 3, 4], [5, 6, 7, 100], [9, 40, 20, 12]]),
            Z=np.array([2, 2, 2]),
            allowed_clusters_by_sku={
                0: [2],
                1: [1, 2],
                2: [0, 1, 2],
                3: [0, 1, 2],
            },
        ),
        np.array(
            [
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                False,
                False,
            ]
        ),
    ),
    (
        LAPWithCostFlowRestrictedClusters(
            Cost=-np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
            Z=np.array([2, 2, 2]),
            allowed_clusters_by_sku={0: [0], 1: [0], 2: [0, 1, 2], 3: [1]},
        ),
        np.array(
            [
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                True,
                False,
            ]
        ),
    ),
    (
        LAPWithCostFlowRestrictedClustersPlusDivision(
            Cost=-np.array(
                [[10, 20, 0, 0], [5, 6, 70, 80], [9, 10, 110, 120]]
            ),
            Z=np.array([[2, 2], [2, 0], [2, 0]]),
            division_types=np.array([0, 0, 1, 1]),
            allowed_clusters_by_sku={
                0: [0, 1, 2],
                1: [0, 1, 2],
                2: [0, 1, 2],
                3: [0, 1, 2],
            },
        ),
        np.array(
            [
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ]
        ),
    ),
    (
        LAPWithCostFlowRestrictedClustersPlusDivision(
            Cost=-np.array([[1, 2, 3, 400], [5, 6, 7, 20], [9, 10, 11, 12]]),
            Z=np.array([[3, 0], [1, 1], [1, 1]]),
            division_types=np.array([0, 0, 0, 1]),
            allowed_clusters_by_sku={
                0: [0],
                1: [0],
                2: [0, 1, 2],
                3: [0, 1, 2],
            },
        ),
        np.array(
            [
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                True,
                False,
            ]
        ),
    ),
]


@pytest.mark.parametrize("lap_input, slotting", LAP_PROBLEMS_TEST)
def test_laps_problems(lap_input, slotting):

    individual = lap_input.solve()

    if slotting.ndim > 1:
        assert np.any(np.all(individual.flatten() == slotting, axis=1))
    else:
        assert np.all(individual.flatten() == slotting)
