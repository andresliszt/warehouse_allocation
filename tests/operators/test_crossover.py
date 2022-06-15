# -*- coding: utf-8 -*-
"""Pruebas unitarias para ``warehouse_allocation.models.operators.crossover``"""

from collections import namedtuple

import pytest
import numpy as np

from warehouse_allocation.operators import (
    ChungPartiallyMappedCrossoverWeigthed,
)


CROSSOVER = ChungPartiallyMappedCrossoverWeigthed()

TRAFFIC_CROSSOVER = ChungPartiallyMappedCrossoverWeigthed(aff_prob=0)

AFFINITY_CROSSOVER = ChungPartiallyMappedCrossoverWeigthed(aff_prob=1)

InputReassignment = namedtuple(
    "InputReassignment",
    ["individuals", "n_clusters", "OCM", "D", "clusters_penalizations"],
)

ExpectedCostReassignment = namedtuple(
    "ExpectedCostReassignment", ["AFF_COST", "TRAFF_COST"]
)


TO_REASSIGNMENT = [
    InputReassignment(
        individuals=np.array(
            [
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            ]
        ),
        n_clusters=4,
        OCM=np.array(
            [
                [0, 1, 10, 100],
                [1, 0, 20, 200],
                [10, 20, 0, 300],
                [100, 200, 300, 0],
            ]
        ),
        D=np.array([100, 200, 300, 400]),
        clusters_penalizations=np.array([1, 1, 1, 1]),
    ),
    InputReassignment(
        individuals=np.array(
            [
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            ]
        ),
        n_clusters=4,
        OCM=np.array(
            [
                [0, 1, 10, 100],
                [1, 0, 20, 200],
                [10, 20, 0, 300],
                [100, 200, 300, 0],
            ]
        ),
        D=np.array([100, 200, 300, 400]),
        clusters_penalizations=np.array([0.5, 0.1, 1, 0.1]),
    ),
]

EXPECTED_COST_REASSIGNMENT = [
    ExpectedCostReassignment(
        AFF_COST=[
            np.array([[-10, -100], [-20, -200], [0, 0], [0, 0]]),
            np.array([]),
            np.array([[0], [-100], [-500], [0]]),
            np.array([]),
            np.array([]),
            np.array([[0], [-600], [0], [0]]),
            np.array([[-200], [-20], [-1], [0]]),
        ],
        TRAFF_COST=[
            np.array([[400, 500], [500, 600], [300, 400], [300, 400]]),
            np.array([]),
            np.array([[400], [500], [900], [400]]),
            np.array([]),
            np.array([]),
            np.array([[400], [1000], [400], [400]]),
            np.array([[600], [500], [300], [200]]),
        ],
    ),
    ExpectedCostReassignment(
        AFF_COST=[
            np.array([[-5, -50], [-2, -20], [0, 0], [0, 0]]),
            np.array([]),
            np.array([[0], [-10], [-500], [0]]),
            np.array([]),
            np.array([]),
            np.array([[0], [-60], [0], [0]]),
            np.array([[-100], [-2], [-1], [0]]),
        ],
        TRAFF_COST=[
            np.array([[400, 500], [500, 600], [300, 400], [300, 400]]),
            np.array([]),
            np.array([[400], [500], [900], [400]]),
            np.array([]),
            np.array([]),
            np.array([[400], [1000], [400], [400]]),
            np.array([[600], [500], [300], [200]]),
        ],
    ),
]


@pytest.mark.parametrize(
    "test_input,expected", zip(TO_REASSIGNMENT, EXPECTED_COST_REASSIGNMENT)
)
def test_affinity_and_demand_cost_matrix_in_crossover_mapping(
    test_input, expected
):

    for ind, C_AFF_EXPECTED, C_TRAFF_EXPECTED in zip(
        test_input.individuals, expected.AFF_COST, expected.TRAFF_COST
    ):
        ind = ind.reshape(
            (test_input.n_clusters, int(len(ind) / test_input.n_clusters))
        )
        RE = np.where(~ind.any(axis=0))[0]
        C_AFF = CROSSOVER.affinity_cost(
            ind, RE, test_input.OCM, test_input.clusters_penalizations
        )
        C_TRAFF = CROSSOVER.traffic_cost(ind, RE, test_input.D)

        assert (C_AFF == C_AFF_EXPECTED).all()

        assert (C_TRAFF == C_TRAFF_EXPECTED).all()
