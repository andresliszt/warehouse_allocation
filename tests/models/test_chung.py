# -*- coding: utf-8 -*-
"""Pruebas unitarias para ``warehouse_allocation.models.chung``"""

from collections import namedtuple

import pytest
import numpy as np

from warehouse_allocation.exc import InvalidOcurrenceMatrixz
from warehouse_allocation.exc import NonPositiveParameterError
from warehouse_allocation.exc import SkuAndOCMDimensionError
from warehouse_allocation.exc import SkusAndWeightDimensionError
from warehouse_allocation.exc import ClusterAndWeightToleranceDimensionError
from warehouse_allocation.exc import NoCapacityForSkusError
from warehouse_allocation.exc import LAPNoFeasibleSolutionError
from warehouse_allocation.exc import NoParametersError

from warehouse_allocation.models.chung import ChungProblem, ChungProblemWithDivisions
from warehouse_allocation.models.chung import ChungProblemWithDivisionsPlusWeight
from warehouse_allocation.models.chung import ChungProblemStrictlyWeighted


InputChungProblemTest = namedtuple("InputChungProblem", ["problem", "X_pop"])


ExpectedChungProblemTest = namedtuple(
    "ExpectedChungProblem",
    [
        "F",
        "G",
    ],
)


PROBLEMS = [
    InputChungProblemTest(
        problem=ChungProblem(
            D=np.array([10, 10]),
            Z=np.array([2, 2]),
            OCM=np.array([[0, 2], [2, 0]]),
        ),
        X_pop=np.array(
            [
                [1, 1, 1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [1, 0, 1, 0],
                [1, 1, 1, 1],
            ]
        ),
    ),
    InputChungProblemTest(
        problem=ChungProblem(
            D=np.array([5, 10, 15]),
            Z=np.array([2, 1]),
            OCM=np.array([[0, 2, 3], [2, 0, 5], [3, 5, 0]]),
        ),
        X_pop=np.array(
            [
                [1, 1, 1, 0, 0, 0],
                [1, 0, 0, 0, 1, 1],
                [1, 1, 0, 0, 0, 1],
                [1, 0, 1, 0, 1, 0],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0],
            ]
        ),
    ),
    InputChungProblemTest(
        problem=ChungProblemStrictlyWeighted(
            D=np.array([5, 10, 15]),
            Z=np.array([2, 1]),
            OCM=np.array([[0, 2, 3], [0, 0, 5], [0, 0, 0]]),
            W=np.array([1, 5, 10]),
            WT=np.array([[0, 10], [5, 10]]),
        ),
        X_pop=np.array(
            [
                [1, 1, 1, 0, 0, 0],
                [1, 0, 0, 0, 1, 1],
                [1, 1, 0, 0, 0, 1],
                [1, 0, 1, 0, 1, 0],
                [1, 1, 1, 1, 1, 1],  # [1, 2, 1, 1, 1, 0, 1, 1]
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0, 0],  # [1, 0, 1, 0, 0, 0, 0, 0]
                [0, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0],
            ]
        ),
    ),
    InputChungProblemTest(
        problem=ChungProblemWithDivisions(
            D=np.array([5, 10, 15]),
            Z=np.array([[2, 1], [1, 1]]),
            OCM=np.array([[0, 0, 0], [2, 0, 0], [3, 5, 0]]),
            division_types=np.array([0, 0, 1]),
        ),
        X_pop=np.array(
            [
                [1, 1, 1, 0, 0, 0],
                [1, 0, 0, 0, 1, 1],
                [1, 1, 0, 0, 0, 1],
                [1, 0, 1, 0, 1, 0],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0],
            ]
        ),
    ),
]


# Lista de tuplas, donde cada tupla es de la siguiente forma:
# Primera entrada corresponde al problema, y las siguientes entradas
# (de largo arbritriario) son individuos en su forma matricial, definidos
# para dicho problema

EXPECTED_FOR_PROBLEM = [
    ExpectedChungProblemTest(
        F=np.array([[-2, 20], [0, 10], [0, 0], [0, 10], [0, 10], [-4, 20]]),
        G=np.array(
            [
                [0, -1, 1, 0],
                [-2, -1, 1, 0],
                [-2, -2, 1, 1],
                [-2, -1, 0, 1],
                [-1, -1, 1, 1],
                [0, 0, 1, 1],
            ]
        ),
    ),
    ExpectedChungProblemTest(
        F=np.array(
            [
                [-10, 30],
                [-5, 25],
                [-2, 15],
                [-3, 20],
                [-20, 30],
                [-12, 30],
                [-10, 30],
                [-5, 25],
                [-2, 15],
                [-10, 30],
                [0, 0],
            ]
        ),
        G=np.array(
            [
                [1, -1, 0, 0, 0],
                [-1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 2, 1, 1, 1],
                [1, 1, 1, 1, 0],
                [1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [-1, 1, 0, 0, 0],
                [-2, 2, 0, 0, 0],
                [-2, -1, 1, 1, 1],
            ]
        ),
    ),
    ExpectedChungProblemTest(
        F=np.array(
            [
                [-10, 30],
                [-5, 25],
                [-2, 15],
                [-3, 20],
                [-20, 30],
                [-12, 30],
                [-10, 30],
                [-5, 25],
                [-2, 15],
                [-10, 30],
                [0, 0],
            ]
        ),
        G=np.array(
            [
                [1, -1, 0, 0, 0, 0, 0, 0],
                [-1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 2, 1, 1, 1, 0, 1, 1],
                [1, 1, 1, 1, 0, 0, 1, 0],
                [1, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [-1, 1, 0, 0, 0, 1, 0, 0],
                [-2, 2, 0, 0, 0, 1, 0, 0],
                [-2, -1, 1, 1, 1, 1, 1, 1],
            ]
        ),
    ),
    ExpectedChungProblemTest(
        F=np.array(
            [
                [-10, 30],
                [-5, 25],
                [-2, 15],
                [-3, 20],
                [-20, 30],
                [-12, 30],
                [-10, 30],
                [-5, 25],
                [-2, 15],
                [-10, 30],
                [0, 0],
            ]
        ),
        G=np.array(
            [
                [0, -1, 0, -1, 0, 0, 0],
                [-1, 0, -1, 0, 0, 0, 0],
                [0, -1, -1, 0, 0, 0, 0],
                [-1, 0, 0, -1, 0, 0, 0],
                [0, 1, 0, 0, 1, 1, 1],
                [0, 1, 0, -1, 1, 1, 0],
                [0, 0, 0, -1, 1, 0, 0],
                [-1, 0, 0, -1, 0, 0, 0],
                [-2, 1, 0, -1, 0, 0, 0],
                [-2, 1, -1, 0, 0, 0, 0],
                [-2, -1, -1, -1, 1, 1, 1],
            ]
        ),
    ),
]


@pytest.mark.parametrize(
    "test_input,expected",
    zip(PROBLEMS, EXPECTED_FOR_PROBLEM),
)
def test_constraint_and_objective_function_for_chung_problem_class(
    test_input, expected
):

    problem = test_input.problem
    F = problem.bi_objective_function(test_input.X_pop)
    G = problem.constraints(test_input.X_pop)
    assert (F == expected.F).all()
    assert np.allclose(
        G, expected.G, atol=problem.EPSILON_VALUE_FOR_EQUALITY_CONSTRAINT
    )


# ================================= Parámetros =====================================

WRONG_PROBLEMS = [
    (
        ChungProblem,
        {
            "D": np.array([1, 1]),
            "Z": [1, 1],
            "OCM": np.array([[0, 1], [1, 1]]),
        },
        InvalidOcurrenceMatrixz,
    ),
    (
        ChungProblem,
        {
            "D": np.array([1, 1, 1]),
            "Z": [1, 1, 1],
            "OCM": np.array([[0, 1, 1], [1, 0, 1], [10, 10, 0]]),
        },
        InvalidOcurrenceMatrixz,
    ),
    (
        ChungProblem,
        {
            "D": np.array([1, 1, 1]),
            "Z": [1, 1, 1],
            "OCM": np.array([[0, 1, 1], [1, 0, 1], [0, 0, 0]]),
        },
        InvalidOcurrenceMatrixz,
    ),
    (
        ChungProblemWithDivisionsPlusWeight,
        {
            "D": np.array([1, 1, 1]),
            "Z": np.array([[1, 1], [1, 1], [1, 1]]),
            "division_types": np.array([1, 1, 1]),
            "W": np.array([1, 1]),
            "WT": np.array([[0, 5], [0, 5]]),
            "OCM": np.array([[0, 1, 1], [1, 0, 1], [0, 0, 0]]),
        },
        InvalidOcurrenceMatrixz,
    ),
    (
        ChungProblem,
        {
            "D": np.array([-1, 1]),
            "Z": [1, 1],
            "OCM": np.array([[0, 1], [1, 0]]),
        },
        NonPositiveParameterError,
    ),
    (
        ChungProblem,
        {
            "D": np.array([1, 1, 1]),
            "Z": [1, 1],
            "OCM": np.array([[0, 1], [1, 0]]),
        },
        SkuAndOCMDimensionError,
    ),
    (
        ChungProblemStrictlyWeighted,
        {
            "D": np.array([1, 1]),
            "W": np.array([0.5, 0.5, 0.5]),
            "Z": np.array([1, 1]),
            "OCM": np.array([[0, 1], [1, 0]]),
            "WT": [(0, 11), (0, 1)],
        },
        SkusAndWeightDimensionError,
    ),
    (
        ChungProblemStrictlyWeighted,
        {
            "D": np.array([1, 1]),
            "W": np.array([0.5, 0.5]),
            "Z": np.array([1, 1]),
            "OCM": np.array([[0, 1], [1, 0]]),
            "WT": [(0, 1), (0, 1), (0, 1)],
        },
        ClusterAndWeightToleranceDimensionError,
    ),
    (
        ChungProblemStrictlyWeighted,
        {
            "D": np.array([1, 1]),
            "W": np.array([0.5, 0.5]),
            "Z": np.array([1, 1]),
            "OCM": np.array([[0, 1], [1, 0]]),
            "WT": [
                (0, 1),
                (10, 5),
            ],
        },
        TypeError,
    ),
    (
        ChungProblem,
        {
            "D": np.array([1, 1, 1]),
            "Z": np.array([1, 1]),
            "OCM": np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
        },
        NoCapacityForSkusError,
    ),
    (
        ChungProblemStrictlyWeighted,
        {
            "D": np.array([1, 1, 1]),
            "Z": np.array([2, 2]),
            "OCM": np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
            "W": np.array([10, 10, 10]),
            "WT": np.array([[0, 5], [0, 5]]),
        },
        LAPNoFeasibleSolutionError,
    ),
    (
        ChungProblem,
        {
            "D": np.array([5, 10, 15]),
            "Z": np.array([-2, 1]),
            "OCM": np.array([[0, 2, 3], [2, 0, 5], [3, 5, 0]]),
        },
        NonPositiveParameterError,
    ),
    (
        ChungProblem,
        {
            "D": np.array([5, -10, 15]),
            "Z": np.array([2, 1]),
            "OCM": np.array([[0, 2, 3], [2, 0, 5], [3, 5, 0]]),
        },
        NonPositiveParameterError,
    ),
    (
        ChungProblem,
        {
            "D": np.array([5, 10, 15]),
            "Z": np.array([2, 1]),
            "OCM": np.array([[0, 2, 3], [2, 0, 5], [3, -5, 0]]),
        },
        NonPositiveParameterError,
    ),
    (
        ChungProblem,
        {
            "D": np.array([]),
            "Z": np.array([]),
            "OCM": np.array([[0, 2, 3], [2, 0, 5], [3, 5, 0]]),
        },
        NoParametersError,
    ),
    (
        ChungProblem,
        {
            "D": np.array([5, 10, 15]),
            "Z": np.array([2, 1]),
            "OCM": np.array([[1, 2, 3], [2, 0, 5], [3, 5, 0]]),
        },
        InvalidOcurrenceMatrixz,
    ),
]


@pytest.mark.parametrize("problem, kwargs, error", WRONG_PROBLEMS)
def test_correct_parameters(problem, kwargs, error):

    with pytest.raises(error):
        problem(**kwargs)
