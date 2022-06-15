import pytest
import numpy as np

from pymoo.core.population import Population

from warehouse_allocation.operators.duplicates import DropDuplicates


TEST_DUPLICATES_DATA = [
    (
        Population.new(
            X=[
                [1, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 1, 1],
                [1, 1, 1, 0, 1, 1],
                [1, 1, 0, 0, 0, 0],
            ]
        ),
        Population.new(X=[[1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 1, 1]]),
        np.array([False, True, False, True]),
    ),
    (
        Population.new(
            X=[
                [1, 1, 0, 0, 0, 0, 1, 1],
                [1, 1, 0, 0, 1, 1, 1, 1],
                [1, 1, 0, 0, 0, 0, 1, 1],
                [1, 1, 0, 0, 0, 0, 0, 0],
            ]
        ),
        None,
        np.array([False, False, True, False]),
    ),
    (
        Population.new(
            X=[
                [1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
                [1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ),
        Population.new(
            X=[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
        ),
        np.array([False, False, False, False]),
    ),
    (
        Population.new(
            X=[
                [1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
                [1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ),
        Population.new(
            X=[
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        ),
        np.array([False, False, True, True, False]),
    ),
    (
        Population.new(
            X=[
                [1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ),
        Population.new(
            X=[
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ),
        np.array([False, True, False, True, False, True, True]),
    ),
]


DROP_DUPLICATE = DropDuplicates()


@pytest.mark.parametrize("pop, other, is_duplicate", TEST_DUPLICATES_DATA)
def test_drop_duplicates(pop, other, is_duplicate):

    assert (
        DROP_DUPLICATE._do(pop, other, np.full(len(pop), False))
        == is_duplicate
    ).all()
