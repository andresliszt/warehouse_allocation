# -*- coding: utf-8 -*-
"""Pruebas unitarias para casos atípicos."""

import pytest

from warehouse_allocation.algorithms import solve_chung_problem


@pytest.mark.atypical
def test_atypical_number_of_clusters(algorithm_data):

    res = solve_chung_problem(
        processes=1,
        algorithm_name="NSGA2",
        D=algorithm_data["D"],
        Z=algorithm_data["Z"] * 20,  # Muchos clusters
        OCM=algorithm_data["OCM"],
        verbose=True,
        iterations=100,
        pop_size=30,
        n_offsprings=30,
        constraints=True,
        min_infeas_pop_size=30,
    )

    assert (res.CV == 0).all()


@pytest.mark.atypical
def test_atypical_capacity(algorithm_data):

    res = solve_chung_problem(
        processes=1,
        algorithm_name="NSGA2",
        D=algorithm_data["D"],
        Z=[
            z * 20 for z in algorithm_data["Z"]
        ],  # Capacidades con extrema holgura
        OCM=algorithm_data["OCM"],
        verbose=True,
        iterations=100,
        pop_size=30,
        n_offsprings=30,
        constraints=True,
        min_infeas_pop_size=30,
    )

    assert (res.CV == 0).all()
