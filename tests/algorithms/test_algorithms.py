# -*- coding: utf-8 -*-
"""Pruebas unitarias para ``warehouse_allocation.models.algorithms``

En este modulo se testeará con data real del centro de distribución
almacenada en archivos binarios que contienen la información mínima
para la ejecución de los modelos genéticos.


"""


import pytest
import numpy as np
from pymoo.operators.mutation.nom import NoMutation

from warehouse_allocation.algorithms import solve_chung_problem
from warehouse_allocation.operators.callback import (
    StopMutationAfterNgenCallback,
    ReportObjectiveBestValuesCallback,
)
from warehouse_allocation.operators import NonDominatedSortingSamplingWeighted
from warehouse_allocation.operators import NonDominatedSortingSamplingWithDivisions
from warehouse_allocation.operators import NonDominatedSortingSampling
from warehouse_allocation.models.chung import ChungProblem
from warehouse_allocation.models.chung import ChungProblemStrictlyWeighted
from warehouse_allocation.models.chung import ChungProblemWithDivisions
from warehouse_allocation.models.chung import ChungProblemWithDivisionsPlusWeight


def __compare_res(res_few_iteration, res_much_iterations, n_obj=2):

    assert len(res_few_iteration.opt) <= len(res_much_iterations.opt)
    # Función objetivo de afinidad debe ser más óptima

    for n in range(n_obj):
        # Todos los objetivos deben mejorar con el pasar de las iteraciones
        # o al menos no empeorar
        assert (
            res_few_iteration.F[:, n].min()
            >= res_much_iterations.F[:, n].min()
        )

        # Callack de report, debe coincidir con el objetivo alcanzado

        assert (
            res_few_iteration.F[:, n].min()
            == res_few_iteration.algorithm.callback.data[f"F_{n + 1}"][
                :, n
            ].min()
        )

        assert (
            res_much_iterations.F[:, n].min()
            == res_much_iterations.algorithm.callback.data[f"F_{n+ 1}"][
                :, n
            ].min()
        )


def __compare_res_with_crossover_variations(
    res_best_traffic, res_best_affinity
):

    # Crossover con parámetro aff_prob mayor deberá tener mejor
    # objetivo de afinidad
    assert res_best_traffic.F[:, 0].min() >= res_best_affinity.F[:, 0].min()
    # Crossover con parámetro aff_prob menor deberá tener mejor
    # objetivo de tráfico
    assert res_best_traffic.F[:, 1].min() <= res_best_affinity.F[:, 1].min()


def __form_kwargs(algorithm_data, kwargs_list):

    base_dict = {
        "D": algorithm_data["D"],
        "Z": algorithm_data["Z"],
        "OCM": algorithm_data["OCM"],
    }

    if kwargs_list == ["D", "Z", "OCM"]:
        return base_dict
    if kwargs_list == ["D", "Z", "OCM", "W", "WT"]:
        return {
            **base_dict,
            "W": algorithm_data["W"],
            "WT": algorithm_data["WT"],
        }

    if kwargs_list == ["D", "Z_DIV", "OCM", "division_types"]:
        base_dict["Z"] = algorithm_data["Z_DIV"]
        return {
            **base_dict,
            "division_types": algorithm_data["division_types"],
        }

    if kwargs_list == ["D", "Z_DIV", "OCM", "division_types", "W", "WT"]:
        base_dict["Z"] = algorithm_data["Z_DIV"]
        return {
            **base_dict,
            "division_types": algorithm_data["division_types"],
            "W": algorithm_data["W"],
            "WT": algorithm_data["WT"],
        }
    raise NotImplementedError(
        "``kwargs_list`` no válida, revisar ``TEST_ALGORITHM_DATA``"
    )


TEST_ALGORITHMS_DATA = [
    ["D", "Z", "OCM"],
    ["D", "Z", "OCM", "W", "WT"],
    ["D", "Z_DIV", "OCM", "division_types"],
    ["D", "Z_DIV", "OCM", "division_types", "W", "WT"],
]

# Importante no cambiar el orden de los nombres dentro de las listas
# en TEST_ALGORITHMS_DATA. __form_kwargs llama al operador `==` para
# listas


@pytest.mark.CH
@pytest.mark.parametrize("kwargs_list", TEST_ALGORITHMS_DATA)
def test_solve_chung_problem_function(kwargs_list, algorithm_data):

    kwargs = __form_kwargs(algorithm_data, kwargs_list)

    res_2 = solve_chung_problem(
        **kwargs,
        algorithm_name="NSGA2",
        verbose=True,
        iterations=2,
        pop_size=100,
        algorithm_callback=ReportObjectiveBestValuesCallback(),
        n_offsprings=30,
    )

    res_100 = solve_chung_problem(
        **kwargs,
        algorithm_name="NSGA2",
        sampling=res_2.X,
        verbose=True,
        iterations=200,
        pop_size=100,
        n_offsprings=30,
        algorithm_callback=ReportObjectiveBestValuesCallback(),
    )

    __compare_res(res_2, res_100)


@pytest.mark.skip(reason="Test subjetiva")
@pytest.mark.CROSSWE
@pytest.mark.parametrize("kwargs_list", TEST_ALGORITHMS_DATA)
def test_comparing_crossover_operator(
    monkeypatch, kwargs_list, algorithm_data
):
    def mockreturn(iterable, size=None, replace=None):
        if not size and not replace:
            return iterable[0]
        return iterable

    # mockreturn sustiuira a random.choice
    # eso nos permite tener control de los skus
    # que se intercambiarán para poder obtener
    # expresiones directas para los assert

    monkeypatch.setattr(np.random, "choice", mockreturn)

    def __make_sample(kwargs):
        if {"D", "Z", "OCM", "division_types", "W", "WT"} <= kwargs.keys():
            return NonDominatedSortingSamplingWithDivisions()._do(
                ChungProblemWithDivisionsPlusWeight(pool=None, **kwargs),
                n_samples=50,
            )
        if {"D", "Z", "OCM", "division_types"} <= kwargs.keys():
            return NonDominatedSortingSamplingWithDivisions()._do(
                ChungProblemWithDivisions(pool=None, **kwargs),
                n_samples=50,
            )

        if {"D", "Z", "OCM", "W", "WT"} <= kwargs.keys():
            return NonDominatedSortingSamplingWeighted()._do(
                ChungProblemStrictlyWeighted(pool=None, **kwargs),
                n_samples=50,
            )

        return NonDominatedSortingSampling()._do(
            ChungProblem(pool=None, **kwargs),
            n_samples=50,
        )

    kwargs = __form_kwargs(algorithm_data, kwargs_list)

    sample = __make_sample(kwargs)

    res_cross_0 = solve_chung_problem(
        **kwargs,
        algorithm_name="NSGA2",
        crossover_aff_prob=0,
        sampling=sample,
        verbose=True,
        iterations=200,
        pop_size=50,
        n_offsprings=30,
        algorithm_callback=StopMutationAfterNgenCallback(after_gen=1),
    )
    # Comenzamos con el resultado del sampling anterior
    res_cross_1 = solve_chung_problem(
        **kwargs,
        algorithm_name="NSGA2",
        crossover_aff_prob=1,
        sampling=sample,
        verbose=True,
        iterations=200,
        pop_size=50,
        n_offsprings=30,
        algorithm_callback=StopMutationAfterNgenCallback(after_gen=1),
    )

    __compare_res_with_crossover_variations(res_cross_0, res_cross_1)

    assert isinstance(res_cross_0.algorithm.mating.mutation, NoMutation)
    assert isinstance(res_cross_1.algorithm.mating.mutation, NoMutation)


@pytest.mark.CV
@pytest.mark.parametrize("kwargs_list", TEST_ALGORITHMS_DATA)
def test_constraints(kwargs_list, algorithm_data):

    kwargs = __form_kwargs(algorithm_data, kwargs_list)

    res_constr = solve_chung_problem(
        **kwargs,
        algorithm_name="NSGA2",
        verbose=True,
        iterations=200,
        pop_size=50,
        n_offsprings=30,
        constraints=True,
        min_infeas_pop_size=50,
    )

    # Este toma toda la población, incluídos los no feasibles (si es que los hay)
    assert (
        res_constr.problem.constraints(
            np.array([ind.X for ind in res_constr.pop])
        )
        <= 0
    ).all()

    # Este solo toma los feasibles
    assert (res_constr.CV == 0).all()


@pytest.mark.PENAL
def test_penalizattions(algorithm_data):

    res = solve_chung_problem(
        D=algorithm_data["D"],
        Z=np.full(
            len(algorithm_data["Z"]),
            len(algorithm_data["D"]) // len(algorithm_data["Z"]) + 1,
        ),
        OCM=algorithm_data["OCM"],
        clusters_penalizations=np.linspace(0, 1, num=len(algorithm_data["Z"])),
        algorithm_name="NSGA2",
        verbose=True,
        iterations=500,
        pop_size=50,
        n_offsprings=30,
        constraints=True,
        min_infeas_pop_size=50,
    )

    def __demand_on_cluster(cluster, D):
        return D[np.flatnonzero(cluster)].sum()

    worst_aff = []
    best_aff = []

    for individual in res.X:
        matrix_individual = res.problem.matrix_individual(individual)
        worst_aff.append(
            __demand_on_cluster(matrix_individual[0], res.problem.D)
        )
        best_aff.append(
            __demand_on_cluster(matrix_individual[-1], res.problem.D)
        )

    assert np.mean(worst_aff) < np.mean(best_aff)
