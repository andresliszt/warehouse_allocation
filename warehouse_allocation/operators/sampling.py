# -*- coding: utf-8 -*-
"""Operadores de sampling para algoritmo genético."""

import abc
from typing import Union

import numpy as np
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.core.sampling import Sampling

from warehouse_allocation import logger
from warehouse_allocation.models.bases import ChungProblemBase
from warehouse_allocation.models.chung import ChungProblem
from warehouse_allocation.models.chung import ChungProblemStrictlyWeighted
from warehouse_allocation.models.chung import ChungProblemWithDivisions
from warehouse_allocation.models.chung import ChungProblemWithDivisionsPlusWeight
from warehouse_allocation.operators.lap import LAPWithCostFlow
from warehouse_allocation.operators.lap import LAPWithCostFlowRestrictedClusters
from warehouse_allocation.operators.lap import (
    LAPWithCostFlowRestrictedClustersPlusDivision,
)


class NonDominatedSortingSamplingBase(Sampling, abc.ABC):
    """Sampleo para el problema de Chung con crowding selection.

    Este operador define un método para sampleo aleatorio, dado por
    :func:`~warehouse_allocation.models.operators.sampling.SamplingWithRankAndCrowdingSurvivalSelection.random_individual`.
    Este método define un slotting aleatorio satisfaciendo todos los constraints
    del problema de Chung. La técnica usada para esta asignación aleatoria
    es un modelo de programación lineal random resuelto en el framework
    ``or tools`` de Google. Adicionalmente, se considera el slotting
    que minimiza el problema de asignación de flujo lineal usando
    como matriz de costo :math:`M_D`, donde la entrada :math:`i,j`
    es simplemente la demanda del SKU :math:`j`, este problema encontrará
    un slotting que **minimiza** la demanda por cluster, lo que se traduce
    en distribuir de forma uniforme esta por los clusters. Al contrario,
    si se usa el negativo de :math:`M_D`, se encontrará un slotting que
    satura ciertos clusters con SKUs demandados, lo que implica un aumento
    en la afinidad (pero no es una *buena* afinidad). Se consideran ambos
    slotting tomándose como referencia para los valores óptimos de la función
    objetivo del tráfico y de afindiad.

    El solver del problema de flujo mínimo con matriz de costo se puede ver
    en :func:`~warehouse_allocation.models.operators.utils.LAPWithCostFlow`.

    El método :func:`~warehouse_allocation.models.operators.sampling.SamplingWithRankAndCrowdingSurvivalSelection._do`.
    usado internamente por ``pymoo``, genera una cantidad de individuos (slottings)
    dada por el parámetro ``n_samples``, que corresponde a la población inicial
    para el algoritmo genético. Esta clase tiene un método de selección
    de los `mejores individuos`, el procedemiento es el siguiente: Se
    generan ``n_samples`` individuos, y luego se seleccionan los mejores
    (rank = 0) con la clase :class:`pymoo.algorithms.nsga2.RankAndCrowdingSurvivalRankAndCrowdingSurvival`.
    Este procedimiento se repite tantas veces como sea necesario hasta obtener ``n_samples``
    individuos.

    """

    @abc.abstractmethod
    def individual(
        self, problem: ChungProblemBase, Cost: np.ndarray
    ) -> np.ndarray:
        """Genera un individuo (slotting) con todas las restricciones.

        Este método debe ser implementado como un problema de asignación
        lineal definido en :py:mod:`warehouse_allocation.operators.lap`, usando
        el adecuado en función de ``problem``.

        :param problem: Problema que define el individuo.
        :param Cost: Costo para el input del problema de asignación
            lineal que resolverá este método.

        """

    @staticmethod
    def random_cost(problem: ChungProblemBase) -> np.ndarray:
        """Genera una matriz de costo random de números enteros.

        La libreria ``ortools`` en el problema de optimización
        :class:`ortools.graph.pywrapgraph.SimpleMinCostFlow`
        no admite matriz de costo con números no enteros.

        """
        return np.random.randint(
            problem.n_skus, size=(problem.n_clusters, problem.n_skus)
        )

    def _do(
        self, problem: ChungProblemBase, n_samples: int, **kwargs
    ) -> np.ndarray:
        """Funcionalidad principal de uso interno en ``pymoo``"""

        # los kwargs son de uso interno de la librería y no podemos tocarlos

        pop_best = Population()
        evaluator = Evaluator()
        survival = RankAndCrowdingSurvival()

        while len(pop_best) < n_samples:
            pop = Population.new(
                X=np.array(
                    [
                        self.individual(
                            problem, Cost=self.random_cost(problem)
                        )
                        for _ in range(n_samples)
                    ]
                )
            )
            pop = evaluator.eval(problem, pop)
            # Evaluamos la función objetivo, constraints, etc.
            pop = survival._do(problem, pop, n_survive=len(pop))
            # Seteamos el atributo del rank en cada miembro de la población
            pop = pop[pop.get("rank") == 0]
            # Seleccionamos las mejores de acuerdo al ranking
            pop_best = Population.merge(pop_best, pop)
            logger.info(
                f"Cantidad de población actual {len(pop_best)} de {n_samples}"
            )
        pop_best = np.array([ind.X for ind in pop_best])
        # Trabajamos con numpy array en vez de Individual

        return pop_best[:n_samples]


class NonDominatedSortingSampling(NonDominatedSortingSamplingBase):
    def individual(
        self, problem: ChungProblem, Cost: np.ndarray
    ) -> np.ndarray:
        """Genera un individuo (slotting) random."""
        return LAPWithCostFlow(Cost=Cost, Z=problem.Z).solve().flatten()


class NonDominatedSortingSamplingWeighted(NonDominatedSortingSamplingBase):
    def individual(
        self, problem: ChungProblemStrictlyWeighted, Cost: np.ndarray
    ) -> np.ndarray:
        """Genera un individuo (slotting) random con restricciones de pesos."""

        return (
            LAPWithCostFlowRestrictedClusters(
                Cost=Cost,
                Z=problem.Z,
                allowed_clusters_by_sku=problem.allowed_clusters_by_sku,
            )
            .solve()
            .flatten()
        )


class NonDominatedSortingSamplingWithDivisions(
    NonDominatedSortingSamplingBase
):
    def individual(
        self,
        problem: Union[
            ChungProblemWithDivisions, ChungProblemWithDivisionsPlusWeight
        ],
        Cost: np.ndarray,
    ) -> np.ndarray:
        """Genera un individuo (slotting) random con restricciones de divisiones y/o pesos."""

        return (
            LAPWithCostFlowRestrictedClustersPlusDivision(
                Cost=Cost,
                Z=problem.Z,
                allowed_clusters_by_sku=problem.allowed_clusters_by_sku,
                division_types=problem.division_types,
            )
            .solve()
            .flatten()
        )
