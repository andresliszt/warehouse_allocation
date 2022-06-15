# -*- coding: utf-8 -*-
"""Operadores de crossover para algoritmo genético."""

import abc
from typing import Tuple
from typing import Union

import numpy as np
from pymoo.core.crossover import Crossover

from warehouse_allocation.exc import LAPNoFeasibleSolutionError
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

# TODO: Test: ¿Que pasa si un cluster de cambio no tiene ningun SKU asociado?


class ChungPartiallyMappedCrossoverBase(Crossover, abc.ABC):
    """`Partial Mapped Crossover` adaptado con mapping de asignación Lineal.

    Esta clase implementa una modificación del algoritmo de crossover
    presentado en :cite:`1999:Larranaga`. La cruza entre dos individuos,
    generalmente denotados por ``P1`` y ``P2``, intenta generar un
    descendiente (offspring) que un su mayoría *conserve la genética* de
    ``P1``, y la información faltante sea suplida con la *genética* de
    ``P2``. En este operador en particular, el descendiente tendrá todos
    los clusters *casi ídenticos* a ``P2``, salvo por uno escogido de
    manera aleatoria, que tendrá la información de ``P1``. Al hacer esta
    cruza, al igual que en :cite:`1999:Larranaga`, habrán duplicados y
    posibles SKUs sin asignar, para ello la re asignación es hecha con
    asignación lineal usando `maximum flow`, presentado en
    :cite:`2019:Dash`.

    """

    def __init__(self, aff_prob: float = 0.5, prob: float = 0.9) -> None:
        if not 0 <= aff_prob <= 1:
            raise ValueError(
                f"Parámetro `aff_prob` es una probabilidad. Recibido = `{aff_prob}`"
            )

        super().__init__(2, 2, prob)

        self.aff_prob = aff_prob

    @staticmethod
    def shift_allowed(reallocation_set, allowed_clusters_by_sku):
        return {
            sku_index: allowed_clusters_by_sku[sku]
            for sku_index, sku in dict(
                zip(range(len(reallocation_set)), reallocation_set)
            ).items()
        }

    @staticmethod
    def __cluster_traff(
        cluster_idx: int, D: np.ndarray, reallocation_set: np.ndarray
    ):
        return D[np.flatnonzero(cluster_idx)].sum() + D[reallocation_set]

    def traffic_cost(
        self,
        individual: np.ndarray,
        reallocation_set: np.ndarray,
        D: np.ndarray,
    ) -> np.ndarray:
        """Construye matriz de costo para asignación LAP basada en demanda.

        La matriz de costo está definida por :math:`C = (c_{k,i})`, donde
        el :math:`c_{k,i}` es el costo de asignar el sku :math:`i` en el cluster
        :math:`k`, que en este caso esta dado por la **demanda** neta que se
        generaría al incorporar el sku en el cluster, calculandose en relación
        a los skus ya asignados a dicho cluster.

        :param individual: El individuo es un su estado de cruce parcial. Esto es,
                corresponde al individuo en su forma matricial (de tamaño
                :math:`n_{\\text{clusters}}\\times n_{\\text{skus}}`) que ha
                sido llenado con toda la información de la madre y el padre
                dada por la lógica que define esta clase, salvo por los skus
                que hay que reasignar con un ``mapping``.
        :param reallocation_set: Corresponden a los índices de los skus que deben ser
                reasignados.
        :param D: Vector de demandas de los skus.
        :return: La matriz de costos desde demanda para ser usada en el problema LAP.

        """

        return np.apply_along_axis(
            self.__cluster_traff, 1, individual, D, reallocation_set
        )

    @staticmethod
    def __cluster_aff(
        cluster: np.ndarray,
        cluster_penalization: float,
        OCM: np.ndarray,
        reallocation_set: np.ndarray,
    ):
        # Lap problem de ortools intenta MINIMIZAR, por ende -OCM para maximizar
        return -cluster_penalization * OCM[:, reallocation_set][
            np.flatnonzero(cluster)
        ].sum(axis=0)

    def affinity_cost(
        self,
        individual: np.ndarray,
        reallocation_set: np.ndarray,
        OCM: np.ndarray,
        clusters_penalizations: np.ndarray,
    ) -> np.ndarray:
        """Construye matriz de costo para asignación LAP basada en afinidad.

        La matriz de costo está definida por :math:`C = (c_{k,i})`, donde
        el :math:`c_{k,i}` es el costo de asignar el sku :math:`i` en el cluster
        :math:`k`, que en este caso esta dado por la **afinidad** neta que se
        generaría al incorporar el sku en el cluster, calculandose en relación
        a los skus ya asignados a dicho cluster.

        :param individual: El individuo es un su estado de cruce parcial. Esto es,
                corresponde al individuo en su forma matricial (de tamaño
                :math:`n_{\\text{clusters}}\\times n_{\\text{skus}}`) que ha
                sido llenado con toda la información de la madre y el padre
                dada por la lógica que define esta clase, salvo por los skus
                que hay que reasignar con un ``mapping``.
        :param reallocation_set: Corresponden a los índices de los skus que deben ser
                reasignados.
        :param OCM: Matriz de afinidad entre skus.

        :return: La matriz de costos desde afinidad para ser usada en el problema LAP.

        """

        return np.array(
            [
                self.__cluster_aff(
                    cluster, cluster_penalization, OCM, reallocation_set
                )
                for cluster, cluster_penalization in zip(
                    individual, clusters_penalizations
                )
            ]
        )

        # return np.apply_along_axis(
        #     self.__cluster_aff, 1, individual, OCM, reallocation_set
        # )

    def affinity_or_traffic_cost(
        self,
        problem: ChungProblemBase,
        offspring: np.ndarray,
        reallocation_set: np.ndarray,
    ):
        """Cuando usar matriz de afinidad/demanda con prob :attr:`.ChungPartiallyMappedCrossoverBase.aff_prob`"""
        return (
            self.affinity_cost(
                offspring,
                reallocation_set,
                problem.OCM,
                problem.clusters_penalizations,
            )
            if np.random.random(1)[0] <= self.aff_prob
            else self.traffic_cost(offspring, reallocation_set, problem.D)
        )

    @abc.abstractmethod
    def mapping(
        self,
        problem: ChungProblemBase,
        offspring: np.ndarray,
        reallocation_set: np.ndarray,
    ) -> np.ndarray:
        """Reasigna skus en un offspring con `maximum flow`

        Usa  Matriz de costos :math:`C = (c_{k,i})` es generada
        basada en afinidad o en tráfico dada por
        :meth:`.ChungPartiallyMappedCrossoverBase.affinity_or_traffic_cost```

        :param offspring: Problema que se intenta resolver.
        :param offspring: Offspring donde serán asignados
            skus usando `maximum flow`
        :param reallocation_set: Índices de skus que deben ser reasignados.

        :return: Offspring completo para retornar en operador crossover.

        """

    @staticmethod
    def swap(
        P1: np.ndarray, P2: np.ndarray, cluster_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Realiza intercambio de genética entre ``P1`` y ``P2``

        El offspring conserva la genética de ``P2`` en todos
        los clusters, salvo en ``cluster_idx``. Los skus
        problemáticos (repetidos o sin asignar) están dados
        son retornados para poder reasignarse en los problemas
        LAP.

        :param P1: Individuo de cruza.
        :param P2: Individuo de cruza.

        :return: Offspring con intercambio de genética y skus que
            deben ser reasignados.

        """

        swap_set = np.flatnonzero(P1[cluster_idx])
        crossover_skus = np.flatnonzero(
            np.logical_xor(P1[cluster_idx], P2[cluster_idx])
        )
        null_set = np.intersect1d(crossover_skus, swap_set)
        # Apagamos los bits de la madre del conjunto NULL
        # Estos skus corresponden a skus que se van a poner
        # en el pasillo de intercambio, pero que en la madre
        # están en otro pasillo, por lo tanto hay que apagar
        # los bits de la madre para evitar duplicados
        reallocation_set = np.setdiff1d(crossover_skus, swap_set)
        # Estos skus corresponden a los skus de M
        # que están en el pasillo de intercambio, pero no están en SWAP
        # por lo tanto necesitan reasignación
        _P2 = P2.copy()
        _P2[:, null_set] = 0
        offspring = _P2
        offspring[cluster_idx] = P1[cluster_idx]
        offspring[:, reallocation_set] = 0

        return offspring, reallocation_set

    @abc.abstractmethod
    def _crossover(self, P1: np.ndarray, P2, problem) -> np.ndarray:
        """Partially mapped crossover con re-asingnación como problema de flujo.

        Se considera la idea principal del operador
        `Partially Mapped Crossover` (PMC), esto es, dados
        ``P1`` y ``P2``, se genera un offspring que contiene
        todos los clusters idénticos a los de ``P2``,
        salvo por uno que corresponde al de ``P1``. Este cluster tomado
        de ``P1`` es denominado en la literatura como conjunto `SWAP`.
        Cuando se hace la cruza antes mencionada, hay que lidiar
        con los SKUs que puedan tener doble asignación
        (se copia la información tanto de ``P1`` como de ``P2``
        en el descendiente) y con los SKUs que pueden quedar
        sin asignación. Se distinguen dos casos: Si un SKU está en el cluster
        `SWAP` proveniente del ``P1``, y está en otro cluster en ``P2``,
        se tendría doble asignación, por lo tanto la información de este SKU
        en la madre debe ser eliminada. Si un SKU de ``P2`` esta alocado
        en el cluster donde se hará el intercambio, pero no está en `SWAP`,
        entonces dicho SKU quedaría sin asignación. El conjunto de todos
        los SKUs sin asignación, son reasignados en el offspring usando
        una técnica de programación lineal con `maximum flow`.

        La matriz de costo para re asignar, será construída con el método
        :class:`~ChungPartiallyMappedCrossover.affinity_or_traffic_cost`.

        :param P1: Individuo de cruza.
        :param P2: Individuo de cruza.
        :param problem: Problema que se intenta resolver.

        :return: Offspring de la cruza entre ``P1`` y ``P2`` con PMC.

        """

    def crossover(
        self,
        P1: np.ndarray,
        P2: np.ndarray,
        problem: ChungProblemBase,
    ) -> np.ndarray:
        """Funcionalidad principal de cruza.

        Se generan dos offspring desde
        ``P1`` y ``P2`` usando el método
        :meth:`~ChungPartiallyMappedCrossover.partially_mapped_crossover`

        :return: Los dos offspringg de la cruza entre ``P1``
            con ``P2`` y su intercambio de roles.

        """

        return np.array(
            [
                self._crossover(P1=P1, P2=P2, problem=problem),
                self._crossover(P1=P2, P2=P1, problem=problem),
            ]
        )

    def _do(
        self, problem: ChungProblemBase, X: np.ndarray, **kwargs
    ) -> np.ndarray:

        # TODO: CAMBIAR A UN MÉTODO apply!
        _, n_matings, __ = X.shape

        for ind_idx in range(n_matings):
            X[0][ind_idx], X[1][ind_idx] = self.crossover(
                X[0][ind_idx], X[1][ind_idx], problem
            )

        return X


class ChungPartiallyMappedCrossover(ChungPartiallyMappedCrossoverBase):
    """Crossover para problema de Chung estándar."""

    def mapping(
        self,
        problem: ChungProblem,
        offspring: np.ndarray,
        reallocation_set: np.ndarray,
    ) -> np.ndarray:
        return LAPWithCostFlow(
            Cost=self.affinity_or_traffic_cost(
                problem, offspring, reallocation_set
            ),
            Z=problem.Z - problem.clusters_usage_matrix_individual(offspring),
        ).solve()

    def _crossover(
        self,
        P1: np.ndarray,
        P2: np.ndarray,
        problem: ChungProblem,
    ) -> np.ndarray:
        """Partially mapped crossover sin restricciones de pesos.

        Siempre hay solución feasible para la reasignación dado un
        cluster ``SWAP``.

        """

        P1 = problem.matrix_individual(P1)
        P2 = problem.matrix_individual(P2)
        cluster_idx = np.random.choice(
            np.arange(problem.n_clusters),
        )
        offspring, reallocation_set = self.swap(P1, P2, cluster_idx)
        if reallocation_set.size == 0:
            return offspring.flatten()

        offspring[:, reallocation_set] = self.mapping(
            problem,
            offspring=offspring,
            reallocation_set=reallocation_set,
        )

        return offspring.flatten()


class ChungPartiallyMappedCrossoverWeigthed(ChungPartiallyMappedCrossoverBase):
    """Crossover para problema de Chung con restricciones de peso en los clusters."""

    def mapping(
        self,
        problem: ChungProblemStrictlyWeighted,
        offspring: np.ndarray,
        reallocation_set: np.ndarray,
    ) -> np.ndarray:
        return LAPWithCostFlowRestrictedClusters(
            Cost=self.affinity_or_traffic_cost(
                problem, offspring, reallocation_set
            ),
            Z=problem.Z - problem.clusters_usage_matrix_individual(offspring),
            allowed_clusters_by_sku=self.shift_allowed(
                reallocation_set=reallocation_set,
                allowed_clusters_by_sku=problem.allowed_clusters_by_sku,
            ),
        ).solve()

    def _crossover(
        self, P1: np.ndarray, P2: np.ndarray, problem
    ) -> np.ndarray:
        """Partially mapped crossover con restricciones de peso.

        No siempre hay solución feasible para la reasignación dado un
        cluster ``SWAP``. Al fallar, se toma otro cluster ``SWAP``. La
        conjetura es que hay al menos un cluster donde siempre será
        posible hacer una reasignación.

        """

        P1 = problem.matrix_individual(P1)
        P2 = problem.matrix_individual(P2)
        clusters = np.random.choice(
            np.arange(problem.n_clusters),
            size=problem.n_clusters,
            replace=False,
        )

        for clus in clusters:
            offspring, reallocation_set = self.swap(P1, P2, clus)
            if reallocation_set.size == 0:
                return offspring.flatten()
            try:

                offspring[:, reallocation_set] = self.mapping(
                    problem,
                    offspring=offspring,
                    reallocation_set=reallocation_set,
                )

                return offspring.flatten()

            except LAPNoFeasibleSolutionError:
                # Si es imposible la reasignación, se continua con otro cluster
                continue
        # La conjetura es que nunca entrará en este último return
        return P1.flatten()


class ChungPartiallyMappedCrossoverWithDivisions(
    ChungPartiallyMappedCrossoverWeigthed
):
    """Crossover para problema de Chung con divisiones.

    Este crossover es compatible para el problema que considera
    divisiones y el problema que considera tanto divisiones como
    restricciones de peso por cluster.

    """

    def mapping(
        self,
        problem: Union[
            ChungProblemWithDivisions, ChungProblemWithDivisionsPlusWeight
        ],
        offspring: np.ndarray,
        reallocation_set: np.ndarray,
    ) -> np.ndarray:

        return LAPWithCostFlowRestrictedClustersPlusDivision(
            Cost=self.affinity_or_traffic_cost(
                problem, offspring, reallocation_set
            ),
            Z=problem.Z - problem.clusters_usage_matrix_individual(offspring),
            division_types=problem.division_types[reallocation_set],
            allowed_clusters_by_sku=self.shift_allowed(
                reallocation_set=reallocation_set,
                allowed_clusters_by_sku=problem.allowed_clusters_by_sku,
            ),
        ).solve()
