# -*- coding: utf-8 -*-
"""Operadores de mutación para algoritmo genético."""

import abc
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
from pymoo.core.mutation import Mutation

from warehouse_allocation import logger
from warehouse_allocation.models.bases import ChungProblemBase
from warehouse_allocation.models.chung import ChungProblem
from warehouse_allocation.models.chung import ChungProblemStrictlyWeighted
from warehouse_allocation.models.chung import ChungProblemWithDivisions
from warehouse_allocation.models.chung import ChungProblemWithDivisionsPlusWeight

# TODO: En vez de seleccionar el cluster a permutar pimero, pasar directo al sku a permutar


class ChungAislePermutationBase(Mutation, abc.ABC):
    """Clase base para mutación de intercambio de skus por pasillos.

    Este operador de mutación intercambia skus entre pasillos que sean
    compatibles en términos de restricciones de pesos. Este operador no
    modifica las cantidades de skus por cluster, al ser solo una
    permutación.

    """

    # TODO: Elaborar permutaciones de más de un par de SKUs

    def __init__(self, prob: float = 0.1) -> None:
        super().__init__()
        self.prob = prob

    @staticmethod
    def log_warning_empty_cluster(
        problem: ChungProblemBase, permutation_cluster_index: int
    ) -> None:
        """Este log avisa que hay un cluster vacío.

        Nunca se debiese entrar en esta función si el problema que se
        intenta resolver fue bien planteado.

        """
        logger.warning(
            "Cluster vacío. Capacidad ``Z`` es demasiado grande. Se recomienda re formular problema",
            n_skus=problem.n_skus,
            n_clusters=problem.n_clusters,
            Z=list(problem.Z),
            cluster_index=permutation_cluster_index,
        )

    @abc.abstractmethod
    def mutate_individual(
        self, ind: np.ndarray, problem: ChungProblemBase
    ) -> np.ndarray:
        """Método principal que muta un individuo."""

    @staticmethod
    def permute_ind(
        ind: np.ndarray,
        current_cluster_index: int,
        permutation_cluster_index: int,
        current_sku_index: int,
        permutation_sku_index: int,
    ) -> np.ndarray:
        """Permuta SKUs de acuerdo a índices.

        :param ind: Individuo en su forma matricial.
        :param current_cluster_index: Índice de cluster del
            sku que se va permutar, cuyo
            índice es ``current_sku_index``.
        :param permutation_cluster_index: Índice de cluster del
            sku cuyo índice es ``permutation_sku_index``, que
            se va permutar con con el sku de
            índice ``current_sku_index``.
        :param sku_current_index: Índice de sku por el cúal se
            permutará el sku con índice ``permutation_sku_index``.
        :param permutation_sku_index: Índice de sku por el cúal se
            permutará el sku con índice ``current_sku_index``.

        """

        ind[permutation_cluster_index][current_sku_index] = 1
        ind[permutation_cluster_index][permutation_sku_index] = 0
        ind[current_cluster_index][current_sku_index] = 0
        ind[current_cluster_index][permutation_sku_index] = 1

        return ind.flatten()

    def _do(
        self, problem: ChungProblemBase, X: np.ndarray, **kwargs
    ) -> np.ndarray:

        M = np.random.random(X.shape[0])
        _X = np.full(X.shape, np.inf)
        # Contenedor para el offspring
        mutate, no_mutate = M < self.prob, M >= self.prob
        _X[no_mutate] = X[no_mutate]
        if X[mutate].size != 0:
            _X[mutate] = np.apply_along_axis(
                self.mutate_individual, 1, arr=X[mutate], problem=problem
            )
        return _X


class ChungAislePermutationMutation(ChungAislePermutationBase):
    """Intercambia dos skus de clusters sin considerar restricciones de pesos."""

    def mutate_individual(
        self, ind: np.ndarray, problem: ChungProblem
    ) -> np.ndarray:
        """Mutación de individuo al permutar un sku con otro.

        Se seleccionan random dos skus y se permutan

        :param ind: Individuo como vector binario.
        :param problem: Problema que define el individuo.

        """

        ind = problem.matrix_individual(ind)
        # Matriz de tamaño N_clusters x N_skus
        sku_indexes = np.arange(problem.n_skus)
        current_sku_index = np.random.choice(sku_indexes)
        current_cluster_index = np.flatnonzero(ind[:, current_sku_index])[0]
        # corresponde al cluster (único!!!)
        # donde está ubicado el sku
        try:
            permutation_sku_index = np.random.choice(
                np.setdiff1d(
                    sku_indexes, np.flatnonzero(ind[current_cluster_index])
                )
            )
            permutation_cluster_index = np.flatnonzero(
                ind[:, permutation_sku_index]
            )[0]
            ind = self.permute_ind(
                ind,
                current_cluster_index,
                permutation_cluster_index,
                current_sku_index,
                permutation_sku_index,
            )
        except ValueError:
            self.log_warning_empty_cluster(problem, permutation_cluster_index)
            return ind.flatten()
        return ind


class _ChungAislePermutationRestrictedClusters(ChungAislePermutationBase):
    """Base para operadores de mutación con restricciones sobre los clusters."""

    @staticmethod
    @abc.abstractmethod
    def candidates_selector(
        problem: Union[
            ChungProblemWithDivisions,
            ChungProblemStrictlyWeighted,
            ChungProblemWithDivisionsPlusWeight,
        ],
        skus_in_permutation_cluster: np.ndarray,
        current_sku_index: int,
        current_cluster_index: int,
    ) -> np.ndarray:
        """Retorna los skus candidatos para realizar permutación.

        Cuando de forma aleatoria se selecciona un sku
        (``current_sku_index``), se debe tener en cuenta
        las características del cluster donde está
        alocado (``current_cluster_index``) para poder
        seleccionar uno de los posibles skus alocados
        en el cluster target (``skus_in_permutation_cluster``).

        :param skus_in_permutation_cluster: Todos los skus
            alocados en el cluster target de la permutación.
        :param current_cluster_index: Índice del primer
            cluster que fue seleccionado para la permutación.
        :param current_sku_index: Índice del primer sku
            que fue seleccionado para la permutación

        """

    @staticmethod
    def pop_current_cluster_from_allowed(
        allowed_clusters: List[int], current_cluster_index: int
    ):
        """Cluster posibles para permutar, sin considerar el cluster actual."""

        return [
            clus_idx
            for clus_idx in allowed_clusters
            if clus_idx != current_cluster_index
        ]

    @staticmethod
    def _random_sku_to_permute(
        ind: np.ndarray, n_skus: int
    ) -> Tuple[int, int]:
        sku_current_index = np.random.choice(np.arange(n_skus))
        return sku_current_index, np.flatnonzero(ind[:, sku_current_index])[0]

    @staticmethod
    def _skus_in_permutation_cluster(
        ind: np.ndarray, allowed_clusters: List[int]
    ) -> Tuple[int, np.ndarray]:

        permutation_cluster_index = np.random.choice(allowed_clusters)
        # Cluster al que haremos el cambio
        return permutation_cluster_index, np.flatnonzero(
            ind[permutation_cluster_index]
        )

    def mutate_individual(
        self,
        ind: np.ndarray,
        problem: Union[
            ChungProblemWithDivisions,
            ChungProblemStrictlyWeighted,
            ChungProblemWithDivisionsPlusWeight,
        ],
    ):
        """Mutación de individuo al permutar un sku con otro.

        Método para problemas donde no es posible permutar
        libremente un sku con otro, pues hay
        restricciones de peso sobre los clusters o
        restricciones de división.

        :param ind: Individuo como vector binario.
        :param problem: Problema que define el individuo.

        """

        ind = problem.matrix_individual(ind)

        current_sku_index, current_cluster_index = self._random_sku_to_permute(
            ind, problem.n_skus
        )

        allowed_clusters = self.pop_current_cluster_from_allowed(
            problem.allowed_clusters_by_sku[current_sku_index],
            current_cluster_index,
        )

        if not allowed_clusters:
            return ind.flatten()

        (
            permutation_cluster_index,
            skus_in_permutation_cluster,
        ) = self._skus_in_permutation_cluster(ind, allowed_clusters)

        if skus_in_permutation_cluster.size == 0:
            self.log_warning_empty_cluster(problem, permutation_cluster_index)

        candidates = self.candidates_selector(
            problem,
            skus_in_permutation_cluster,
            current_sku_index,
            current_cluster_index,
        )

        if candidates.size == 0:
            return ind.flatten()

        return self.permute_ind(
            ind,
            current_cluster_index=current_cluster_index,
            permutation_cluster_index=permutation_cluster_index,
            current_sku_index=current_sku_index,
            permutation_sku_index=np.random.choice(candidates),
        )


class ChungAislePermutationMutationWeighted(
    _ChungAislePermutationRestrictedClusters
):
    """Mutación de permutación considerando restricción de pesos."""

    @staticmethod
    def candidates_selector(
        problem: ChungProblemStrictlyWeighted,
        skus_in_permutation_cluster: np.ndarray,
        current_sku_index: int,
        current_cluster_index: int,
    ) -> np.ndarray:

        # No usamos current_sku_index, pero el abstract si lo necesita
        weights_permutation_cluster = problem.W[skus_in_permutation_cluster]
        WT_current_cluster = problem.WT[current_cluster_index]

        return skus_in_permutation_cluster[
            np.where(
                (WT_current_cluster[0] <= weights_permutation_cluster)
                & (weights_permutation_cluster <= WT_current_cluster[1])
            )[0]
        ]


class ChungAislePermutationWithDivisions(
    _ChungAislePermutationRestrictedClusters
):
    """Mutación de permutación considerando divisiones."""

    @staticmethod
    def candidates_selector(
        problem: ChungProblemWithDivisions,
        skus_in_permutation_cluster: np.ndarray,
        current_sku_index: int,
        current_cluster_index: int,
    ) -> np.ndarray:

        # No usamos current_cluster_index, pero el abstract si lo necesita
        division_types_permutation_cluster = problem.division_types[
            skus_in_permutation_cluster
        ]

        return skus_in_permutation_cluster[
            division_types_permutation_cluster
            == problem.division_types[current_sku_index]
        ]


class ChungAislePermutationWithDivisionsPlusWeight(
    _ChungAislePermutationRestrictedClusters
):
    """Mutación de permutación considerando restricción de pesos y divisiones."""

    def candidates_selector(
        self,
        problem: ChungProblemWithDivisionsPlusWeight,
        skus_in_permutation_cluster: np.ndarray,
        current_sku_index: int,
        current_cluster_index: int,
    ) -> np.ndarray:

        candidates_by_weight = (
            ChungAislePermutationMutationWeighted.candidates_selector(
                problem,
                skus_in_permutation_cluster,
                current_sku_index,
                current_cluster_index,
            )
        )
        candidates_by_division = (
            ChungAislePermutationWithDivisions.candidates_selector(
                problem,
                skus_in_permutation_cluster,
                current_sku_index,
                current_cluster_index,
            )
        )

        return np.intersect1d(candidates_by_weight, candidates_by_division)
