# -*- coding: utf-8 -*-
"""Modelo de Chung múltiple objetivo."""


from collections import Counter
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np

from warehouse_allocation import logger
from warehouse_allocation.exc import ClusterAndWeightToleranceDimensionError
from warehouse_allocation.exc import IncompatibleDivisionAndWeightRestrictionError
from warehouse_allocation.exc import LAPNoFeasibleSolutionError
from warehouse_allocation.exc import NoCapacityForSkusWithDivisionsError
from warehouse_allocation.exc import NonPositiveParameterError
from warehouse_allocation.exc import SkusAndWeightDimensionError
from warehouse_allocation.models.bases import ChungProblemMultiObjectiveBase


class ChungProblem(ChungProblemMultiObjectiveBase):
    """Problema de Chung idéntico al paper original."""

    @property
    def _n_constr(self) -> int:
        return self.n_clusters + self.n_skus

    @property
    def _n_obj(self) -> int:
        return 2


class ChungProblemStrictlyWeighted(ChungProblemMultiObjectiveBase):
    """Problema de Chung con pesos en constraints por cluster.

    Esta clase incorpora nuevas restricciones asociadas al peso. Cada
    cluster define una toleracia en peso, es decir, peso mínimo y máximo
    sobre los SKUs que va recibir. Cada SKU es alocado en un cluster si
    cumple todas las restriciones del paper de Chung, en adición a esta
    nueva.

    """

    def __init__(
        self,
        W: Union[List[float], np.ndarray],
        WT: Union[List[Tuple[float, float]], np.ndarray],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        :param W: Vector de pesos de los SKUs.
        :param WT: Tolerancia en pesos de los clusters.
            La convención es una lista de tuplas de la forma
            :math:`(w_{\text{k,min}}, w_{\text{k,max}})`, con
            :math:`k` moviendose en la indexación de los clusters.
        *args: Args de :class:`~warehouse_allocation.models.bases.ChungProblemCompleteBase`
        *kargs: Kwargs de :class:`~warehouse_allocation.models.bases.ChungProblemCompleteBase`.

        """

        super().__init__(*args, **kwargs)

        WT = np.array(WT)

        if WT.shape[0] != self.n_clusters:
            raise ClusterAndWeightToleranceDimensionError(
                n_clusters=self.n_clusters, n_wt=WT.shape[0]
            )
        if WT.shape[1] != 2:
            raise TypeError(
                "Vector `WT` debe ser un `ndarray` de shape `(n_clusters, 2)` o lista de largo `n_clusters` formada de 2-tuplas"
            )
        if not all(0 <= w[0] < w[1] for w in WT):
            raise TypeError(
                "Las tuplas de pesos deben ser del estilo `(w_min, w_max)` y la cota inferior es 0"
            )
        if not all(w > 0 for w in W):
            raise NonPositiveParameterError(param="weights")
        if self.n_skus != len(W):
            raise SkusAndWeightDimensionError(
                n_weights=len(W), n_skus=self.n_skus
            )

        self.WT = WT
        self.W = np.array(W)
        self.allowed_clusters_by_sku_matrix = (
            self.WT[:, 0][:, np.newaxis] <= self.W
        ) & (self.W <= self.WT[:, 1][:, np.newaxis]).astype(int)
        # Matriz binaria para escribir restricciones de peso
        # Matriz de tamaño N_cluster x N_SKUS, donde
        # la entrada (i,j) es True si el SKU j pertenece
        # al cluster i, y False en otro caso
        self.allowed_clusters_by_sku = (
            self._allowed_clusters_by_sku_using_weight
        )

        self._validate()

        # Es necesario considerar el diccionario cuya llave
        # es el índice del SKU y el valor es la lista
        # de índice de clusters en los que puede ser alocado
        # debido a la restricción de peso. Este diccionario
        # es útil para los operadores de mating, sobre todo para
        # warehouse_allocation.models.operators.crossover.ChungPartiallyMappedCrossover
        # que usa ortools, y esta librería no es compatible con numpy

    def _validate(self):
        try:
            # A este punto si el problema LAP falla, es porque los rangos
            # de pesos de los clusters no están bien definidos.
            from warehouse_allocation.operators.lap import (
                LAPWithCostFlowRestrictedClusters,
            )

            LAPWithCostFlowRestrictedClusters(
                Cost=np.zeros((self.n_clusters, self.n_skus)),
                Z=self.Z,
                allowed_clusters_by_sku=self.allowed_clusters_by_sku,
            ).solve()
        except LAPNoFeasibleSolutionError as e:
            logger.error(
                "Rangos de pesos por clusters no son compatibles con pesos de skus"
            )
            raise e

    @property
    def _n_constr(self) -> int:
        return self.n_clusters + 2 * self.n_skus

    @property
    def _n_obj(self) -> int:
        return 2

    @property
    def _allowed_clusters_by_sku_using_weight(self) -> Dict[int, List[int]]:
        return {
            sku_idx: list(
                np.flatnonzero(self.allowed_clusters_by_sku_matrix[:, sku_idx])
            )
            for sku_idx in range(self.n_skus)
        }

    def __individual_weight(self, ind: np.ndarray) -> np.ndarray:
        return (
            np.einsum(
                "ij, ij -> j",
                self.allowed_clusters_by_sku_matrix,
                self.matrix_individual(ind),
            )
            - 1
        )

    def sku_weight_constraint(
        self,
        X_pop: np.ndarray,
    ) -> np.ndarray:
        """Restricción de pesos para alocación de SKUs en clusters.

        :param X_pop: Matriz de indiviuos y variables.
        :return:
            Evaluación de las restricciones por cada individuo.

        """

        return np.apply_along_axis(self.__individual_weight, 1, X_pop)

    def constraints(self, X_pop: np.ndarray) -> np.ndarray:
        """Todas las constraints de :cite:t:`2019:chung` más restricción de peso.

        :param X_pop: X_pop: Matriz de indiviuos y variables.

        :return:
            Evaluación de las restricciones por cada individuo.
        :rtype: np.ndarray

        """

        return np.concatenate(
            (
                super().constraints(X_pop),
                np.abs(self.sku_weight_constraint(X_pop))
                - self.EPSILON_VALUE_FOR_EQUALITY_CONSTRAINT,
            ),
            axis=1,
        )


class ChungProblemWithDivisions(ChungProblem):
    """Problema de Chung con restricciones de división sobre skus.

    Este problema considera que los skus están clasificados
    por un valor de *división* dados por el vector
    ``division_types``. La capacidad de cada cluster es modelada
    ya no como un valor entero, si no como un vector de largo
    igual al número de valores únicos en ``division_types``, donde
    cada entrada corresponde a la capacidad del cluster para dicho
    valor de división.


    """

    def __init__(self, division_types: np.ndarray, *args, **kwargs) -> None:

        self.division_types = np.array(division_types)
        self.unique_division_types = np.unique(self.division_types)

        super().__init__(*args, **kwargs)

        if len(self.unique_division_types) != self.Z.shape[1]:
            raise NotImplementedError(
                "Cantidad de columnas de ``Z`` debe coincidir con valores únicos de ``division_types``"
            )
        self.allowed_clusters_by_sku = (
            self._allowed_clusters_by_sku_using_divisions
        )

        if (
            self.Z.sum(axis=0)
            - self.__unique_counts(Counter(self.division_types))
            < 0
        ).any():
            raise NoCapacityForSkusWithDivisionsError(
                needed_capacity=self.__unique_counts(
                    Counter(self.division_types)
                )
            )
        self._validate()

    def _validate(self):
        try:
            from warehouse_allocation.operators.lap import (
                LAPWithCostFlowRestrictedClustersPlusDivision,
            )

            LAPWithCostFlowRestrictedClustersPlusDivision(
                Cost=np.zeros((self.n_clusters, self.n_skus)),
                Z=self.Z,
                allowed_clusters_by_sku=self.allowed_clusters_by_sku,
                division_types=self.division_types,
            ).solve()
        except LAPNoFeasibleSolutionError as e:
            logger.error(
                "Incompatibilidad de capacidad de clusters con divisiones"
            )
            raise e

    @property
    def _n_constr(self) -> int:
        return self.n_clusters * len(self.unique_division_types) + self.n_skus

    @property
    def _allowed_clusters_by_sku_using_divisions(self) -> Dict[int, List[int]]:

        return {
            sku_idx: list(
                np.flatnonzero(self.Z[:, self.division_types[sku_idx]] > 0)
                # de acuerdo al tipo de division del sku
                # solo será admisible en un cluster, si es que
                # este tiene capacidad > 0 para dicha división
            )
            for sku_idx in range(self.n_skus)
        }

    def __unique_counts(self, cnt: Counter) -> List[int]:
        return [cnt[dv_type] for dv_type in self.unique_division_types]

    def __count_divisions(self, cluster: np.array):
        cnt = Counter(self.division_types[np.flatnonzero(cluster)])
        return self.__unique_counts(cnt)

    def clusters_usage_matrix_individual(
        self, matrix_individual: np.ndarray
    ) -> np.ndarray:
        """Retorna la cantidad posiciones usada por individuo en forma matricial."""
        return np.apply_along_axis(
            self.__count_divisions, 1, matrix_individual
        )

    def aviable_location_in_clusters(self, ind: np.ndarray) -> np.ndarray:
        """Restricción de capacidad de alocación por cluster de un individuo.

        Este cálculo se hace para evaluar la constraint en ``pymoo``.

        """

        return (
            self.clusters_usage_matrix_individual(self.matrix_individual(ind))
            - self.Z
        ).T.flatten()

    def cluster_storage_capacity_constraint(
        self, X_pop: np.ndarray
    ) -> np.ndarray:
        """Restricción de capacidad de alocación por cluster poblacional.

        Este cálculo se hace para evaluar la constraint en ``pymoo``.

        """

        return np.apply_along_axis(self.aviable_location_in_clusters, 1, X_pop)


class ChungProblemWithDivisionsPlusWeight(
    ChungProblemWithDivisions, ChungProblemStrictlyWeighted
):
    """Problema de Chung con restricciones de división y pesos sobre clusters."""

    def __init__(self, division_types: np.ndarray, *args, **kwargs) -> None:

        # Observación: Segun el MRO de python, este __init__
        # primero llama al __init__ ChungProblemWithDivisions,
        # pero como en este hay una llamada del super() también,
        # pasa a llamar el __init__ de ChungProblemStrictlyWeighted
        # en ese super call. Por ende el ordén es: Inicia el __init__
        # de ChungProblemWithDivisions, inicia el __init__ de
        # ChungProblemStrictlyWeighted, finaliza el __init__
        # de ChungProblemStrictlyWeighted, y finaliza el
        # __init__ ChungProblemStrictlyWeighted

        super().__init__(division_types, *args, **kwargs)

        self.allowed_clusters_by_sku = self.__intersect_allowed_clusters(
            self.allowed_clusters_by_sku,  # Este es el atributo es de las divisiones
            self._allowed_clusters_by_sku_using_weight,
        )

    def _validate(self):
        # Simplemente sobreescribe _validate de super()
        # No es necesario llamar en este __init__, pues
        # se hace en el super() call
        try:
            from warehouse_allocation.operators.lap import (
                LAPWithCostFlowRestrictedClustersPlusDivision,
            )

            LAPWithCostFlowRestrictedClustersPlusDivision(
                Cost=np.zeros((self.n_clusters, self.n_skus)),
                Z=self.Z,
                allowed_clusters_by_sku=self.allowed_clusters_by_sku,
                division_types=self.division_types,
            ).solve()
        except LAPNoFeasibleSolutionError as e:
            logger.error(
                "Incompatibilidad de capacidad de clusters con divisiones y/o pesos."
            )
            raise e

    @property
    def _n_constr(self) -> int:
        return self.n_clusters * len(self.unique_division_types) + self.n_skus

    @staticmethod
    def __raise_intersection(sku_idx: int, set1: set, set2: set) -> List[int]:
        intersection = set1.intersection(set2)
        if intersection == set():
            raise IncompatibleDivisionAndWeightRestrictionError(
                sku_idx=sku_idx
            )
        return list(intersection)

    def __intersect_allowed_clusters(
        self,
        allowed_using_weights: Dict[int, List[int]],
        allowed_using_divisions: Dict[int, List[int]],
    ) -> Dict[int, List[int]]:
        return {
            sku_idx: self.__raise_intersection(
                sku_idx,
                set(allowed_using_weights[sku_idx]),
                set(allowed_using_divisions[sku_idx]),
            )
            for sku_idx in allowed_using_weights
        }
