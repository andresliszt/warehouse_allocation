# -*- coding: utf-8 -*-
"""Utilidades para operadores."""

import abc
from typing import Dict
from typing import List
from typing import Union

import numpy as np
from ortools.graph import pywrapgraph

from warehouse_allocation.exc import LAPNoFeasibleSolutionError


class LAPWithCostFlowAbstract(abc.ABC):
    """Clase abstracta para problema de asignación lineal con flujos para skus.

    La matriz de costos :math:`C = (c_{k,i})`
    representa el costo de asignar el SKU :math:`k` en el cluster
    :math:`i`. La interpretación o definición de esta matriz de
    costo depende netamente de las necesidades de quien usará
    este método.

    Este problema de asignación lineal usa la técnica de optimización
    del flujo mínimo.

    """

    def __init__(self, Cost: np.ndarray, Z: np.ndarray) -> None:
        self.min_cost_flow = pywrapgraph.SimpleMinCostFlow()
        self.Cost = Cost
        self.Z = Z
        self.n_skus = Cost.shape[1]
        self.n_clusters = len(Z)
        self._mapping_clusters = {}

    @property
    def source_node(self):
        return 0

    @property
    @abc.abstractmethod
    def sink_node(self) -> int:
        """Id del nodo sink del grafo."""

    def arc_has_flow(self, arc: int) -> bool:
        """Si un arco que conecta sku con cluster tiene flujo o no."""
        return (
            self.min_cost_flow.Tail(arc) != self.source_node
            and self.min_cost_flow.Head(arc) != self.sink_node
            and self.min_cost_flow.Flow(arc) > 0
        )

    @abc.abstractmethod
    def is_allowed(self, sku_idx, cluster_idx) -> bool:
        """``True`` si ``sku_idx`` puede ser alocado en ``cluster_idx``."""

    @abc.abstractmethod
    def cluster_node_id(self, *args, **kwargs) -> int:
        """Id del cluster ``cluster_idx`` en el grafo."""

    @abc.abstractmethod
    def arc_sku_cluster(self, cluster_idx: int, sku_node_id: int):
        """Construye arco entre cluster y sku.

        Debe actualizar ``self.min_cost_flow``.

        """

    def connect_sku_with_clusters(self) -> None:
        """Conecta skus con clusters permitidos.

        Actualiza ``self.min_cost_flow``.

        """

        for sku_node_id in range(1, self.n_skus + 1):
            self.min_cost_flow.AddArcWithCapacityAndUnitCost(
                self.source_node, sku_node_id, 1, 0
            )
            for cluster_idx in range(self.n_clusters):
                if not self.is_allowed(sku_node_id - 1, cluster_idx):
                    continue
                self.min_cost_flow.SetNodeSupply(sku_node_id, 0)
                self.arc_sku_cluster(cluster_idx, sku_node_id)

    @abc.abstractmethod
    def connect_clusters_with_sink(self):
        """Conecta todos los clusters con el nodo ``self.sink_node``."""

    def build_graph(self) -> None:
        """Construye el grafo para resolver problema de flujo mínimo."""
        self.min_cost_flow.SetNodeSupply(self.source_node, self.n_skus)
        self.connect_sku_with_clusters()
        self.connect_clusters_with_sink()
        self.min_cost_flow.SetNodeSupply(self.sink_node, -self.n_skus)

    def decode_solution(self) -> np.ndarray:
        """Decodifica solución en individuo slotting."""
        ind = np.zeros((self.n_clusters, self.n_skus))

        for arc in range(self.min_cost_flow.NumArcs()):
            if self.arc_has_flow(arc):
                ind[self._mapping_clusters[self.min_cost_flow.Head(arc)]][
                    self.min_cost_flow.Tail(arc) - 1
                ] = 1

        return ind.astype(bool)

    def solve(self) -> np.ndarray:
        """Resuelve problema de flujo mínimo y retorna individuo slotting."""
        self.build_graph()
        if self.min_cost_flow.Solve() != self.min_cost_flow.OPTIMAL:
            raise LAPNoFeasibleSolutionError
        return self.decode_solution()


class LAPWithCostFlow(LAPWithCostFlowAbstract):
    """Problema de asignación lineal sin restricciones de clusters.

    En este problema todo sku es permitido en todo cluster. Este
    problema es útil en el contexto de slotting cuando los clusters no
    tienen restricciones de peso ni existen rotaciones.

    """

    @property
    def sink_node(self) -> int:
        return self.n_skus + self.n_clusters + 2

    def is_allowed(self, sku_idx: int, cluster_idx: int) -> bool:
        """En este problema todo sku puede ir en cualquier cluster."""
        return True

    def cluster_node_id(self, cluster_idx: int) -> int:
        cluster_node_id = cluster_idx + self.n_skus + 1
        self._mapping_clusters[cluster_node_id] = cluster_idx
        return cluster_node_id

    def arc_sku_cluster(self, cluster_idx, sku_node_id):
        self.min_cost_flow.AddArcWithCapacityAndUnitCost(
            sku_node_id,
            self.cluster_node_id(cluster_idx),
            1,
            int(self.Cost[cluster_idx][sku_node_id - 1]),
        )

    def connect_clusters_with_sink(self) -> None:
        for cluster_idx in range(self.n_clusters):
            cluster_node_id = self.cluster_node_id(cluster_idx)
            self.min_cost_flow.SetNodeSupply(cluster_node_id, 0)
            self.min_cost_flow.AddArcWithCapacityAndUnitCost(
                cluster_node_id,
                self.sink_node,
                int(self.Z[cluster_idx]),
                0,
            )


class LAPWithCostFlowRestrictedClusters(LAPWithCostFlow):

    """Problema de asignación lineal con restricciones de clusters.

    En este problema cada sku tiene una cantidad pre fijada de clusters
    donde puede ser alocado. Las llaves del diccionario
    ``allowed_clusters_by_skus`` corresponden a los índices de los
    skus y cada valor asociado es una lista de clusters en los cuales
    el sku es permitido de alocar.

    Este problema es útil en el contexto de slotting
    cuando los clusters tienen restricciones de peso.

    """

    def __init__(
        self,
        Cost: np.ndarray,
        Z: np.ndarray,
        allowed_clusters_by_sku: Dict[int, List[int]],
    ) -> None:
        super().__init__(Cost=Cost, Z=Z)
        self.allowed_clusters_by_sku = allowed_clusters_by_sku

    def is_allowed(self, sku_idx: int, cluster_idx: int) -> bool:
        return cluster_idx in self.allowed_clusters_by_sku[sku_idx]


class LAPWithCostFlowRestrictedClustersPlusDivision(LAPWithCostFlowAbstract):

    """Problema de asignación lineal con restricciones de rotación.

    En este problema cada sku tiene asignado un tipo de *división* dado
    por el vector ``division_types``. La entrada i-ésima de dicho vector
    contiene la clasificación de división para el sku i-ésimo, y
    consiste en una enumeración 0, 1, 2, etc. Las capacidades ``Z``
    ahora corresponden a una matriz de tantas filas como clusters y
    tantas columnas como sea la enumeración máxima en
    ``division_types``. La interpretación en el contexto de slotting es
    la siguiente: Hay skus en ciertas áreas de los centros de
    ditribución que ocupan media ubicación, un tercio de ubicación, etc
    (el área con más divisiones por el momento es valiosos ``PVAM`` del
    centro de distribución Noviciado ``E070`` donde skus llegan a ocupar
    un doceavo de ubicación.). Generalmente el valor ``0`` en
    ``division_types`` significa que el sku ocupa 1 slot completo,
    ``1``la división siguiente (como un medio de slot o un tercio), y
    así sucesivamente. Por ende, la entrada ``(j,k)`` de ``Z``,
    representa la cantidad de skus del tipo división ``k`` que puede
    alocar el cluster ``j``.

    """

    def __init__(
        self,
        Cost: np.ndarray,
        Z: np.ndarray,
        division_types: np.ndarray,
        allowed_clusters_by_sku: Dict[int, List[int]],
    ) -> None:

        super().__init__(Cost=Cost, Z=Z)

        self.division_types = division_types
        self.unique_division_types = np.unique(division_types)
        self.allowed_clusters_by_sku = allowed_clusters_by_sku

    @property
    def sink_node(self) -> int:
        return self.n_skus + self.n_clusters * self.Z.shape[1] + 2

    def is_allowed(self, sku_idx: int, cluster_idx: int) -> bool:
        return cluster_idx in self.allowed_clusters_by_sku[sku_idx]

    def cluster_node_id(
        self, cluster_idx: int, div_type: Union[np.int32, int]
    ) -> int:
        cluster_node_id = (
            cluster_idx + self.n_skus + self.n_clusters * int(div_type) + 1
        )
        self._mapping_clusters[cluster_node_id] = cluster_idx
        return cluster_node_id

    def arc_sku_cluster(self, cluster_idx, sku_node_id):
        self.min_cost_flow.AddArcWithCapacityAndUnitCost(
            sku_node_id,
            self.cluster_node_id(
                cluster_idx, self.division_types[sku_node_id - 1]
            ),
            1,
            int(self.Cost[cluster_idx][sku_node_id - 1]),
        )

    def connect_clusters_with_sink(self) -> None:
        for cluster_idx in range(self.n_clusters):
            for div_type in self.unique_division_types:
                cluster_node_id = self.cluster_node_id(cluster_idx, div_type)
                self.min_cost_flow.SetNodeSupply(cluster_node_id, 0)
                self.min_cost_flow.AddArcWithCapacityAndUnitCost(
                    cluster_node_id,
                    self.sink_node,
                    int(self.Z[cluster_idx][div_type]),
                    0,
                )
