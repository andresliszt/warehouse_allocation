# -*- coding: utf-8 -*-
"""Modelo de Chung de un solo objetivo."""

from warehouse_allocation.models.bases import ChungProblemSingleObjectiveBase
from warehouse_allocation.models.bases import ChungProblemSingleObjectiveTrafficBase


class ChungProblemSO(ChungProblemSingleObjectiveBase):
    """Problema de Chung de un solo objetivo sin considerar tráfico."""

    @property
    def _n_constr(self) -> int:
        return self.n_clusters + self.n_skus


class ChungProblemSOTraffic(ChungProblemSingleObjectiveTrafficBase):
    """Problema de Chung con un solo objetivo considerarando tráfico."""

    # TODO: Cierto validador para W_max

    @property
    def _n_constr(self) -> int:
        return 2 * self.n_clusters + self.n_skus
