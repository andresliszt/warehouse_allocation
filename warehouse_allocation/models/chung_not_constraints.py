import abc

import numpy as np

from warehouse_allocation.models.chung import ChungProblem
from warehouse_allocation.models.chung import ChungProblemStrictlyWeighted
from warehouse_allocation.models.chung import ChungProblemWithDivisions
from warehouse_allocation.models.chung import ChungProblemWithDivisionsPlusWeight

# args, kwargs unused, pero el constructor de pymoo los llama
# no eliminar nunca aunque sean redundantes

# pylint: disable=unused-argument


class NotConstraints(abc.ABC):
    """Evaluación sin constraints para problemas de ``warehouse_allocation``

    Simplemente sobre escribe el método ``_evaluate`` presente en todos
    los problemas de ``warehouse_allocation``, eliminando el llamado al cálculo
    de constraints.

    """

    @property
    def _n_constr(self) -> int:
        return 0

    def _evaluate(
        self, x: np.ndarray, out: np.ndarray, *args, **kwargs
    ) -> None:

        out["F"] = self.bi_objective_function(x)  # pylint: disable=no-member


class ChungProblemNotConstraints(NotConstraints, ChungProblem):
    """Problema original de Chung sin constraints."""


class ChungProblemStrictlyWeightedNotConstraints(
    NotConstraints, ChungProblemStrictlyWeighted
):
    """Problema de Chung con clusters con tolerancia de pesos sin restricciones."""


class ChungProblemWithDivisionsNotConstraints(
    NotConstraints, ChungProblemWithDivisions
):
    """Problema de Chung considerando divisiones sin constraints."""


class ChungProblemWithDivisionsPlusWeightNotConstraints(
    NotConstraints, ChungProblemWithDivisionsPlusWeight
):
    """Problema de Chung considerando divisiones y pesos sin constraints."""
