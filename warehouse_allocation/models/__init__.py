# -*- coding: utf-8 -*-
"""API modelos/problemas de sltotings."""

from warehouse_allocation.models.chung import ChungProblem
from warehouse_allocation.models.chung import ChungProblemStrictlyWeighted
from warehouse_allocation.models.chung import ChungProblemWithDivisions
from warehouse_allocation.models.chung import ChungProblemWithDivisionsPlusWeight
from warehouse_allocation.models.chung_not_constraints import (
    ChungProblemNotConstraints,
)
from warehouse_allocation.models.chung_not_constraints import (
    ChungProblemStrictlyWeightedNotConstraints,
)
from warehouse_allocation.models.chung_not_constraints import (
    ChungProblemWithDivisionsNotConstraints,
)
from warehouse_allocation.models.chung_not_constraints import (
    ChungProblemWithDivisionsPlusWeightNotConstraints,
)

__all__ = (
    "ChungProblem",
    "ChungProblemStrictlyWeighted",
    "ChungProblemWithDivisions",
    "ChungProblemWithDivisionsPlusWeight",
    "ChungProblemNotConstraints",
    "ChungProblemStrictlyWeightedNotConstraints",
    "ChungProblemWithDivisionsNotConstraints",
    "ChungProblemWithDivisionsPlusWeightNotConstraints",
)
