# -*- coding: utf-8 -*-
"""Pruebas unitarias para ``warehouse_allocation.models.operators.sampling``"""

import numpy as np

from warehouse_allocation.operators import NonDominatedSortingSamplingWeighted

SAMPLER = NonDominatedSortingSamplingWeighted()


def test_sampling(weighted_problems):
    """Test sampling, individuos deben respetar constraints del problema"""

    for prob in weighted_problems:
        X_pop = SAMPLER._do(prob.problem, n_samples=10)
        assert np.all(prob.problem.constraints(X_pop) <= 0)
