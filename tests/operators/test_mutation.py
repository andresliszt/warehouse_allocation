# -*- coding: utf-8 -*-
"""Pruebas unitarias para ``warehouse_allocation.models.operators.mutation``"""


import numpy as np

from warehouse_allocation.operators import (
    ChungAislePermutationMutationWeighted,
)


X_POP_MUTATED = [
    np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]]),
    np.array(
        [
            [0, 1, 0, 1, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 1, 0, 0, 0, 1, 1],
        ]
    ),
]
# Se evalua con los problemas y X_pop PROBLEMS en conftest respetando el orden"""

MUTATION = ChungAislePermutationMutationWeighted(prob=1)
# Mutamos todos los individuos de la población de prueba


def test_mutation(monkeypatch, weighted_problems):
    def mockreturn(iterable):
        return iterable[0]

    # mockreturn sustiuira a random.choice
    # eso nos permite tener control de los skus
    # que se intercambiarán para poder obtener
    # expresiones directas para los assert

    monkeypatch.setattr(np.random, "choice", mockreturn)
    # Cambiamos random.choice por mockreturn

    for prob, X_pop_mutated in zip(weighted_problems, X_POP_MUTATED):
        assert (MUTATION._do(prob.problem, prob.X_pop) == X_pop_mutated).all()
