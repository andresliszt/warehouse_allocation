# -*- coding: utf-8 -*-
"""Manejo de Excepciones."""

import abc
from contextlib import contextmanager
from typing import Any
from typing import Generator
from typing import Type
from typing import Union


class ErrorMixin(abc.ABC, BaseException):
    """Clase base para excepciones propias.

    Example:

        >>> class MyError(ErrorMixin):
                msg_template = "Valor ``{value}`` no es encontrado"
        >>> raise MyError(value="can't touch this")
        (...)
        MyError: Valor `can't touch this` no es encontrado

    """

    @property
    @abc.abstractmethod
    def msg_template(self) -> str:
        """Template para imprimir cuando una excepción es levantada.

        Ejemplo:
            "Valor ``{value}`` no se encuentra "

        """

    def __init__(self, **ctx: Any) -> None:
        self.ctx = ctx
        super().__init__()

    def __str__(self) -> str:
        txt = self.msg_template
        for name, value in self.ctx.items():
            txt = txt.replace("{" + name + "}", str(value))
        txt = txt.replace("`{", "").replace("}`", "")
        return txt


@contextmanager
def change_exception(
    raise_exc: Union[ErrorMixin, Type[ErrorMixin]],
    *except_types: Type[BaseException],
) -> Generator[None, None, None]:
    """Context manager para reemplazar excepciones propias.

    Ver:
        :func:`pydantic.utils.change_exception`

    """
    try:
        yield
    except except_types as exception:
        raise raise_exc from exception  # type: ignore


class EnvVarNotFound(ErrorMixin, NameError):
    """Levantar cuando no se encuentra una variable de entorno."""

    msg_template = "Variable de entorno `{env_var}` no se encuentra"


class InvalidOcurrenceMatrixz(ErrorMixin, NameError):
    """Levantar cuando matriz de occurrencias sea inválida.

    Los algoritmos genéticos estudiados usan una matriz que
    tiene cierta estructura. Este error es para la violación
    de la estructura.

    .

    """

    msg_template = "Matriz de occurrencias invalida. {reason}"


class NoCapacityForSkusError(ErrorMixin, NameError):
    """Levantar cuando la cantidad de skus supera la capacidad."""

    msg_template = "Capacidad no soporta cantidad de skus. Capacidad: `{capacity}`, n_skus: `{n_skus}`"


class NoCapacityForSkusWithDivisionsError(ErrorMixin, NameError):
    """Levantar cuando la cantidad de skus supera la capacidad."""

    msg_template = "Capacidad no soporta skus en algún tipo de división. Capacidad necesitada por division: `{needed_capacity}`"


class NonPositiveParameterError(ErrorMixin, NameError):
    """Levantar cuando alguno de los parámetros no es positivo."""

    msg_template = "Todos los parámetros del modelo deben ser mayor o igual a 0. Parámetro : `{param}`"


class WeightCapacityError(ErrorMixin, NameError):
    """Levantar cuando las capacidades de peso de los clusters no pueden alocar los skus."""

    msg_template = "No es posible alocar los skus en los pasillos debido a la capacidad de pesos. Último peso registrado: `{last_weight}`"


class SkusAndWeightDimensionError(ErrorMixin, NameError):
    """Levantar cuando la cantidad de skus es distinta a la cantidad de pesos."""

    msg_template = "Cantidad de skus no coincide con cantidad de pesos. n_pesos: `{n_weights}`, n_skus: `{n_skus}`."


class ClusterAndWeightToleranceDimensionError(ErrorMixin, NameError):
    """Levantar cuando la cantidad de clusters es distinta a la cantidad de tolerancia de pesos."""

    msg_template = "Cantidad de clusters no coincide con tolerancia de pesos en clusters. n_clusters: `{n_clusters}`, n_wt: `{n_wt}`."


class NoParametersError(ErrorMixin, NameError):
    """Levantar cuando el vector de demandas no coincide con dimensión de matriz de ocurrencias."""

    msg_template = "Uno de los parámetros para el algoritmo genético es un `np.ndarray` vacío."


class SkuAndOCMDimensionError(ErrorMixin, NameError):
    """Levantar cuando el vector de demandas no coincide con dimensión de matriz de ocurrencias."""

    msg_template = "Cantidad de skus no coincide con tamaño de matriz de ocurrencia. m_dimension: `{m_dimension}`, n_skus: `{n_skus}`."


class IncompatibleDivisionAndWeightRestrictionError(ErrorMixin, NameError):
    msg_template = "Incompatibilidad entre clasificación de división y rangos de pesos, sku_idx:`{sku_idx}`"


class LAPNoFeasibleSolutionError(ErrorMixin, NameError):
    """Levantar cuando el vector de demandas no coincide con dimensión de matriz de ocurrencias."""

    msg_template = "Problema LAP no tiene soluciones feasibles"


class ClustersPenalizationsDimensionError(ErrorMixin, NameError):
    """Levantar cuando el vector de penalizaciones no coincide con cantidad de clusters."""

    msg_template = "Dimensión de vector de penalizaciones no coincide con cantidad de clusters. n_clusters: `{n_clusters}`, n_pen: `{n_pen}`."
