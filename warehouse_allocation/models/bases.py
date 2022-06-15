# -*- coding: utf-8 -*-
"""Implementación del problema de Chung usando pymoo."""

import abc
from typing import Any
from typing import List
from typing import Optional
from typing import Union

import numpy as np
from pymoo.core.problem import Problem

from warehouse_allocation.exc import ClustersPenalizationsDimensionError
from warehouse_allocation.exc import InvalidOcurrenceMatrixz
from warehouse_allocation.exc import NoCapacityForSkusError
from warehouse_allocation.exc import NonPositiveParameterError
from warehouse_allocation.exc import NoParametersError
from warehouse_allocation.exc import SkuAndOCMDimensionError

# _evaluate method tien *args, **kwargs que con no utilizados
# implicitamente en este modulo, sin embargo son utilizados
# internamente por ``pymoo``. No eliminar nunca.


class ChungProblemBase(Problem, abc.ABC):
    """Implementación del paper de Chung en ``pymoo``.

    En esta clase se definen los método bases para el problema de
    optimización que es propuesto en el paper de Chung. El método de
    resolver será basado en el enfoque matricial que ofrece ``pymoo``.
    Esta clase sirve para el problema multi objetivo como también para
    el problema de un solo objetivo.

    """

    # TODO: Hay muchas llamadas a matrix_individual,
    # Tal vez mejor aplicar matrix_individual a X_pop

    EPSILON_VALUE_FOR_EQUALITY_CONSTRAINT = 1e-1
    """Constante para definir constraint de igualdad en ``pymoo``"""

    def __init__(
        self,
        Z: Union[np.ndarray, List[int]],
        OCM: np.ndarray,
        clusters_penalizations: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> None:
        """
        :param Z:
                Lista conteniendo la capacidad por tipos
                de skus. Siguiendo la notación de Chung,
                corresponden a los valores :math:`Z_k`.
        :param OCM:
                Matriz de ocurrencia en la cual cada entrada
                :math:`(j,j')` corresponde a la cantidad de veces
                que los skus j y j' han aparecido juntos
                en ordenes de pedidos. Siguiendo la notación
                de Chung, cada entrada corresponde a el
                valor :math:`N_{j,j'}`.
        """

        Z = np.array(Z)
        OCM = np.array(OCM)

        if clusters_penalizations is None:
            self.clusters_penalizations = np.ones(len(Z))
        else:
            if len(clusters_penalizations) != len(Z):
                raise ClustersPenalizationsDimensionError(
                    n_pen=len(clusters_penalizations), n_clusters=len(Z)
                )
            self.clusters_penalizations = clusters_penalizations

        if not np.greater_equal(Z, 0).all():
            raise NonPositiveParameterError(param="clusters-cap")

        if not np.greater_equal(OCM, 0).all():
            raise NonPositiveParameterError(param="occurrence-matrix")

        if not (np.diagonal(OCM) == 0).all():
            raise InvalidOcurrenceMatrixz(
                reason="Diagonal contiene valores distintos a 0."
            )

        if (OCM == OCM.T).all():
            # Si la matriz es simétrica, la hacemos triangular superior
            OCM = np.triu(OCM)
        elif (OCM == np.triu(OCM)).all():
            # Si la matriz ya es triangular superior, no hacemos nada
            pass

        elif (OCM == np.tril(OCM)).all():
            # Trabajamos con matriz triangular superior siempre
            OCM = OCM.T

        else:
            raise InvalidOcurrenceMatrixz(
                reason="Matriz no es simétrica ni tríangular superior"
            )

        if OCM.shape[0] > Z.sum():
            raise NoCapacityForSkusError(capacity=Z.sum(), n_skus=OCM.shape[0])

        if Z.size == 0 or OCM.size == 0:
            raise NoParametersError
            # Este validador aunque muy inocente que parezca, es útil para tener
            # un error controlado para algunas situaciones. Por ejemplo, si los parámetros
            # son extraídos desde una query a una base de datos, si esta query no protege
            # los valores cuando no hay match, entrarán a esta clase valores vacíos de parámetros
            # Basta validar estos dos parámetros solamente, pues en las subclases hay validarores
            # de dimensiones, que cubren el caso de que si otros parámetros distintos a estos dos
            # vienen vacíos

        self.Z = Z
        self.OCM = OCM
        self.n_clusters = len(Z)
        self.n_skus = OCM.shape[0]

        super().__init__(
            n_var=self.n_skus * self.n_clusters,
            n_obj=self._n_obj,
            n_constr=self._n_constr,
            type_var=np.bool8,
            xl=0,
            xu=1,
            **kwargs,
        )

    @property
    @abc.abstractmethod
    def _n_obj(self):
        """Número de objetivos que tiene el problema a resolver."""

    @property
    @abc.abstractmethod
    def _n_constr(self):
        """Número de constraints que tiene el problema a resolver."""

    def matrix_individual(self, ind: np.ndarray) -> np.ndarray:
        """Herramienta para hacer un reshape de lista a matriz.

        En el problema de optimización que estamos intentando
        resolver, la definición de las variables involucra
        una formulación usando multi índices, esto significa
        que las variables son de la forma :math:`x_{i,k}`. Hay muchas
        restricciónes que se hacen fijando un índice i y variando
        el índice k (y viceversa). Las variables binarias son
        de tamaño :math:`Q\\times C`, donde :math:`Q` es el total
        de skus y :math:`C` es el total de clusters.
        En nuestra formulación el orden de las variables está dado
        por la siguiente forma:
        :math:`[x_{1,1}, ..., x_{Q,1}, ..., x_{1,C}, ..., x_{Q,C}]`.

        Este método hace un *reshape* para obtener la representación
        matricial del individuo :math:`I = (x_{k,j})\\in\\mathbb{M}^{C\\times Q}`.
        Notar que en la forma matricial las **filas** corresponden a los
        clusters y la columnas a los SKUs.

        :param ind:
                Representación vectorial del individuo.

        :return:
                Array con la representación matricial del individuo.

        """

        return (
            np.array(ind)
            .reshape((int(len(ind) / self.n_skus), self.n_skus))
            .astype(bool)
        )

    def __individual_in_one_cluster(self, ind: np.ndarray) -> np.ndarray:
        return self.matrix_individual(ind).T.sum(axis=1) - 1

    @staticmethod
    def clusters_usage_matrix_individual(
        matrix_individual: np.ndarray,
    ) -> np.ndarray:
        """Uso de los clusters por individuo en forma matricial.

        Este método es lo más atómico posible debido a que es
        usado en el operador genético de crossover
        :py:mod:`warehouse_allocation.operators.crossover`. Este
        método es sobreescrito en
        :class:`~warehouse_allocation.models.chung.ChungProblemWithDivisions`,
        pues ``Z`` deja de ser un vector y se convierte en una matriz.

        """
        return matrix_individual.sum(axis=1)

    def sku_only_in_one_cluster_constraint(
        self,
        X_pop: np.ndarray,
    ) -> np.ndarray:
        """Restricción (3) del paper de Chung.

        :param X_pop: Matriz de indiviuos y variables.
        :return:
            Evaluación de las restricciones por cada individuo.

        """

        return np.apply_along_axis(self.__individual_in_one_cluster, 1, X_pop)

    def __individual_storage_capacity(self, ind):
        return (
            self.clusters_usage_matrix_individual(self.matrix_individual(ind))
            - self.Z
        )

    def cluster_storage_capacity_constraint(
        self, X_pop: np.ndarray
    ) -> np.ndarray:
        """Restricción (4) of Chung's paper.

        :param X_pop: Matriz de indiviuos y variables.
        :return:
            Evaluación de las restricciones por cada individuo.

        """

        return np.apply_along_axis(
            self.__individual_storage_capacity, 1, X_pop
        )

    @staticmethod
    def __Quadratic(var, M):
        return np.dot(var, np.matmul(M, var))

    def __weighted_sum(self, clusters_values):
        return np.sum(
            np.multiply(self.clusters_penalizations, clusters_values)
        )

    def __sum_Quadratic(self, ind):
        return self.__weighted_sum(
            np.apply_along_axis(
                self.__Quadratic, 1, self.matrix_individual(ind), self.OCM
            )
        )

    def Q(self, X_pop: np.ndarray) -> np.ndarray:
        """Función objetivo de afinidad, :math:`f_1` en :cite:t:`2019:chung`

        :param X_pop: Matriz de indiviuos y variables.
        :param M: Matriz que representa la forma cuadrática.

        """

        return -np.apply_along_axis(self.__sum_Quadratic, 1, X_pop)

    def constraints(self, X_pop: np.ndarray) -> np.ndarray:
        """Constraints base de :cite:t:`2019:chung` para input en ``pymooo``

        Este método incorpora las constraints (3) y (4) del paper de
        Chung.

        :param X_pop: Matriz de indiviuos y variables.

        """

        return np.concatenate(
            (
                self.cluster_storage_capacity_constraint(X_pop),
                np.abs(self.sku_only_in_one_cluster_constraint(X_pop))
                - self.EPSILON_VALUE_FOR_EQUALITY_CONSTRAINT,
            ),
            axis=1,
        )

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        """Evalua el problema según construcción de ``pymoo``

        En este método se definen las funciones objetivos y los
        constraints. La sintaxís de implementación es netamente debido a
        ``pymoo``

        """

        out["F"] = self.Q(x)
        out["G"] = self.constraints(x)


class ChungProblemCompleteBase(ChungProblemBase):
    """Clase base para el problema de Chung completo.

    Esta clase aborda las condiciones originales del paper
    de Chung, la diferencia con la clase
    :class:`~warehouse_allocation.models.bases.ChungProblemBase` es que esta clase
    considera el vector de demandas :math:`D`, que sirve para tener
    un control del flujo de demananda sobre los clusters.


    Es muy importante hacer notar que en esta clase no se define la constraint
    (5) del paper de Chung, esto es, pues :math:`W_max`, visto como objetivo,
    corresponde a la función identidad, en cada iteración este valor
    se actualiza calculando la demanda máxima sobre los clusters de un individuo
    implicando que la constraint se cumpla trivialmente. Cuando :math:`W_max`
    es fijo (un parámetro) y no parte del objetivo, si es necesario
    considerar dicha desigualdad, tal cual se define en
    :class:`warehouse_allocation.models.bases.ChungProblemSingleObjectiveTrafficBase`,
    la cual es la base para tratar el problema de Chung con un solo objetivo,
    y el parámetro :math:`W_max` es fijo.

    """

    # TODO: Validar Demanda vs Matriz de afinidad

    def __init__(
        self, D: Union[np.ndarray, List[float]], *args: Any, **kwargs: Any
    ) -> None:
        """

        :param D: Vector de demandas de los SKUs.
        :param args: Corresponden a los ``args`` de :class:`~warehouse_allocation.models.bases.ChungProblemBase`
        :param kwargs: Corresponden a los ``kwargs`` de :class:`~warehouse_allocation.models.bases.ChungProblemBase`

        """

        D = np.array(D)

        if D.size == 0:
            # Previene si la ingesta viene de una query sin resultados
            raise NoParametersError

        if not np.greater_equal(D, 0).all():
            raise NonPositiveParameterError(param="demand")

        self.D = D
        super().__init__(*args, **kwargs)

        if self.OCM.shape[0] != len(self.D) or self.OCM.shape[1] != len(
            self.D
        ):
            raise SkuAndOCMDimensionError(
                m_dimension=self.OCM.shape, n_skus=len(self.D)
            )


class ChungProblemSingleObjectiveBase(ChungProblemBase):
    """Clase base para el problema de Chung con un objetivo.

    Esta clase se usará para resolver el problema sin considerar la
    restricción de tráfico en el problema de un solo objetivo.

    """

    @property
    def _n_obj(self):
        return 1


class ChungProblemSingleObjectiveTrafficBase(ChungProblemCompleteBase):
    """Clase base para el problema de Chung con un objetivo con tráfico.

    Esta clase se usará para resolver el problema considerando la
    restricción de tráfico no como un objetivo (paper original) si no
    como un valor fijo.

    """

    def __init__(self, W_max: float, *args, **kwargs) -> None:
        """
        :param W_max: Tolerancia de demanda máxima que los pasillos admiten.
        *args: Args de :class:`~warehouse_allocation.models.bases.ChungProblemCompleteBase`
        *kargs: Kwargs de :class:`~warehouse_allocation.models.bases.ChungProblemCompleteBase`

        """

        if not W_max > 0:
            raise ValueError("Si `W_max` debe ser estríctamente positivo.")

        super().__init__(*args, **kwargs)
        self.W_max = W_max

    @property
    def _n_obj(self) -> int:
        return 1

    def __cluster_frequency(self, clus: np.ndarray) -> float:
        return np.inner(clus, self.D)

    def __individual_pickup_frequency(self, ind: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(
            self.__cluster_frequency, 1, self.matrix_individual(ind)
        )

    def cluster_pickup_frequency_constraint(
        self, X_pop: np.ndarray
    ) -> np.ndarray:
        """Restricción (5) de :cite:t:`2019:chung`.

        :param X_pop: Matriz de indiviuos y variables binarias.
        :param W_max:
                Vector con valores de la variable W_max
                que se define en el paper de Chung.
        :return:
            Evaluación de las restricciones por cada individuo.

        """
        W_max = self.W_max * np.ones(len(X_pop))

        return (
            np.apply_along_axis(self.__individual_pickup_frequency, 1, X_pop)
            - W_max[:, np.newaxis]
        )

    def constraints(self, X_pop: np.ndarray) -> np.ndarray:
        """Todas las constraints de :cite:t:`2019:chung` para input en ``pymooo``

        Este método contiene todas las restricciones propuestas ((3), (4) y (5))
        en el paper de Chung.

        :param X_pop: Matriz de indiviuos y variables binarias.
        :param W_max: Vector con valores de la variable W_max
                que se define en el paper de Chung. El elemento
                ``i`` del vector, corresponde al valor para
                el individuo ``i`` de ``X_pop``.
        :return:
            Evaluación de las restricciones por cada individuo.

        """

        return np.concatenate(
            (
                super().constraints(X_pop),
                self.cluster_pickup_frequency_constraint(X_pop),
            ),
            axis=1,
        )


class ChungProblemMultiObjectiveBase(ChungProblemCompleteBase):
    """Clase base para el problema de Chung del paper original."""

    def W_max(self, ind: np.ndarray) -> float:
        """Demanda máxima de los clusters de un individuo.

        Se tiene que comprender que la variable :math:`W_{\text{max}}`
        en :cite:t:`2019:chung`, define la función objetivo (2) de dicho
        trabajo, que es simplementa la función identidad, por lo tanto,
        la función objetivo (2) no necesita evaluarse, si no que solo
        recuperar el valor desde el vector de variables. El valor
        para esta variable depende netamente de como varian las variables
        binaria a lo largo de los operadores mating.

        :param matricial_ind: Individuo en su forma vectorial.

        """

        return (
            np.multiply(self.matrix_individual(ind), self.D[np.newaxis, :])
            .sum(axis=1)
            .max()
        )

    def W_max_pop(self, X_pop: np.ndarray) -> np.ndarray:
        """Demanda máxima poblacional.

        :param X_pop: Matriz de indiviuos y variables.
        :return: Demanda máxima poblacional como vector
            1-D.

        """
        return np.apply_along_axis(self.W_max, 1, X_pop)

    def bi_objective_function(self, X_pop: np.ndarray) -> np.ndarray:
        """Función bi objectivo de afinidad y demanda máxima.

        :param X_pop: Matriz de indiviuos y variables.
        :return: Array 2-D, donde la primera columna corresponde
            a la evaluación de de la afinidad en ``X_pop`` y la segunda
            a la evaluación de la demanda máxima.

        """
        return np.column_stack([self.Q(X_pop), self.W_max_pop(X_pop)])

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        """Evalua el problema según construcción de ``pymoo``

        En este método se definen las funciones objetivos y los
        constraints. La sintaxís de implementación es netamente debido a
        ``pymoo``

        """

        out["F"] = self.bi_objective_function(x)
        out["G"] = self.constraints(x)
