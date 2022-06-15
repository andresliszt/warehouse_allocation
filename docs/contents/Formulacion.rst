Formulación Matemática del problema de Chung
--------------------------------------------

El problema de Chung descrito :cite:t:`2019:chung`, trata generar eficiencia en un cetro de distribución de productos,
intentando alocar los productos que se piden juntos por los locales que necesitan abastecerse, en ubicaciones cercanas, pues esto hace
que la persona encargada de colectar dichos productos lo haga visitando una menor cantidad de pasillos dentro del centro de distribución.
Además, el trabajo aborda la problemática del tráfico que se podría generar en el centro de distribución, intentando acotar la demanda neta
en cada uno de los pasillos.

El modelamiento matemático es el siguiente: Se consideran :math:`C` clusters (pensar en pasillos), que corresponden a los lugares donde se
alocaran :math:`Q` skus. Se definen como :math:`F_i` como la demanda *neta* del sku :math:`i`, es decir, la cantidad de veces que apareció este 
skus en las órdenes de estudio, y :math:`N_{i,i^{'}}` como la *coaparición* o *afinidad* en el sku :math:`i` y :math:`i^{'}`, que significa la cantidad
de veces que aparecieron juntos dichos skus en las órdenes de trabajo. Denotaremos siempre con :math:`k` la indexación para los clusters, y con
:math:`i` la indexación para los skus, se define la variable binaria :math:`x_{k,i}`, cuyo valor es :math:`1` si es que el sku :math:`i` esta
alocado en el cluster :math:`k`, y :math:`0` en otro caso. Finalmente, cada cluster tiene una capacidad relativa a la cantidad de skus que puede
alocar, dicho valor es representado como :math:`Z_k`. 


El problema de optimización es el siguiente


.. math::
   :nowrap:

   \begin{eqnarray}
      f_1   & = & \max \sum_{i = 1}^{Q-1}\sum_{i^{'} = i + 1}^{Q}\sum_{k = 1}^{C}N_{i,i^{'}}x_{k,i}x_{k,i^{'}}\\
      f_2 & = & \min W_{\text{max}}\\

   \end{eqnarray}

Con las restricciones dadas por

.. math::
   :nowrap:

   \begin{eqnarray}
   
      \sum_{k = 1}^{C} x_{k,i} & = & 1 \hspace{0.5cm} \forall i = 1, ..., Q\\
      \sum_{i = 1}^{Q} x_{k,i} & \leq & Z_{k}  \hspace{0.5cm} \forall k = 1, ..., C\\
      \sum_{i = 1}^{Q} F_ix_{k,i} & \leq & W_{\text{max}}  \hspace{0.5cm} \forall k = 1, ..., C\\
      x_{k,i} & \in & \lbrace 0,1 \rbrace \\
      W_{\text{max}} & \geq & 0

   \end{eqnarray}

La función :math:`f_1` es la que intenta maximizar la *afinidad* entre pares de skus, y la función :math: `f_2`, que es una función
no dependiente de las variables binarias, funciona como regularizador de la demanda en cada uno de los pasillos, al estar amarrada
con la tercera restricción. La primera restricción asegura que cada sku este solamente en un cluster, la segunda asegura que la cantidad
de skus alocados en cluster no sobrepase la capacidad que este posee y la tercera, como se mencionó, controla la demanda en cada cluster.

Adicionalmente, se elaboró un nuevo problema de optimización que considera una **restricción de peso** de los skus. Cada cluster :math:`k`
tiene un atributo :math:`(w_{k,\text{min}}, w_{k,\text{max}})`, donde dichos valores son números reales positivos que representan los rangos
de pesos que tolera el cluster, es decir, un sku puede ser alocado ahí si solo sí su peso esta comprendido entre dichos valores.

.. math::

   \sum_{k = 1}^{C} w_{k,i}x_{k,i} = 1 \hspace{0.5cm} \forall i = 1, ..., Q\\

Donde :math:`w_{k,i}` es una variable binaria que vale 1 si el sku `i` tiene el peso adecuado para ser alocado en el cluster :math:`k`, y 0
en otro caso. 
