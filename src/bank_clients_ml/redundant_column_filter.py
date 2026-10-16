"""Filtrado de columnas redundantes y transformación por bines con Polars.

Este módulo implementa el proceso de selección de variables utilizado en el notebook
para reducir la dimensionalidad antes del modelado. Elimina en forma secuencial
columnas constantes, columnas binarias desbalanceadas y columnas numéricas
altamente correlacionadas.

El flujo habitual consiste en crear `RedundantColumnFilter` con los conjuntos de
entrenamiento y prueba, inspeccionar los descartes, graficar el análisis bivariado
y finalmente aplicar transformaciones de bines con `apply_binning`.

## Clases exportadas:
    RedundantColumnFilter: Filtro secuencial de columnas redundantes con análisis bivariado.
"""

import dataclasses
import types
from collections.abc import Iterable
from typing import Self

import numpy as np
import polars as pl

from bank_clients_ml import config, eda, training


def _get_true_column_names(df: pl.DataFrame) -> tuple[str, ...]:
    """Obtiene los nombres de las columnas que contienen valores verdaderos.

    ## Args:
        df: DataFrame booleano donde cada celda indica si una columna cumple
            una condición evaluada previamente.

    ## Returns:
        Lista con los nombres de las columnas cuyo valor es verdadero.
    """
    return tuple(df.unpivot().filter(pl.col("value")).get_column("variable").to_list())


def _get_constant_columns(df: pl.DataFrame) -> tuple[str, ...]:
    """Retorna las columnas con un único valor.

    ## Args:
        df: DataFrame sobre el que se evalúa la cardinalidad.

    ## Returns:
        Lista con los nombres de las columnas constantes, las cuales no aportan
        información para el modelado.
    """
    return _get_true_column_names(df.select(pl.all().n_unique() == 1))


def _get_imbalanced_binary_columns(
    df: pl.DataFrame,
    threshold: float,
    settings: config.Settings,
) -> tuple[str, ...]:
    """Retorna las columnas binarias desbalanceadas.

    Detecta las columnas con exactamente dos valores únicos. La proporción se
    calcula contra el primer valor de cada columna y se considera desbalanceada
    cuando cae fuera del intervalo definido por el `threshold` y su complemento.
    La columna target se excluye del análisis.

    ## Args:
        df: DataFrame sobre el que se buscan columnas binarias desbalanceadas.
        threshold: Proporción mínima aceptada para la clase minoritaria. Debe
            encontrarse entre 0 y 0.5.
        settings: Configuración con el nombre de la columna target. Si no se
            indica, se obtiene la configuración global.

    ## Returns:
        Lista con los nombres de las columnas binarias desbalanceadas,
        las cuales no aportan información para el modelado.

    ## Raises:
        ValueError: Si threshold no está en el intervalo abierto entre 0.0 y 0.5.
    """
    if not (0.0 < threshold < 0.5):
        msg = f"Invalid threshold {threshold}: must satisfy 0.0 < threshold < 0.5"
        raise ValueError(msg)

    c_threshold = 1.0 - threshold
    is_binary = pl.all().n_unique() == 2
    first_val_ratio = (pl.all() == pl.all().first()).mean()
    is_imbalanced = (first_val_ratio < threshold) | (first_val_ratio > c_threshold)

    return _get_true_column_names(
        df.drop(settings.dataset.target).select(is_binary & is_imbalanced)
    )


def _get_redundant_correlated_columns(
    corr_df: pl.DataFrame, threshold: float
) -> tuple[str, ...]:
    """Identifica columnas numéricas redundantes por correlación.

    Calcula el valor absoluto de la matriz de correlación y conserva el triángulo
    superior para evaluar cada columna únicamente contra las columnas previas.
    Toda columna cuyo máximo de correlación supere el umbral se considera redundante.

    Deja siempre la primer columna fuera de la lista.
    Si N columnas están correlacionadas entre sí, devolverá N-1 en la lista.
    El triángulo inferior y la diagonal quedan en 0.0,
    lo que no afecta al max() ya que |r| >= 0

    ## Args:
        corr_df: Matriz de correlación cuadrada entre variables numéricas.
        threshold: Umbral de correlación absoluta a partir del cual una columna
            se considera redundante.

    ## Returns:
        Lista con los nombres de las columnas redundantes a eliminar.
    """
    abs_corr_np = np.abs(corr_df.to_numpy())
    upper_triangle = np.triu(abs_corr_np, k=1)

    maximums_per_column = upper_triangle.max(axis=0)
    columns_and_maximums = zip(corr_df.columns, maximums_per_column, strict=True)
    correlated_columns = tuple(
        [col for col, max_val in columns_and_maximums if max_val > threshold]
    )
    return correlated_columns


@dataclasses.dataclass(frozen=True, slots=True)
class RedundantColumnFilter:
    """Filtro secuencial de columnas redundantes con análisis bivariado.

    Aplica tres etapas de reducción sobre los conjuntos de entrenamiento y prueba:
    eliminación de columnas constantes, eliminación de columnas binarias
    desbalanceadas y eliminación de columnas numéricas altamente correlacionadas.
    Conserva los resultados intermedios para su inspección y acumula las tablas
    de análisis bivariado para usarlas posteriormente en la transformación por bines.

    ## Attributes:
        constant_cols: Nombres de las columnas constantes detectadas.
        reduced_train: Conjunto de entrenamiento sin columnas constantes.
        reduced_test: Conjunto de prueba sin columnas constantes.
        imbalanced_binary_columns: Nombres de las columnas binarias desbalanceadas.
    """

    reduced_train: pl.DataFrame
    correlated_train: pl.DataFrame
    correlated_test: pl.DataFrame
    uncorrelated_train: pl.DataFrame
    constant_cols: tuple[str, ...]
    imbalanced_binary_columns: tuple[str, ...]
    correlated_columns: tuple[str, ...]
    imbalanced_binary_threshold: float
    correlation_threshold: float
    settings: config.Settings
    _corr_df: pl.DataFrame

    @classmethod
    def generate(
        cls,
        *,
        train: pl.DataFrame,
        test: pl.DataFrame,
        imbalanced_binary_threshold: float = 0.10,
        correlation_threshold: float = 0.80,
        settings: config.Settings | None = None,
    ) -> Self:
        """Ejecuta las etapas de reducción.

        Detecta columnas constantes, columnas binarias desbalanceadas y columnas
        correlacionadas, y genera las versiones reducidas de los conjuntos de
        entrenamiento y prueba.

        ## Args:
            train: Conjunto de entrenamiento sobre el que se detectan las columnas
                a eliminar.
            test: Conjunto de prueba al que se aplican las mismas eliminaciones.
            imbalanced_binary_threshold: Proporción mínima aceptada para la clase
                minoritaria en columnas binarias.
            correlation_threshold: Umbral de correlación para considerar
                redundante una columna numérica.
            settings: Configuración con los nombres de las columnas identificadora
                y target. Si no se indica, se obtiene la configuración global.
        """
        settings = settings or config.get_settings()

        constant_cols = _get_constant_columns(train)
        reduced_train = train.drop(constant_cols)
        reduced_test = test.drop(constant_cols)

        imbalanced_binary_columns = _get_imbalanced_binary_columns(
            reduced_train, imbalanced_binary_threshold, settings
        )
        correlated_train = reduced_train.drop(imbalanced_binary_columns)
        correlated_test = reduced_test.drop(imbalanced_binary_columns)

        _corr_df = correlated_train.drop(
            settings.dataset.id, settings.dataset.target
        ).corr()
        correlated_columns = _get_redundant_correlated_columns(
            _corr_df, correlation_threshold
        )
        uncorrelated_train = correlated_train.drop(correlated_columns)

        return cls(
            reduced_train,
            correlated_train,
            correlated_test,
            uncorrelated_train,
            constant_cols,
            imbalanced_binary_columns,
            correlated_columns,
            imbalanced_binary_threshold,
            correlation_threshold,
            settings,
            _corr_df,
        )

    @property
    def imbalanced_binary_counts(self) -> pl.DataFrame:
        """Calcula la frecuencia de cada valor en las columnas binarias desbalanceadas.

        Filtra y cuenta las ocurrencias de los valores presentes en las columnas
        definidas en `imbalanced_binary_columns`.

        ## Returns:
            DataFrame con la distribución de frecuencias conteniendo las columnas
            `column`, `value` y `count`.
        """
        return eda.low_cardinality_value_counts(
            self.reduced_train.select(self.imbalanced_binary_columns)
        )

    def _get_correlations_for(self, column: str, threshold: float) -> pl.DataFrame:
        """Obtiene las variables correlacionadas con una columna dada.

        ## Args:
            column: Nombre de la columna de referencia presente en la matriz
                de correlación.
            threshold: Umbral mínimo de correlación para incluir una variable
                en el resultado.

        ## Returns:
            Nuevo DataFrame con las columnas de nombre de variable y correlación.
        """
        return (
            self._corr_df.select(
                pl.Series("feature_name", self._corr_df.columns),
                pl.col(column).alias("correlation"),
            )
            .filter(
                (pl.col("feature_name") != column)
                & (pl.col("correlation").abs() > threshold)
            )
            .sort(pl.col("correlation").abs(), descending=True)
        )

    def correlations_for_each(
        self, columns: Iterable[str], threshold: float
    ) -> types.MappingProxyType[str, pl.DataFrame]:
        """Obtiene las variables correlacionadas de cada columna indicada.

        ## Args:
            target_columns: Nombres de las columnas sobre las cuales se buscan
                correlaciones altas.

        ## Returns:
            Diccionario que mapea cada columna con sus resultados de
            correlación correspondientes.
        """
        return types.MappingProxyType(
            {
                column: self._get_correlations_for(column, threshold)
                for column in columns
            }
        )

    def heavy_operations_and_correlations(
        self, top_grouped_features_name: str
    ) -> tuple[
        training.GroupsLGBMTrainer,
        training.LGBMTrainer,
        types.MappingProxyType[str, pl.DataFrame],
    ]:
        """Add docs."""
        group_trainers = training.GroupsLGBMTrainer.fit_and_build(
            self.uncorrelated_train, settings=self.settings
        )
        group_trainers.plot_top_features()

        top_grouped_features = group_trainers.get_top_grouped_features()
        trainer = training.LGBMTrainer.fit_and_build(
            self.uncorrelated_train, top_grouped_features, self.settings
        )
        trainer.plot_top_features(top_grouped_features_name)

        top_ranked_features = trainer.get_top_ranked_features()

        return (
            group_trainers,
            trainer,
            self.correlations_for_each(top_ranked_features, self.correlation_threshold),
        )
