"""Evaluación de desempeño de modelos de clasificación binaria.

## Funciones exportadas:
    evaluate: Evalúa las predicciones de un modelo sobre los conjuntos de datos.

## Clases exportadas:
    EvaluationMetrics: Métricas de evaluación de un modelo de clasificación binaria.
"""

import dataclasses
from collections.abc import Sequence
from typing import cast

import numpy as np
import polars as pl
import sklearn.metrics
import sklearn.model_selection

from bank_clients_ml import config


def _compute_prediction_deciles(
    df: pl.DataFrame,
    probabilities: np.ndarray,
    settings: config.Settings,
    bins: Sequence[float] | None = None,
) -> pl.DataFrame:
    """Calcula métricas de desempeño por decil de probabilidad predicha.

    Segmenta las probabilidades en diez grupos y calcula por cada decil el
    conteo de clientes, la tasa de la clase positiva, la ganancia acumulada,
    el lift y la estadística KS.

    ## Args:
        df: Datos que incluyen la variable target.
        probabilities: Probabilidades predichas por el modelo para la clase positiva.
        settings: Configuración con el nombre de la columna target.
        bins: bins para segmentar las probabilidades. Si no se indica, se
            calculan los deciles a partir de `probabilities`.

    ## Returns:
        Tabla por decil con conteos, probabilidades mínima y máxima, tasa de
            la clase positiva, ganancia acumulada, lift y KS, ordenada de
            forma descendente por decil.
    """
    decile_labels = ["10", "9", "8", "7", "6", "5", "4", "3", "2", "1"]
    decile_dtype = pl.Enum(decile_labels)

    decile_expr = (
        pl.col("probabilities").cut(breaks=bins, labels=decile_labels)
        if bins
        else pl.col("probabilities").qcut(
            10, labels=decile_labels, allow_duplicates=True
        )
    )

    deciles_df = (
        df.select(
            settings.dataset.target,
            probabilities=probabilities,
        )
        .with_columns(decile=decile_expr.cast(decile_dtype))
        .group_by("decile")
        .agg(
            count=pl.len(),
            target_1_count=pl.col(settings.dataset.target).sum().cast(pl.UInt32),
            min_prob=(pl.col("probabilities").min() * 100),
            max_prob=(pl.col("probabilities").max() * 100),
        )
        .sort("decile", descending=True)
    )

    target_1_count = pl.col("target_1_count")
    count = pl.col("count")
    target_0_count = count - target_1_count

    total_target_1_rate = target_1_count.sum() / count.sum() * 100
    target_1_rate = target_1_count / count * 100
    cum_gain = target_1_count.cum_sum() / target_1_count.sum() * 100
    cum_target_0_rate = target_0_count.cum_sum() / target_0_count.sum() * 100

    return deciles_df.with_columns(
        target_1_rate=target_1_rate,
        cum_gain=cum_gain,
        lift=(target_1_rate / total_target_1_rate),
        ks=cum_gain - cum_target_0_rate,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """Métricas de evaluación de un modelo de clasificación binaria.

    Contiene las métricas globales, los puntos de la curva ROC y las tablas
    de desempeño por decil para los conjuntos de entrenamiento y prueba.

    ## Attributes:
        roc_auc: Área bajo la curva ROC calculada sobre el conjunto de prueba.
        accuracy: Proporción de aciertos con umbral de 0.5 sobre el conjunto de prueba.
        fpr: Tasas de falsos positivos de la curva ROC.
        tpr: Tasas de verdaderos positivos de la curva ROC.
        train_deciles: Tabla de desempeño por decil del conjunto de entrenamiento.
        test_deciles: Tabla de desempeño por decil del conjunto de prueba.
    """

    roc_auc: float
    accuracy: float
    fpr: np.ndarray
    tpr: np.ndarray
    train_deciles: pl.DataFrame
    test_deciles: pl.DataFrame

    @property
    def train_deciles_without_probs(self) -> pl.DataFrame:
        """Tabla de deciles de entrenamiento sin columnas de probabilidad.

        ## Returns:
            Copia de la tabla de deciles de entrenamiento sin las columnas
                de probabilidad mínima y máxima.
        """
        return self.train_deciles.drop("min_prob", "max_prob")

    @property
    def test_deciles_without_probs(self) -> pl.DataFrame:
        """Tabla de deciles de prueba sin columnas de probabilidad.

        ## Returns:
            Copia de la tabla de deciles de prueba sin las columnas de
                probabilidad mínima y máxima.
        """
        return self.test_deciles.drop("min_prob", "max_prob")


def evaluate(
    *,
    train: pl.DataFrame,
    test: pl.DataFrame,
    probabilities_train: np.ndarray,
    probabilities_test: np.ndarray,
    settings: config.Settings,
) -> EvaluationMetrics:
    """Evalúa las predicciones de un modelo sobre los conjuntos de datos.

    Calcula las métricas ROC AUC y accuracy junto con los puntos de la curva
    ROC sobre el conjunto de prueba, y construye las tablas de deciles de
    entrenamiento y prueba. Los cortes de los deciles de prueba se derivan
    de los cuantiles de las probabilidades de entrenamiento.

    ## Args:
        train: Datos de entrenamiento utilizados para definir los cortes de deciles.
        test: Datos de prueba sobre los que se calculan las métricas.
        probabilities_train: Probabilidades predichas para la clase positiva
            en el conjunto de entrenamiento.
        probabilities_test: Probabilidades predichas para la clase positiva
            en el conjunto de prueba.
        settings: Configuración con el nombre de la columna target.

    ## Returns:
        Métricas globales, puntos de la curva ROC y tablas de deciles de
            entrenamiento y prueba.
    """
    y_true_arr = test.get_column(settings.dataset.target).to_numpy()
    y_pred = (probabilities_test >= 0.5).astype(int)

    roc_auc = float(sklearn.metrics.roc_auc_score(y_true_arr, probabilities_test))
    accuracy = float(sklearn.metrics.accuracy_score(y_true_arr, y_pred))
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_true_arr, probabilities_test)

    quantiles = np.linspace(0.1, 0.9, 9)
    train_based_bins = np.quantile(probabilities_train, quantiles).tolist()

    train_deciles = _compute_prediction_deciles(train, probabilities_train, settings)
    test_deciles = _compute_prediction_deciles(
        test, probabilities_test, settings, train_based_bins
    )

    return EvaluationMetrics(
        roc_auc,
        accuracy,
        cast(np.ndarray, fpr),  # pyrefly: ignore[redundant-cast]
        cast(np.ndarray, tpr),  # pyrefly: ignore[redundant-cast]
        train_deciles,
        test_deciles,
    )
