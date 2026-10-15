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
        bins: bins para segmentar las probabilidades (util para test).
            Si no se indica, se calcula en deciles según `probabilities`.
        settings: Configuración con el nombre de la columna target. Si no se
            indica, se obtiene la configuración global.

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
    """Métricas de un modelo LightGBM.

    ## Attributes:
        roc_auc: Métrica ROC AUC del modelo evaluado.
        accuracy: Exactitud del modelo evaluado.
        fpr: Tasas de falsos positivos de la curva ROC.
        tpr: Tasas de verdaderos positivos de la curva ROC.
    """

    roc_auc: float
    accuracy: float
    fpr: np.ndarray
    tpr: np.ndarray
    train_deciles: pl.DataFrame
    test_deciles: pl.DataFrame

    @property
    def train_deciles_without_probs(self) -> pl.DataFrame:
        return self.train_deciles.drop("min_prob", "max_prob")

    @property
    def test_deciles_without_probs(self) -> pl.DataFrame:
        return self.test_deciles.drop("min_prob", "max_prob")


def evaluate(
    *,
    train: pl.DataFrame,
    test: pl.DataFrame,
    probabilities_train: np.ndarray,
    probabilities_test: np.ndarray,
    settings: config.Settings,
) -> EvaluationMetrics:
    """Evalúa el mejor modelo sobre el conjunto de prueba.

    Calcula las probabilidades de entrenamiento y prueba, obtiene las
    métricas ROC AUC y accuracy junto con los puntos de la curva ROC, y
    construye las tablas de deciles. Los bins de los deciles de prueba se
    derivan de los bins de las probabilidades de entrenamiento.

    ## Args:
        searcher: Buscador entrenado cuyo mejor estimador se desea evaluar.
        train: Datos de entrenamiento utilizados para definir los bins.
        test: Datos de prueba sobre los que se calculan las métricas.
        columns: Columnas utilizadas de `train`.
        settings: Configuración con el nombre de la columna target. Si no se
            indica, se obtiene la configuración global.

    ## Returns:
        Tupla con ROC AUC, exactitud, tasas de falsos positivos y de
            verdaderos positivos de la curva ROC, y tablas de deciles de
            entrenamiento y de prueba.
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
