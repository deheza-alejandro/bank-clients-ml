from typing import cast

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.stats import uniform as sp_uniform
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
)

from bank_clients_ml.config import Settings, get_settings

RANDOM_STATE: int = 314


def stratified_train_test_split(
    df: pl.DataFrame,
    test_ratio: float = 0.3,
    random_state: int = RANDOM_STATE,
    settings: Settings | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Genera particiones de entrenamiento y test estratificadas
    (manteniendo la proporcion de buenos y malos en ambos sets de train y test) usando Polars.

    Parámetros:
    -----------
    df : DataFrame de Polars con los datos.
    target : Nombre de la columna objetivo.
    test_size : Proporción del conjunto de test.
    random_state : Semilla aleatoria.

    Retorna:
    --------
        Tupla con los DataFrames de entrenamiento y test.
    """
    if settings is None:
        settings = get_settings()

    df_shuffled = df.sample(fraction=1.0, shuffle=True, seed=random_state)

    test_indices = (
        df_shuffled.select(pl.col(settings.col_target))
        .with_row_index("_idx")
        .group_by(settings.col_target)
        .agg(pl.col("_idx").head((pl.len() * test_ratio).round().cast(pl.Int64)))
        .explode("_idx")
        .get_column("_idx")
    )

    test = df_shuffled.filter(pl.int_range(0, pl.len()).is_in(test_indices))
    train = df_shuffled.filter(~pl.int_range(0, pl.len()).is_in(test_indices))

    return train, test


def get_feature_importances(
    train: pl.DataFrame,
    columns: list[str],
    n_iter: int = 2,
    splits_cross_validation: int = 3,
    random_state: int = RANDOM_STATE,
    settings: Settings | None = None,
) -> tuple[RandomizedSearchCV, pl.DataFrame]:
    """Entrena un modelo LightGBM usando RandomizedSearchCV y devuelve las
    importancias de features en un DataFrame de Polars.

    Args:
        train : Datos de entrenamiento; debe contener
            ``columns`` y la columna target.
        columns : Columnas de features usadas para entrenar el modelo.
        No debe tener la columna target.
        n_iter : Cantidad de combinaciones de hiperparámetros a probar al azar con
            ``RandomizedSearchCV``.
        target : Nombre de la columna target.
        splits_cross_validation : Cantidad de k-fold cross-validation para usar con
        ``RandomizedSearchCV``. por defecto es 3
        random_state :
        debug :

    Returns:
        ``(searcher, importances)`` El objeto searcher entrenado
        y el DataFrame con las importancias ordenadas descendentemente.
    """
    if settings is None:
        settings = get_settings()

    verbose: int = 3 if settings.debug else 1

    model = lgb.LGBMClassifier(
        random_state=random_state,
        n_jobs=1,
        verbose=verbose,
        metric="auc",
    )

    param_test = {
        "n_estimators": np.arange(6, 50, 1),
        "max_depth": np.arange(
            4, 10, 1
        ),  # [4, 5, 6, 7, 8, 9] de 4 a (10-1) aumentando de a 1
        "num_leaves": np.arange(3, 20, 1),
        "subsample": sp_uniform(loc=0.2, scale=0.8),
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "min_child_samples": np.arange(1000, 3000, 100),
    }

    searcher: RandomizedSearchCV = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_test,
        n_iter=n_iter,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
        cv=StratifiedKFold(
            n_splits=splits_cross_validation,
            shuffle=True,
            random_state=random_state,
        ),
        verbose=verbose,
        random_state=random_state,
    )

    X_train = train.select(columns)
    y_train = train[settings.col_target]
    searcher.fit(X_train, y_train)

    best_estimator: lgb.LGBMClassifier = searcher.best_estimator_
    importances = pl.DataFrame(
        {
            settings.col_feature: columns,
            settings.col_importance: best_estimator.feature_importances_,
        }
    ).sort(settings.col_importance, descending=True)
    return searcher, importances


def oversample_with_unique_ids(
    train: pl.DataFrame,
    target_proportion: float = 0.5,
    random_state: int = RANDOM_STATE,
    settings: Settings | None = None,
) -> pl.DataFrame:
    """Asigna IDs únicos a las filas sobremuestreadas

    No deberias usar esta funcion si usas LightGBM + RandomizedSearchCV + StratifiedKFold.
    En ese caso deberias usar algo como imbalanced-learn para hacer oversampling
    solo sobre los datos de entrenamiento de cada fold de StratifiedKFold,
    dejando intactos los datos de validacion de cada fold"""
    if not (0 < target_proportion < 1):
        raise ValueError("El parámetro 'target_proportion' debe estar entre 0 y 1 (excluyentes).")

    if settings is None:
        settings = get_settings()

    df_majority = train.filter(pl.col(settings.col_target) == 0)
    df_minority = train.filter(pl.col(settings.col_target) == 1)

    max_id = train.select(pl.col(settings.col_id).max()).item()

    count_majority = len(df_majority)
    count_minority_target = round(count_majority * (target_proportion / (1 - target_proportion)))

    df_minority_oversampled = df_minority.sample(
        n=count_minority_target, with_replacement=True, seed=random_state
    ).with_columns((max_id + 1 + pl.int_range(0, pl.len())).alias(settings.col_id))

    balanced_train = pl.concat([df_majority, df_minority_oversampled]).sample(
        fraction=1.0, shuffle=True, seed=random_state
    )
    return balanced_train


QUANTILES = np.linspace(0.1, 0.9, 9)


def get_scoring(
    searcher: RandomizedSearchCV,
    train: pl.DataFrame,
    test: pl.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """y_pred predice si es 0 o 1, si la probabilidad es > 0.5 lo pone como 1

    train_based_bins son los 9 puntos de corte (cuantiles 10% a 90%)
    basados exclusivamente en el set de entrenamiento.
    """
    final_model: lgb.LGBMClassifier = searcher.best_estimator_

    y_pred = cast(np.ndarray, final_model.predict(test.select(features)))

    probabilities_train = cast(
        np.ndarray, final_model.predict_proba(train.select(features))
    )[:, 1]
    probabilities_test = cast(
        np.ndarray, final_model.predict_proba(test.select(features))
    )[:, 1]

    train_based_bins = np.quantile(probabilities_train, QUANTILES).tolist()

    return y_pred, probabilities_train, probabilities_test, train_based_bins


DECILE_LABELS = ["10", "9", "8", "7", "6", "5", "4", "3", "2", "1"]
DECILE_DTYPE = pl.Enum(DECILE_LABELS)


def compute_prediction_deciles(
    df: pl.DataFrame,
    probabilities: np.ndarray,
    bins: list[float] | None = None,
    settings: Settings | None = None,
) -> pl.DataFrame:
    """Combina los datos del cliente con sus probabilidades de predicción,
    calcula los deciles y calcula métricas por decil utilizando Polars.

    Parámetros:
    -----------
    df :
        DataFrame original que contiene las columnas 'Target' y 'client_id'.
    probabilities :
        Array unidimensional con las probabilidades predichas por el modelo.
    bins : opcional
        Si no es nulo, usa los bins (por defecto es nulo).

    Retorna:
    --------
        DataFrame con metricas de los deciles
    """
    if settings is None:
        settings = get_settings()

    decile_expr = (
        pl.col("probabilities").cut(breaks=bins, labels=DECILE_LABELS)
        if bins
        else pl.col("probabilities").qcut(
            10, labels=DECILE_LABELS, allow_duplicates=True
        )
    )

    deciles_df = (
        df.select(
            settings.col_target,
            probabilities=probabilities,
        )
        .with_columns(decile=decile_expr.cast(DECILE_DTYPE))
        .group_by("decile")
        .agg(
            count=pl.len(),
            target_1_count=pl.col(settings.col_target).sum().cast(pl.UInt32),
            min_prob=(pl.col("probabilities").min() * 100).round(2),
            max_prob=(pl.col("probabilities").max() * 100).round(2),
        )
        .sort("decile", descending=True)
    )

    target_1_count = pl.col("target_1_count")
    count = pl.col("count")
    target_0_count = count - target_1_count

    total_target_1_rate = (target_1_count.sum() / count.sum() * 100).round(2)
    target_1_rate = (target_1_count / count * 100).round(2)
    cum_gain = (target_1_count.cum_sum() / target_1_count.sum() * 100).round(2)
    cum_target_0_rate = (target_0_count.cum_sum() / target_0_count.sum() * 100).round(2)

    return deciles_df.with_columns(
        target_1_rate=target_1_rate,
        cum_gain=cum_gain,
        lift=(target_1_rate / total_target_1_rate).round(2),
        ks=cum_gain - cum_target_0_rate,
    )
