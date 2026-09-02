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
    X_train: pl.DataFrame,
    columns: list[str],
    n_iter: int = 2,
    splits_cross_validation: int = 3,
    random_state: int = RANDOM_STATE,
    settings: Settings | None = None,
) -> tuple[RandomizedSearchCV, pl.DataFrame]:
    """Entrena un modelo LightGBM usando RandomizedSearchCV y devuelve las
    importancias de features en un DataFrame de Polars.

    Args:
        X_train : Datos de entrenamiento; debe contener
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

    searcher.fit(X_train.select(columns), X_train[settings.col_target])
    best_estimator: lgb.LGBMClassifier = searcher.best_estimator_
    importances = pl.DataFrame(
        {
            settings.col_feature: columns,
            settings.col_importance: best_estimator.feature_importances_,
        }
    ).sort(settings.col_importance, descending=True)
    return searcher, importances


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
        Array bidimensional con las probabilidades predichas por el modelo.
    bins : opcional
        Si no es nulo, usa los bins (por defecto es nulo).

    Retorna:
    --------
        DataFrame con metricas de los deciles
    """
    if settings is None:
        settings = get_settings()

    decil_expr = (
        pl.col("probabilities").cut(breaks=bins, labels=DECILE_LABELS)
        if bins
        else pl.col("probabilities").qcut(
            10, labels=DECILE_LABELS, allow_duplicates=True
        )
    )

    return (
        df.select(
            settings.col_target,
            probabilities=probabilities[:, 1],
        )
        .with_columns(decil=decil_expr.cast(DECILE_DTYPE))
        .group_by("decil")
        .agg(
            count=pl.len(),
            target_1_count=pl.col(settings.col_target).sum(),
            min_probability=pl.col("probabilities").min(),
        )
        .sort("decil")
    )


def print_train_deciles(train_deciles: pl.DataFrame):
    """imprime metricas de los deciles del set de entrenamiento.

    Args:
        train_deciles: deciles del set de entrenamiento
    """
    print(f"train:\n{train_deciles}")


def print_test_deciles(
    test_deciles: pl.DataFrame, df: pl.DataFrame, probabilities: np.ndarray
):
    """imprime metricas de los deciles del set de test
    e imprime metricas de los deciles recalculados ("trampa").

    Args:
        test_deciles: deciles del set de test
    """
    print(f"test:\n{test_deciles}")

    print("test trampa: recalculo las cotas...")  # TODO
    test_deciles = compute_prediction_deciles(df, probabilities)
    print(test_deciles)
