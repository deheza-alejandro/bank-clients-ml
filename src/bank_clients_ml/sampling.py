from datetime import date

import polars as pl

from bank_clients_ml.config import Settings, get_settings


def get_date_windows(
    df: pl.DataFrame, date_column: str, prediction_window_size: int
) -> tuple[list[date], list[date]]:
    """el parámetro prediction_window_size se usa para determinar que "offset_by(...)" usar.
        si prediction_window_size es 2, se usa offset_by("-1mo") para el prediction_months,
        si prediction_window_size es 3, se usa offset_by("-2mo") para el prediction_months, y asi.

        la separación entre la ventana de predicción y entrenamiento (Lead Windows
    ) es siempre de 1 mes
    """
    pred_offset = f"-{prediction_window_size - 1}mo"
    train_offset = f"-{prediction_window_size + 1}mo"

    last_month = pl.col(date_column).max()
    first_month = pl.col(date_column).min()

    windows = df.select(
        prediction_months=pl.date_range(
            last_month.dt.offset_by(pred_offset), last_month, interval="1mo"
        ).implode(),
        training_months=pl.date_range(
            first_month, last_month.dt.offset_by(train_offset), interval="1mo"
        ).implode(),
    )

    prediction_months = windows["prediction_months"][0].to_list()
    training_months = windows["training_months"][0].to_list()

    return training_months, prediction_months


def stratified_train_test_split(
    df: pl.DataFrame,
    test_ratio: float = 0.3,
    settings: Settings | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Genera particiones de entrenamiento y test estratificadas
    (manteniendo la proporción de buenos y malos en ambos sets de train y test) usando Polars.

    Parámetros:
    -----------
    df : DataFrame de Polars con los datos.
    target : Nombre de la columna objetivo.
    test_size : Proporción del conjunto de test.

    Retorna:
    --------
        Tupla con los DataFrames de entrenamiento y test.
    """
    if settings is None:
        settings = get_settings()

    df_flagged = df.sample(
        fraction=1.0, shuffle=True, seed=settings.random_state
    ).with_columns(
        _group_id=pl.col(settings.col_target).cum_count().over(settings.col_target),
        _group_total=pl.len().over(settings.col_target),
    )

    is_test = pl.col("_group_id") <= (pl.col("_group_total") * test_ratio).round()

    test = df_flagged.filter(is_test).drop("_group_id", "_group_total")
    train = df_flagged.filter(~is_test).drop("_group_id", "_group_total")

    return train, test


def oversample_with_unique_ids(
    train: pl.DataFrame,
    target_proportion: float = 0.5,
    settings: Settings | None = None,
) -> pl.DataFrame:
    """Asigna IDs únicos a las filas nuevas generadas por el oversampling

    No deberías usar esta función si usas LightGBM + RandomizedSearchCV + StratifiedKFold.
    En ese caso deberías usar algo como imbalanced-learn para hacer oversampling
    solo sobre los datos de entrenamiento de cada fold de StratifiedKFold,
    dejando intactos los datos de validación de cada fold"""
    if not (0 < target_proportion < 1):
        raise ValueError(
            f"Invalid target_proportion {target_proportion}: must satisfy 0 < target_proportion < 1"
        )

    if settings is None:
        settings = get_settings()

    df_majority = train.filter(pl.col(settings.col_target) == 0)
    df_minority = train.filter(pl.col(settings.col_target) == 1)

    max_id = train.select(pl.col(settings.col_id).max()).item()

    count_majority = len(df_majority)
    count_minority_target = round(
        count_majority * (target_proportion / (1 - target_proportion))
    )

    df_minority_oversampled = df_minority.sample(
        n=count_minority_target, with_replacement=True, seed=settings.random_state
    ).with_columns((max_id + 1 + pl.int_range(0, pl.len())).alias(settings.col_id))

    balanced_train = pl.concat([df_majority, df_minority_oversampled]).sample(
        fraction=1.0, shuffle=True, seed=settings.random_state
    )
    return balanced_train
