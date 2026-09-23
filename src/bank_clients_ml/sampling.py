"""Ventanas temporales y muestreo para el modelo de "bank_clients_ml".

Centraliza las funciones de muestreo del proyecto basadas en Polars

Funciones exportadas:
    get_date_windows: Divide en ventanas de entrenamiento y predicción.
    stratified_train_test_split: Divide en entrenamiento y prueba de forma estratificada.
    oversample_with_unique_ids: Realiza oversampling con identificadores únicos.
"""

from datetime import date

import polars as pl

from bank_clients_ml.config import Settings, get_settings


def get_date_windows(
    df: pl.DataFrame, date_column: str, prediction_window_size: int
) -> tuple[list[date], list[date]]:
    """Divide en ventanas de entrenamiento y predicción.

    Toma el mes mínimo y el mes máximo de la columna de fechas y reserva los
    últimos `prediction_window_size` meses como ventana de predicción. La
    ventana de entrenamiento abarca desde el primer mes hasta dos meses antes
    del inicio de la predicción, de modo que siempre queda un mes intermedio de
    separación (Lead Window) que evita la fuga de información entre ambas ventanas.

    Args:
        df: DataFrame con todos los meses.
        date_column: Nombre de la columna con las fechas mensuales.
        prediction_window_size: Cantidad de meses reservados para predicción.

    Returns:
        Tupla con la lista de meses de entrenamiento seguida de la lista de
        meses de predicción, ambas en orden cronológico ascendente.
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
    """Divide un DataFrame en entrenamiento y prueba de forma estratificada.

    Mezcla las filas con la semilla configurada.
    Se conserva la proporción original del target en ambos conjuntos.

    Args:
        df: Dataframe a dividir.
        test_ratio: Fracción de cada clase destinada al conjunto de prueba.
        settings: Configuración con el nombre de la columna target y la
            semilla aleatoria. Si es None, se obtiene la configuración global.

    Returns:
        Tupla con el DataFrame de entrenamiento seguido del DataFrame de
        prueba.
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
    """Realiza oversampling con identificadores únicos.

    Mantiene intacta la clase mayoritaria (target igual a 0) y muestrea con
    reemplazo la clase minoritaria (target igual a 1) hasta alcanzar la
    proporción deseada. A las filas duplicadas se les asignan identificadores
    nuevos y consecutivos a partir del máximo existente, y el resultado final
    se mezcla para evitar bloques ordenados por clase.

    No deberías usar esta función si usas LightGBM + RandomizedSearchCV + StratifiedKFold.
    En ese caso deberías usar algo como imbalanced-learn para hacer oversampling
    solo sobre los datos de entrenamiento de cada fold de StratifiedKFold,
    dejando intactos los datos de validación de cada fold.

    Args:
        train: Conjunto de entrenamiento con el target binario.
        target_proportion: Proporción de target de la clase minoritaria en el
            resultado, expresada como un valor entre 0 y 1 sin incluir los
            extremos.
        settings: Configuración con los nombres de las columnas de
            identificador y target, y la semilla aleatoria.
            Si es None, se obtiene la configuración global.

    Returns:
        DataFrame balanceado según la proporción solicitada, con
        identificadores únicos y filas mezcladas.

    Raises:
        ValueError: Si la proporción de target no está en el intervalo abierto
            entre 0 y 1.
    """
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
