"""Ventanas temporales y muestreo para el modelo de "bank_clients_ml".

Centraliza las funciones de muestreo del proyecto basadas en Polars.

## Funciones exportadas:
    stratified_train_test_split: Divide en sets de entrenamiento y prueba de forma estratificada.
    oversample_with_unique_ids: Aplica oversampling con identificadores únicos.

## Clases exportadas:
    TimeWindowSplit: Resultado de la partición de ventanas de entrenamiento y predicción.
"""

import dataclasses
import datetime
from typing import Self

import polars as pl

from bank_clients_ml import config


def _build_date_windows(
    df: pl.DataFrame,
    prediction_window_size: int,
    settings: config.Settings,
) -> tuple[tuple[datetime.date, ...], tuple[datetime.date, ...]]:
    """Calcula las ventanas de entrenamiento y predicción a partir de los meses disponibles.

    Toma el mes mínimo y el mes máximo de la columna de meses y reserva los
    últimos `prediction_window_size` meses como ventana de predicción. La
    ventana de entrenamiento abarca desde el primer mes hasta dos meses antes
    del inicio de la predicción, de modo que siempre queda un mes intermedio de
    separación (Lead Window) que evita fugas de información entre ambas ventanas.

    ## Args:
        df: DataFrame con todos los meses observados.
        prediction_window_size: Cantidad de meses reservados para predicción.
        settings: Configuración con el nombre de la columna de meses.

    ## Returns:
        Tupla con la lista de meses de entrenamiento seguida de la lista de
            meses de predicción, ambas en orden cronológico ascendente.
    """
    pred_offset = f"-{prediction_window_size - 1}mo"
    train_offset = f"-{prediction_window_size + 1}mo"

    last_month = pl.col(settings.dataset.month).max()
    first_month = pl.col(settings.dataset.month).min()

    windows = df.select(
        prediction_months=pl.date_range(
            last_month.dt.offset_by(pred_offset), last_month, interval="1mo"
        ).implode(),
        training_months=pl.date_range(
            first_month, last_month.dt.offset_by(train_offset), interval="1mo"
        ).implode(),
    )

    prediction_months = windows.get_column("prediction_months").item(0).to_list()
    training_months = windows.get_column("training_months").item(0).to_list()

    return tuple(training_months), tuple(prediction_months)


@dataclasses.dataclass(frozen=True, slots=True)
class TimeWindowSplit:
    """Resultado de la partición temporal en ventanas de entrenamiento y predicción.

    Agrupa los meses que definen cada ventana junto con el universo de clientes
    y los subconjuntos de datos correspondientes.

    ## Attributes:
        training_months: Meses incluidos en la ventana de entrenamiento.
        prediction_months: Meses incluidos en la ventana de predicción.
        universe_and_target: Universo de clientes con su variable target.
        universe_and_target_data: Datos completos del universo con la variable
            target incorporada.
        training_data: Registros del universo restringidos a los meses de
            entrenamiento.
        prediction_data: Registros del universo restringidos a los meses de
            predicción.
    """

    training_months: tuple[datetime.date, ...]
    prediction_months: tuple[datetime.date, ...]
    universe_and_target: pl.DataFrame
    universe_and_target_data: pl.DataFrame
    training_data: pl.DataFrame
    prediction_data: pl.DataFrame

    @classmethod
    def generate(
        cls,
        clean_data: pl.DataFrame,
        prediction_window_size: int,
        settings: config.Settings | None = None,
    ) -> Self:
        """Define la partición temporal en ventanas de entrenamiento y predicción.

        Construye el universo de clientes, conserva únicamente a los clientes con
        historial completo y sin el producto en el último mes de entrenamiento, e
        incorpora la variable target. Posteriormente restringe los datos del
        universo a cada ventana.

        ## Args:
            clean_data: Datos limpios con todos los meses, identificadores y
                variable objetivo.
            prediction_window_size: Cantidad de meses reservados para predicción.
            settings: Configuración con los nombres de las columnas relevantes.
                Si no se indica, se obtiene la configuración global.

        ## Returns:
            Partición temporal con los meses de cada ventana, el universo con su
            target y los datos correspondientes de entrenamiento y predicción.
        """
        settings = settings or config.get_settings()

        training_months, prediction_months = _build_date_windows(
            clean_data, prediction_window_size, settings
        )

        is_valid_client = (
            (pl.col(settings.dataset.month) == training_months[-1])
            & (pl.col(settings.dataset.package_active) == "No")
            & (pl.col(settings.dataset.credit_card_co_branding) == "No")
        )
        universe_and_target = (
            clean_data.filter(
                (pl.len().over(settings.dataset.id) == 9)
                & is_valid_client.any().over(settings.dataset.id)
                & pl.col(settings.dataset.month).is_in(prediction_months)
            )
            .select(settings.dataset.id, settings.dataset.target)
            .unique()
        )

        universe_and_target_data = clean_data.drop(settings.dataset.target).join(
            universe_and_target, on=settings.dataset.id, how="inner"
        )

        training_data = universe_and_target_data.filter(
            pl.col(settings.dataset.month).is_in(training_months)
        )

        prediction_data = universe_and_target_data.filter(
            pl.col(settings.dataset.month).is_in(prediction_months)
        )

        return cls(
            training_months,
            prediction_months,
            universe_and_target,
            universe_and_target_data,
            training_data,
            prediction_data,
        )

    @property
    def last_training_month(self) -> datetime.date:
        """Retorna el último mes de la ventana de entrenamiento."""
        return self.training_months[-1]

    @property
    def first_prediction_month(self) -> datetime.date:
        """Retorna el primer mes de la ventana de predicción."""
        return self.prediction_months[0]


def stratified_train_test_split(
    df: pl.DataFrame,
    test_ratio: float = 0.3,
    settings: config.Settings | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Divide un DataFrame en sets de entrenamiento y prueba de forma estratificada.

    Mezcla las filas con la semilla configurada y conserva la proporción
    original de la variable target en ambos conjuntos.

    ## Args:
        df: DataFrame a dividir.
        test_ratio: Fracción de cada clase destinada al conjunto de prueba.
        settings: Configuración con el nombre de la columna target y la semilla
            aleatoria. Si no se indica, se obtiene la configuración global.

    ## Returns:
        Tupla con el DataFrame de entrenamiento seguido del DataFrame de prueba.

    ## Raises:
        ValueError: Si test_ratio no está en el intervalo abierto entre 0 y 1.
    """
    if not (0 < test_ratio < 1):
        msg = f"Invalid test_ratio {test_ratio}: must satisfy 0 < test_ratio < 1"
        raise ValueError(msg)

    settings = settings or config.get_settings()

    df_flagged = df.sample(
        fraction=1.0, shuffle=True, seed=settings.random_state
    ).with_columns(
        _group_id=pl.col(settings.dataset.target)
        .cum_count()
        .over(settings.dataset.target),
        _group_total=pl.len().over(settings.dataset.target),
    )

    is_test = pl.col("_group_id") <= (pl.col("_group_total") * test_ratio).round()

    test = df_flagged.filter(is_test).drop("_group_id", "_group_total")
    train = df_flagged.filter(~is_test).drop("_group_id", "_group_total")

    return train, test


def oversample_with_unique_ids(
    train: pl.DataFrame,
    target_proportion: float = 0.5,
    settings: config.Settings | None = None,
) -> pl.DataFrame:
    """Aplica oversampling con identificadores únicos.

    Mantiene intacta la clase mayoritaria (target igual a 0) y muestrea con
    reemplazo la clase minoritaria (target igual a 1) hasta alcanzar la
    proporción deseada. A las filas duplicadas se les asignan identificadores
    nuevos y consecutivos a partir del máximo existente, y el resultado final
    se mezcla para evitar bloques ordenados por clase.

    No se recomienda usar esta función con LightGBM combinado con
    RandomizedSearchCV y StratifiedKFold. En ese caso se recomienda aplicar
    oversampling solo sobre los datos de entrenamiento de cada fold de
    StratifiedKFold, por ejemplo con imbalanced-learn, y dejar intactos los
    datos de validación de cada fold.

    ## Args:
        train: Conjunto de entrenamiento con la variable target binaria.
        target_proportion: Proporción deseada de la clase minoritaria en el
            resultado, expresada como un valor entre 0 y 1 sin incluir los
            extremos.
        settings: Configuración con los nombres de las columnas de identificador
            y target, y la semilla aleatoria. Si no se indica, se obtiene la
            configuración global.

    ## Returns:
        DataFrame balanceado según la proporción solicitada, con identificadores
            únicos y filas mezcladas.

    ## Raises:
        ValueError: Si target_proportion no está en el intervalo abierto entre
            0 y 1.
    """
    if not (0 < target_proportion < 1):
        msg = f"Invalid target_proportion {target_proportion}: must satisfy 0 < proportion < 1"
        raise ValueError(msg)

    settings = settings or config.get_settings()

    df_majority = train.filter(pl.col(settings.dataset.target) == 0)
    df_minority = train.filter(pl.col(settings.dataset.target) == 1)

    max_id = train.select(pl.col(settings.dataset.id).max()).item()

    count_majority = len(df_majority)
    count_minority_target = round(
        count_majority * (target_proportion / (1 - target_proportion))
    )

    df_minority_oversampled = df_minority.sample(
        n=count_minority_target, with_replacement=True, seed=settings.random_state
    ).with_columns((max_id + 1 + pl.int_range(0, pl.len())).alias(settings.dataset.id))

    balanced_train = pl.concat([df_majority, df_minority_oversampled]).sample(
        fraction=1.0, shuffle=True, seed=settings.random_state
    )
    return balanced_train
