"""Constructores del flujo de preparación de datos de integración."""

import dataclasses
import datetime

import polars as pl

import tests.support.monthly_columns
import tests.support.scenario
from bank_clients_ml import (
    cleaning,
    column_groups,
    config,
    feature_engineering,
    sampling,
)

ELIGIBLE_CLIENT_IDS: tuple[int, ...] = tuple(range(1, 49))
_INVALID_CLIENT_IDS: tuple[int, ...] = tuple(range(101, 111))

_AGE_GROUPS: tuple[str, ...] = ("18-24", "25-34", "35-44", "45-54")
_REGIONS: tuple[str, ...] = ("BUENOS AIRES", "CORDOBA", "SANTA FE")
_CREDIT_CARD_PRODUCTS: tuple[str, ...] = ("VISA", "MASTERCARD", "AMEX")


@dataclasses.dataclass(frozen=True)
class IntegrationPreparedData:
    """Resultado del flujo de preparación de datos hasta la partición final.

    ## Attributes:
        clean_data: Datos crudos ya limpiados y tipados.
        time_window_split: Ventanas temporales, universo y target agregado.
        identity_features: Identity features codificadas y con fechas derivadas.
        transformed_training_data: Historia mensual con transformaciones.
        abt: Analytical Base Table a nivel de cliente.
        train: Conjunto de entrenamiento estratificado.
        test: Conjunto de prueba estratificado.
    """

    clean_data: pl.DataFrame
    time_window_split: sampling.TimeWindowSplit
    identity_features: pl.DataFrame
    transformed_training_data: pl.DataFrame
    abt: pl.DataFrame
    train: pl.DataFrame
    test: pl.DataFrame


def _last_product_date(client_id: int) -> datetime.date:
    """Devuelve una fecha de último producto anterior a la ventana de entrenamiento."""
    return datetime.date(2019, 1, 1) + datetime.timedelta(days=(client_id * 37) % 1_500)


def _monthly_value(
    client_id: int,
    column_index: int,
    month_index: int,
    column: str,
) -> float:
    """Genera un valor mensual determinista y variable por cliente y columna."""
    if column.startswith("SavingAccount_Days_with"):
        return float((client_id * 3 + month_index * 2 + column_index) % 30 + 1)

    value = float((client_id * 31 + column_index * 97 + month_index * 53) % 101 + 1)
    if (client_id + column_index + month_index) % 17 == 0:
        return 0.0
    if any(
        token in column
        for token in ("Amount", "Balance", "Limit", "Spending", "Revolving")
    ):
        value *= 100.0
    return value


def _binary_value(column: str, client_id: int, month_index: int) -> str:
    """Genera un valor binario determinista para una identity feature."""
    if column == "Sex":
        return "M" if client_id % 2 == 0 else "F"
    score = (client_id * 7 + month_index * 3 + len(column)) % 5
    return "Yes" if score == 0 else "No"


def _build_row(
    client_id: int,
    month: datetime.date,
    target: int | None,
    *,
    package_active: str,
    credit_card_co_branding: str,
) -> dict[str, object]:
    """Construye una fila cruda con todas las columnas esperadas por el pipeline."""
    month_index = tests.support.scenario.MONTHS.index(month)
    last_product_date = _last_product_date(client_id)
    first_product_date = last_product_date - datetime.timedelta(
        days=400 + client_id % 500
    )

    row: dict[str, object] = {
        "client_id": str(client_id),
        "Month": month.isoformat(),
        "First_product_dt": first_product_date.isoformat(),
        "Last_product_dt": last_product_date.isoformat(),
        "Target": target,
        "Package_Active": package_active,
        "CreditCard_CoBranding": credit_card_co_branding,
        "Client_Age_grp": _AGE_GROUPS[client_id % len(_AGE_GROUPS)],
        "Region": _REGIONS[client_id % len(_REGIONS)],
        "CreditCard_Product": _CREDIT_CARD_PRODUCTS[
            client_id % len(_CREDIT_CARD_PRODUCTS)
        ],
    }

    for column_index, column in enumerate(
        tests.support.monthly_columns.MONTHLY_PIPELINE_INPUT_COLS
    ):
        row[column] = _monthly_value(client_id, column_index, month_index, column)

    for column in column_groups.BINARY_IDENTITY_FEATURES_COLS:
        if column in {"Package_Active", "CreditCard_CoBranding"}:
            continue
        row[column] = _binary_value(column, client_id, month_index)

    return row


def _eligible_months(client_id: int) -> tuple[datetime.date, ...]:
    """Devuelve nueve meses de historia conservando el último mes de entrenamiento."""
    missing_candidates = tuple(
        month
        for month in tests.support.scenario.MONTHS
        if month != tests.support.scenario.LAST_TRAINING_MONTH
    )
    missing_month = missing_candidates[(client_id - 1) % len(missing_candidates)]
    return tuple(
        month for month in tests.support.scenario.MONTHS if month != missing_month
    )


def _invalid_months(client_id: int) -> tuple[datetime.date, ...]:
    """Devuelve meses que convierten al cliente en no elegible para el universo."""
    months = tests.support.scenario.MONTHS
    if client_id == 101:
        return tuple(month for month in months if month not in months[:2])
    if client_id == 102:
        return tuple(month for month in months if month not in months[2:4])
    if client_id in {103, 104}:
        return months
    if client_id in {105, 106}:
        return tuple(month for month in months if month != months[1])
    if client_id in {107, 108}:
        return tuple(month for month in months if month != months[2])
    return tuple(
        month for month in months if month != tests.support.scenario.LAST_TRAINING_MONTH
    )


def _build_client_rows(client_id: int, months: tuple[datetime.date, ...]) -> list[dict]:
    """Construye las filas de un cliente con target agregado y banderas controladas."""
    is_eligible = client_id in ELIGIBLE_CLIENT_IDS
    present_prediction_months = [
        month for month in tests.support.scenario.PREDICTION_MONTHS if month in months
    ]
    positive_month = (
        present_prediction_months[0]
        if is_eligible and client_id <= 24 and present_prediction_months
        else None
    )

    rows = []
    for month in months:
        is_last_training_month = month == tests.support.scenario.LAST_TRAINING_MONTH
        package_active = (
            "Yes" if client_id in {105, 106} and is_last_training_month else "No"
        )
        credit_card_co_branding = (
            "Yes" if client_id in {107, 108} and is_last_training_month else "No"
        )
        target = 1 if month == positive_month else 0
        rows.append(
            _build_row(
                client_id,
                month,
                target,
                package_active=package_active,
                credit_card_co_branding=credit_card_co_branding,
            )
        )
    return rows


def build_raw_bank_data() -> pl.DataFrame:
    """Genera un conjunto crudo con clientes elegibles, inválidos y targets nulos."""
    rows: list[dict] = []

    for client_id in ELIGIBLE_CLIENT_IDS:
        rows.extend(_build_client_rows(client_id, _eligible_months(client_id)))

    for client_id in _INVALID_CLIENT_IDS:
        rows.extend(_build_client_rows(client_id, _invalid_months(client_id)))

    null_target_rows = (
        _build_row(
            101,
            tests.support.scenario.MONTHS[0],
            None,
            package_active="No",
            credit_card_co_branding="No",
        ),
        _build_row(
            101,
            tests.support.scenario.MONTHS[1],
            None,
            package_active="No",
            credit_card_co_branding="No",
        ),
        _build_row(
            102,
            tests.support.scenario.MONTHS[2],
            None,
            package_active="No",
            credit_card_co_branding="No",
        ),
    )
    rows.extend(null_target_rows)

    return pl.DataFrame(rows)


def build_clean_data(
    raw_bank_data: pl.DataFrame,
    settings: config.Settings,
) -> pl.DataFrame:
    """Devuelve los datos crudos ya limpiados.

    ## Args:
        raw_bank_data: Datos crudos con clientes elegibles, inválidos y nulos.
        settings: Configuración con los nombres de las columnas.

    ## Returns:
        DataFrame limpio.
    """
    return cleaning.clean_up_data(raw_bank_data, settings)


def build_time_window_split(
    clean_data: pl.DataFrame,
    settings: config.Settings,
    prediction_window_size: int = 2,
) -> sampling.TimeWindowSplit:
    """Devuelve la partición temporal generada desde los datos limpios.

    ## Args:
        clean_data: Datos limpios y tipados.
        settings: Configuración con los nombres de las columnas.
        prediction_window_size: Cantidad de meses reservados para predicción.

    ## Returns:
        Ventana temporal con el universo y el target agregado.
    """
    return sampling.TimeWindowSplit.generate(
        clean_data,
        prediction_window_size,
        settings=settings,
    )


def prepare_data(
    clean_data: pl.DataFrame,
    settings: config.Settings,
    prediction_window_size: int = 2,
) -> IntegrationPreparedData:
    """Ejecuta la preparación de datos desde la limpieza hasta el split.

    ## Args:
        clean_data: Datos limpios y tipados.
        settings: Configuración con los nombres de las columnas.
        prediction_window_size: Cantidad de meses reservados para predicción.

    ## Returns:
        Artefactos intermedios y finales del flujo de preparación.
    """
    time_window_split = build_time_window_split(
        clean_data,
        settings,
        prediction_window_size,
    )

    binary_mapping = {"Yes": 1, "No": 0, "M": 1, "F": 0}
    training_data = time_window_split.training_data.with_columns(
        pl.col(column_groups.BINARY_IDENTITY_FEATURES_COLS)
        .replace_strict(binary_mapping)
        .cast(pl.UInt8)
    )

    identity_features = training_data.filter(
        pl.col(settings.dataset.month) == time_window_split.last_training_month
    ).select(column_groups.get_identity_features_cols(settings))

    encoded_identity, training_without_categorical_features = (
        feature_engineering.target_encode_columns(
            identity_features=identity_features,
            training_data=training_data,
            settings=settings,
        )
    )
    dated_identity = feature_engineering.transform_date_features(
        encoded_identity,
        time_window_split.last_training_month,
        settings=settings,
    )
    transformed_training_data = feature_engineering.with_transformations(
        training_without_categorical_features
    )
    aggregated = feature_engineering.aggregate_monthly_to_client(
        transformed_training_data,
        settings=settings,
    )
    abt = feature_engineering.with_extra_transformations(
        dated_identity.join(
            aggregated,
            on=settings.dataset.id,
            how="inner",
        )
    )
    train, test = sampling.stratified_train_test_split(abt, settings=settings)

    return IntegrationPreparedData(
        clean_data=clean_data,
        time_window_split=time_window_split,
        identity_features=dated_identity,
        transformed_training_data=transformed_training_data,
        abt=abt,
        train=train,
        test=test,
    )


def build_prepared_data(
    settings: config.Settings,
    prediction_window_size: int = 2,
) -> IntegrationPreparedData:
    """Construye el flujo completo desde los datos crudos hasta la partición.

    ## Args:
        settings: Configuración con los nombres de las columnas.
        prediction_window_size: Cantidad de meses reservados para predicción.

    ## Returns:
        Artefactos intermedios y finales del flujo de preparación.
    """
    raw_bank_data = build_raw_bank_data()
    clean_data = build_clean_data(raw_bank_data, settings)
    return prepare_data(clean_data, settings, prediction_window_size)


def build_train_test_abt(
    settings: config.Settings,
    prediction_window_size: int = 2,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Devuelve los conjuntos de entrenamiento y prueba de la ABT.

    ## Args:
        settings: Configuración con los nombres de las columnas.
        prediction_window_size: Cantidad de meses reservados para predicción.

    ## Returns:
        Conjuntos de entrenamiento y prueba.
    """
    prepared = build_prepared_data(settings, prediction_window_size)
    return prepared.train, prepared.test


def rename_raw_data(
    raw_data: pl.DataFrame,
    settings: config.Settings,
) -> pl.DataFrame:
    """Renombra las columnas del dataset configuradas.

    ## Args:
        raw_data: Datos crudos con los nombres por defecto.
        settings: Configuración con los nombres de columna alternativos.

    ## Returns:
        DataFrame con los nombres de columna configurados.
    """
    defaults = config.DatasetColumns()
    mapping = {
        getattr(defaults, field_name): getattr(settings.dataset, field_name)
        for field_name in config.DatasetColumns.model_fields
        if getattr(defaults, field_name) != getattr(settings.dataset, field_name)
    }
    return raw_data.rename(mapping)
