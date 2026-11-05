"""Fixtures compartidos por los tests de integración del proyecto."""

import dataclasses
import datetime
import pathlib
from collections.abc import Callable, Iterator

import matplotlib.pyplot as plt
import polars as pl
import pytest
import sklearn.datasets

from bank_clients_ml import (
    cleaning,
    column_groups,
    config,
    feature_engineering,
    sampling,
    training,
)

MONTHS: tuple[datetime.date, ...] = tuple(
    datetime.date(2024, month, 1) for month in range(1, 11)
)
TRAINING_MONTHS: tuple[datetime.date, ...] = MONTHS[:7]
LEAD_MONTH: datetime.date = MONTHS[7]
PREDICTION_MONTHS: tuple[datetime.date, ...] = MONTHS[8:]
LAST_TRAINING_MONTH: datetime.date = TRAINING_MONTHS[-1]
FIRST_PREDICTION_MONTH: datetime.date = PREDICTION_MONTHS[0]
ELIGIBLE_CLIENT_IDS: tuple[int, ...] = tuple(range(1, 49))
INVALID_CLIENT_IDS: tuple[int, ...] = tuple(range(101, 111))
NULL_TARGET_COUNT = 3

MONTHLY_COLUMNS: tuple[str, ...] = (
    "Operations_Bank",
    "Operations_Terminal",
    "Operations_HomeBanking",
    "Operations_Mobile",
    "Operations_Ivr",
    "Operations_Telemarketer",
    "Operations_ATM",
    "CreditCard_Payment_Aut_Debit",
    "CreditCard_Payment_External",
    "CreditCard_Payment_Cash",
    "CreditCard_Payment_Web",
    "CreditCard_Payment_ATM",
    "CreditCard_Payment_TAS",
    "SavingAccount_Balance_LastDate",
    "SavingAccount_Balance_FirstDate",
    "SavingAccount_Days_with_Debits",
    "SavingAccount_Days_with_use",
    "SavingAccount_Days_with_Credits",
    "SavingAccount_Credits_Transactions",
    "SavingAccount_Debits_Transactions",
    "SavingAccount_Transactions_Transactions",
    "SavingAccount_Salary_Payment_Transactions",
    "SavingAccount_Transfer_In_Transactions",
    "SavingAccount_ATM_Extraction_Transactions",
    "SavingAccount_Service_Payment_Transactions",
    "SavingAccount_CreditCard_Payment_Transactions",
    "SavingAccount_Transfer_Out_Transactions",
    "SavingAccount_DebitCard_Spend_Transactions",
    "SavingAccount_Credits_Amounts",
    "SavingAccount_Debits_Amounts",
    "SavingAccount_Total_Amount",
    "SavingAccount_Salary_Payment_Amount",
    "SavingAccount_Transfer_In_Amount",
    "SavingAccount_ATM_Extraction_Amount",
    "SavingAccount_Service_Payment_Amount",
    "SavingAccount_CreditCard_Payment_Amount",
    "SavingAccount_Transfer_Out_Amount",
    "SavingAccount_DebitCard_Spend_Amount",
    "CreditCard_Balance_ARG",
    "CreditCard_Balance_DOLLAR",
    "CreditCard_Total_Limit",
    "CreditCard_Total_Spending",
    "CreditCard_Spending_1_Installment",
    "CreditCard_Spending_Installments",
    "CreditCard_Spending_CrossBoarder",
    "CreditCard_Spending_Aut_Debits",
    "CreditCard_Revolving",
    "CreditCard_Premium",
    "CreditCard_Active",
    "Loan_Active",
    "Mortgage_Active",
    "SavingAccount_Active_ARG_Salary",
    "SavingAccount_Active_ARG",
    "SavingAccount_Active_DOLLAR",
    "DebitCard_Active",
    "Investment_Active",
    "Insurance_Life",
    "Insurance_Home",
    "Insurance_Accidents",
    "Insurance_Mobile",
    "Insurance_ATM",
    "Insurance_Unemployment",
)

ALL_MONTHLY_COLUMNS: tuple[str, ...] = (
    *MONTHLY_COLUMNS,
    "SavingAccount_Balance_Average",
)

AGE_GROUPS: tuple[str, ...] = ("18-24", "25-34", "35-44", "45-54")
REGIONS: tuple[str, ...] = ("BUENOS AIRES", "CORDOBA", "SANTA FE")
CREDIT_CARD_PRODUCTS: tuple[str, ...] = ("VISA", "MASTERCARD", "AMEX")

SYNTHETIC_FEATURE_COLUMNS: tuple[str, ...] = (
    "Feature_1",
    "Feature_2",
    "Feature_3",
    "Feature_4",
    "Feature_5",
)
SYNTHETIC_N_SAMPLES = 2_000
SYNTHETIC_N_ITER = 1
SYNTHETIC_SPLITS_CROSS_VALIDATION = 2


@dataclasses.dataclass(frozen=True)
class PreparedData:
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
    month_index = MONTHS.index(month)
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
        "Client_Age_grp": AGE_GROUPS[client_id % len(AGE_GROUPS)],
        "Region": REGIONS[client_id % len(REGIONS)],
        "CreditCard_Product": CREDIT_CARD_PRODUCTS[
            client_id % len(CREDIT_CARD_PRODUCTS)
        ],
    }

    for column_index, column in enumerate(ALL_MONTHLY_COLUMNS):
        row[column] = _monthly_value(client_id, column_index, month_index, column)

    for column in column_groups.BINARY_IDENTITY_FEATURES_COLS:
        if column in {"Package_Active", "CreditCard_CoBranding"}:
            continue
        row[column] = _binary_value(column, client_id, month_index)

    return row


def _eligible_months(client_id: int) -> tuple[datetime.date, ...]:
    """Devuelve nueve meses de historia conservando el último mes de entrenamiento."""
    missing_candidates = tuple(
        month for month in MONTHS if month != LAST_TRAINING_MONTH
    )
    missing_month = missing_candidates[(client_id - 1) % len(missing_candidates)]
    return tuple(month for month in MONTHS if month != missing_month)


def _invalid_months(client_id: int) -> tuple[datetime.date, ...]:
    """Devuelve meses que convierten al cliente en no elegible para el universo."""
    if client_id == 101:
        return tuple(month for month in MONTHS if month not in MONTHS[:2])
    if client_id == 102:
        return tuple(month for month in MONTHS if month not in MONTHS[2:4])
    if client_id in {103, 104}:
        return MONTHS
    if client_id in {105, 106}:
        return tuple(month for month in MONTHS if month != MONTHS[1])
    if client_id in {107, 108}:
        return tuple(month for month in MONTHS if month != MONTHS[2])
    return tuple(month for month in MONTHS if month != LAST_TRAINING_MONTH)


def _build_client_rows(client_id: int, months: tuple[datetime.date, ...]) -> list[dict]:
    """Construye las filas de un cliente con target agregado y banderas controladas."""
    is_eligible = client_id in ELIGIBLE_CLIENT_IDS
    present_prediction_months = [
        month for month in PREDICTION_MONTHS if month in months
    ]
    positive_month = (
        present_prediction_months[0]
        if is_eligible and client_id <= 24 and present_prediction_months
        else None
    )

    rows = []
    for month in months:
        is_last_training_month = month == LAST_TRAINING_MONTH
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


def _build_raw_bank_data() -> pl.DataFrame:
    """Genera un conjunto crudo con clientes elegibles, inválidos y targets nulos."""
    rows: list[dict] = []

    for client_id in ELIGIBLE_CLIENT_IDS:
        rows.extend(_build_client_rows(client_id, _eligible_months(client_id)))

    for client_id in INVALID_CLIENT_IDS:
        rows.extend(_build_client_rows(client_id, _invalid_months(client_id)))

    null_target_rows = (
        _build_row(
            101,
            MONTHS[0],
            None,
            package_active="No",
            credit_card_co_branding="No",
        ),
        _build_row(
            101,
            MONTHS[1],
            None,
            package_active="No",
            credit_card_co_branding="No",
        ),
        _build_row(
            102,
            MONTHS[2],
            None,
            package_active="No",
            credit_card_co_branding="No",
        ),
    )
    rows.extend(null_target_rows)

    return pl.DataFrame(rows)


def _prepare_data(
    clean_data: pl.DataFrame,
    settings: config.Settings,
    prediction_window_size: int = 2,
) -> PreparedData:
    """Ejecuta la preparación de datos desde la limpieza hasta el split."""
    time_window_split = sampling.TimeWindowSplit.generate(
        clean_data,
        prediction_window_size,
        settings=settings,
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

    return PreparedData(
        clean_data=clean_data,
        time_window_split=time_window_split,
        identity_features=dated_identity,
        transformed_training_data=transformed_training_data,
        abt=abt,
        train=train,
        test=test,
    )


def _build_synthetic_train(random_state: int) -> pl.DataFrame:
    """Genera un conjunto sintético compatible con el esquema del proyecto."""
    features, target = sklearn.datasets.make_classification(
        n_samples=SYNTHETIC_N_SAMPLES,
        n_features=len(SYNTHETIC_FEATURE_COLUMNS),
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        weights=[0.85, 0.15],
        class_sep=1.5,
        flip_y=0.01,
        random_state=random_state,
    )
    return pl.DataFrame(features, schema=list(SYNTHETIC_FEATURE_COLUMNS)).with_columns(
        pl.Series("Target", target, dtype=pl.Int64),
        pl.lit(0.0).alias("Ignored"),
    )


@pytest.fixture(autouse=True)
def clear_global_settings_cache() -> Iterator[None]:
    """Limpia la caché de configuración antes y después de cada test."""
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


@pytest.fixture(autouse=True)
def close_matplotlib_figures() -> Iterator[None]:
    """Cierra todas las figuras de Matplotlib al finalizar cada test."""
    yield
    plt.close("all")


@pytest.fixture
def integration_settings(tmp_path: pathlib.Path) -> config.Settings:
    """Devuelve una configuración aislada para los tests de integración."""
    return config.Settings(
        paths=config.DirectoryPaths(project=tmp_path),
        n_jobs=1,
    )


@pytest.fixture
def custom_columns_settings() -> config.Settings:
    """Devuelve una configuración con identificador, target, mes y fechas propios."""
    return config.Settings(
        dataset=config.DatasetColumns(
            id="customer_id",
            target="churn",
            month="snapshot_month",
            first_product_date="first_seen",
            last_product_date="last_seen",
        )
    )


@pytest.fixture
def raw_bank_data() -> pl.DataFrame:
    """Devuelve datos crudos con clientes elegibles, inválidos y targets nulos."""
    return _build_raw_bank_data()


@pytest.fixture
def expected_eligible_ids() -> tuple[int, ...]:
    """Devuelve los identificadores de los clientes que deben formar el universo."""
    return ELIGIBLE_CLIENT_IDS


@pytest.fixture
def clean_data(
    raw_bank_data: pl.DataFrame,
    integration_settings: config.Settings,
) -> pl.DataFrame:
    """Devuelve los datos crudos ya limpiados."""
    return cleaning.clean_up_data(raw_bank_data, integration_settings)


@pytest.fixture
def time_window_split(
    clean_data: pl.DataFrame,
    integration_settings: config.Settings,
) -> sampling.TimeWindowSplit:
    """Devuelve la partición temporal generada desde los datos limpios."""
    return sampling.TimeWindowSplit.generate(
        clean_data,
        2,
        settings=integration_settings,
    )


@pytest.fixture
def prepare_data() -> Callable[[pl.DataFrame, config.Settings], PreparedData]:
    """Devuelve el helper que ejecuta el flujo completo de preparación."""
    return _prepare_data


@pytest.fixture
def prepared_data(
    clean_data: pl.DataFrame,
    integration_settings: config.Settings,
) -> PreparedData:
    """Devuelve todos los artefactos intermedios del flujo de preparación."""
    return _prepare_data(clean_data, integration_settings)


@pytest.fixture
def abt(prepared_data: PreparedData) -> pl.DataFrame:
    """Devuelve la Analytical Base Table a nivel de cliente."""
    return prepared_data.abt


@pytest.fixture
def train_test_abt(prepared_data: PreparedData) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Devuelve los conjuntos de entrenamiento y prueba de la ABT."""
    return prepared_data.train, prepared_data.test


@pytest.fixture
def rename_raw_data() -> Callable[[pl.DataFrame, config.Settings], pl.DataFrame]:
    """Devuelve un helper que renombra las columnas del dataset configuradas."""

    def _rename(raw_data: pl.DataFrame, settings: config.Settings) -> pl.DataFrame:
        return raw_data.rename(
            {
                "client_id": settings.dataset.id,
                "Target": settings.dataset.target,
                "Month": settings.dataset.month,
                "First_product_dt": settings.dataset.first_product_date,
                "Last_product_dt": settings.dataset.last_product_date,
            }
        )

    return _rename


@pytest.fixture(scope="module")
def custom_settings() -> config.Settings:
    """Devuelve la configuración usada por los tests de entrenamiento."""
    return config.Settings(
        analysis=config.AnalysisColumns(feature="Variable", importance="Weight"),
        n_jobs=1,
    )


@pytest.fixture(scope="module")
def synthetic_train(custom_settings: config.Settings) -> pl.DataFrame:
    """Genera el conjunto sintético compartido por los tests de entrenamiento."""
    return _build_synthetic_train(custom_settings.random_state)


@pytest.fixture(scope="module")
def trained_model(
    custom_settings: config.Settings,
    synthetic_train: pl.DataFrame,
) -> training.LGBMTrainer:
    """Entrena una única vez el modelo real usado por los tests de entrenamiento."""
    return training.LGBMTrainer.fit_and_build(
        synthetic_train,
        SYNTHETIC_FEATURE_COLUMNS,
        custom_settings,
        n_iter=SYNTHETIC_N_ITER,
        splits_cross_validation=SYNTHETIC_SPLITS_CROSS_VALIDATION,
    )
