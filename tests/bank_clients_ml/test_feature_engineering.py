"""Tests unitarios para el módulo de feature engineering."""

from collections.abc import Callable
from datetime import date

import polars as pl
import polars.testing
import pytest

from bank_clients_ml import column_groups, config, feature_engineering

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

EXTRA_TRANSFORMATION_COLUMNS: tuple[str, ...] = (
    "SavingAccount_Days_with_use_count_nonzero",
    "SavingAccount_Days_with_use_min",
    "SavingAccount_CreditCard_Payment_Transactions_count_nonzero",
    "Operations_total_count_nonzero",
    "CreditCard_Payment_total_max",
    "CreditCard_Payment_in_person_max",
    "Operations_in_person_pct_max",
    "CreditCard_Payment_Aut_Debit_max",
    "CreditCard_Payment_TAS_max",
    "SavingAccount_CreditCard_Payment_Amount_max",
)


def _build_float_frame(
    columns: tuple[str, ...], rows: list[dict[str, float]]
) -> pl.DataFrame:
    """Construye un DataFrame de Float64 completando con cero las columnas ausentes.

    ## Args:
        columns: Nombres y orden de las columnas del DataFrame.
        rows: Filas con los valores de las columnas que se desean definir.

    ## Returns:
        DataFrame con todas las columnas indicadas y tipo Float64.
    """
    return pl.DataFrame(
        [{column: row.get(column, 0.0) for column in columns} for row in rows],
        schema={column: pl.Float64 for column in columns},
    )


@pytest.fixture
def identity_features() -> pl.DataFrame:
    """Devuelve identity features con dos categorías y targets binarios.

    ## Returns:
        DataFrame con seis clientes, target balanceado y dos columnas categóricas.
    """
    return pl.DataFrame(
        {
            "client_id": [1, 2, 3, 4, 5, 6],
            "Target": [1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            "Region": ["A", "A", "A", "B", "B", "B"],
            "CreditCard_Product": ["X", "Y", "X", "Y", "X", "Y"],
        },
        schema={
            "client_id": pl.Int64,
            "Target": pl.Float64,
            "Region": pl.String,
            "CreditCard_Product": pl.String,
        },
    )


@pytest.fixture
def training_data() -> pl.DataFrame:
    """Devuelve historia mensual con las columnas categóricas a eliminar.

    ## Returns:
        DataFrame con identificador, mes, dos categóricas y un saldo.
    """
    return pl.DataFrame(
        {
            "client_id": [1, 2, 3],
            "Month": [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)],
            "Region": ["A", "B", "A"],
            "CreditCard_Product": ["X", "Y", "X"],
            "Balance": [100.0, 200.0, 300.0],
        },
        schema={
            "client_id": pl.Int64,
            "Month": pl.Date,
            "Region": pl.String,
            "CreditCard_Product": pl.String,
            "Balance": pl.Float64,
        },
    )


def test_should_encode_multiple_categorical_columns_and_drop_them_from_training_data(
    identity_features: pl.DataFrame,
    training_data: pl.DataFrame,
    settings: config.Settings,
) -> None:
    """Verifica la codificación de varias categóricas y su eliminación del histórico.

    ## Args:
        identity_features: Identity features con target y categorías.
        training_data: Historia mensual con las columnas categóricas.
        settings: Configuración con el nombre del target.
    """
    result, result_training = feature_engineering.target_encode_columns(
        identity_features=identity_features,
        training_data=training_data,
        categorical_cols=["Region", "CreditCard_Product"],
        settings=settings,
    )

    expected_region = [66.667, 66.667, 66.667, 33.333, 33.333, 33.333]
    expected_product = [66.667, 33.333, 66.667, 33.333, 66.667, 33.333]
    assert result["Region"].to_list() == pytest.approx(expected_region)
    assert result["CreditCard_Product"].to_list() == pytest.approx(expected_product)
    assert result["client_id"].to_list() == identity_features["client_id"].to_list()
    assert result_training.columns == ["client_id", "Month", "Balance"]
    assert result_training["Balance"].to_list() == [100.0, 200.0, 300.0]


def test_should_exclude_null_targets_from_encoding_denominator(
    settings: config.Settings,
) -> None:
    """Verifica que los targets nulos no formen parte del denominador.

    ## Args:
        settings: Configuración con el nombre del target.
    """
    identity = pl.DataFrame(
        {
            "client_id": [1, 2, 3, 4],
            "Target": [1.0, None, 0.0, None],
            "Region": ["A", "A", "A", "A"],
        },
        schema={
            "client_id": pl.Int64,
            "Target": pl.Float64,
            "Region": pl.String,
        },
    )
    training = pl.DataFrame(
        {"client_id": [1], "Region": ["A"]},
        schema={"client_id": pl.Int64, "Region": pl.String},
    )

    result, _ = feature_engineering.target_encode_columns(
        identity_features=identity,
        training_data=training,
        categorical_cols=["Region"],
        settings=settings,
    )

    assert result["Region"].to_list() == [50.0, 50.0, 50.0, 50.0]


def test_should_raise_zero_division_error_when_category_has_no_binary_targets(
    settings: config.Settings,
) -> None:
    """Verifica que se rechace una categoría sin targets binarios.

    ## Args:
        settings: Configuración con el nombre del target.
    """
    identity = pl.DataFrame(
        {
            "client_id": [1, 2, 3, 4],
            "Target": [None, None, 1.0, 0.0],
            "Region": ["A", "A", "B", "B"],
        },
        schema={
            "client_id": pl.Int64,
            "Target": pl.Float64,
            "Region": pl.String,
        },
    )
    training = pl.DataFrame(
        {"client_id": [1], "Region": ["A"]},
        schema={"client_id": pl.Int64, "Region": pl.String},
    )

    with pytest.raises(ZeroDivisionError, match="Division by zero"):
        feature_engineering.target_encode_columns(
            identity_features=identity,
            training_data=training,
            categorical_cols=["Region"],
            settings=settings,
        )


def test_should_use_default_categorical_columns_from_column_groups(
    settings: config.Settings,
) -> None:
    """Verifica que se usen las categóricas por defecto si no se indican.

    ## Args:
        settings: Configuración con el nombre del target.
    """
    identity = pl.DataFrame(
        {
            "client_id": [1, 2],
            "Target": [1.0, 0.0],
            "Client_Age_grp": ["young", "old"],
            "Region": ["N", "S"],
            "CreditCard_Product": ["A", "B"],
        },
        schema={
            "client_id": pl.Int64,
            "Target": pl.Float64,
            "Client_Age_grp": pl.String,
            "Region": pl.String,
            "CreditCard_Product": pl.String,
        },
    )
    training = pl.DataFrame(
        {
            "client_id": [1],
            "Client_Age_grp": ["young"],
            "Region": ["N"],
            "CreditCard_Product": ["A"],
        },
        schema={
            "client_id": pl.Int64,
            "Client_Age_grp": pl.String,
            "Region": pl.String,
            "CreditCard_Product": pl.String,
        },
    )

    result, result_training = feature_engineering.target_encode_columns(
        identity_features=identity,
        training_data=training,
        settings=settings,
    )

    for column in column_groups.CATEGORICAL_COLS:
        assert result[column].to_list() == [100.0, 0.0]
        assert column not in result_training.columns


def test_should_use_custom_target_column_in_target_encoding(
    identity_features: pl.DataFrame,
    training_data: pl.DataFrame,
    custom_dataset_settings: config.Settings,
) -> None:
    """Verifica que se use el target configurado para calcular los porcentajes.

    ## Args:
        identity_features: Identity features con target y categorías.
        training_data: Historia mensual con las columnas categóricas.
        custom_dataset_settings: Configuración con el target alternativo.
    """
    renamed_identity = identity_features.rename({"Target": "churn"})

    result, _ = feature_engineering.target_encode_columns(
        identity_features=renamed_identity,
        training_data=training_data,
        categorical_cols=["Region"],
        settings=custom_dataset_settings,
    )

    expected = [66.667, 66.667, 66.667, 33.333, 33.333, 33.333]
    assert result["Region"].to_list() == pytest.approx(expected)


def test_should_use_global_settings_when_settings_is_not_provided_for_target_encoding(
    identity_features: pl.DataFrame,
    training_data: pl.DataFrame,
    custom_dataset_settings: config.Settings,
    patch_global_settings: Callable[[config.Settings], None],
) -> None:
    """Verifica que se use la configuración global si no se recibe una explícita.

    ## Args:
        identity_features: Identity features con target y categorías.
        training_data: Historia mensual con las columnas categóricas.
        custom_dataset_settings: Configuración con el target alternativo.
        patch_global_settings: Helper para reemplazar la configuración global.
    """
    patch_global_settings(custom_dataset_settings)
    renamed_identity = identity_features.rename({"Target": "churn"})

    result, _ = feature_engineering.target_encode_columns(
        identity_features=renamed_identity,
        training_data=training_data,
        categorical_cols=["Region"],
    )

    expected = [66.667, 66.667, 66.667, 33.333, 33.333, 33.333]
    assert result["Region"].to_list() == pytest.approx(expected)


def test_should_compute_days_between_product_dates_and_recency(
    settings: config.Settings,
) -> None:
    """Verifica los días entre fechas de producto y la recencia al mes siguiente.

    ## Args:
        settings: Configuración con los nombres de las columnas.
    """
    data = pl.DataFrame(
        {
            "client_id": [1, 2],
            "First_product_dt": [date(2023, 1, 1), date(2024, 1, 1)],
            "Last_product_dt": [date(2024, 1, 1), date(2024, 6, 1)],
            "Balance": [10.0, 20.0],
        },
        schema={
            "client_id": pl.Int64,
            "First_product_dt": pl.Date,
            "Last_product_dt": pl.Date,
            "Balance": pl.Float64,
        },
    )

    result = feature_engineering.transform_date_features(
        data, date(2024, 6, 1), settings=settings
    )

    assert result["Days_between_first_and_last_product"].to_list() == [365, 152]
    assert result["Recency_in_days"].to_list() == [182, 30]
    assert result.columns == [
        "client_id",
        "Balance",
        "Days_between_first_and_last_product",
        "Recency_in_days",
    ]


def test_should_compute_recency_across_year_boundary(
    settings: config.Settings,
) -> None:
    """Verifica que la recencia se calcule correctamente al cambiar de año.

    ## Args:
        settings: Configuración con los nombres de las columnas.
    """
    data = pl.DataFrame(
        {
            "client_id": [1],
            "First_product_dt": [date(2024, 12, 15)],
            "Last_product_dt": [date(2024, 12, 15)],
        },
        schema={
            "client_id": pl.Int64,
            "First_product_dt": pl.Date,
            "Last_product_dt": pl.Date,
        },
    )

    result = feature_engineering.transform_date_features(
        data, date(2024, 12, 1), settings=settings
    )

    assert result["Days_between_first_and_last_product"].item() == 0
    assert result["Recency_in_days"].item() == 17


def test_should_use_custom_date_columns_in_date_features(
    custom_dataset_settings: config.Settings,
) -> None:
    """Verifica que se usen las columnas de fecha configuradas.

    ## Args:
        custom_dataset_settings: Configuración con las fechas alternativas.
    """
    data = pl.DataFrame(
        {
            "customer_id": [7],
            "first_seen": [date(2024, 1, 1)],
            "last_seen": [date(2024, 6, 1)],
        },
        schema={
            "customer_id": pl.Int64,
            "first_seen": pl.Date,
            "last_seen": pl.Date,
        },
    )

    result = feature_engineering.transform_date_features(
        data, date(2024, 6, 1), settings=custom_dataset_settings
    )

    assert result["Days_between_first_and_last_product"].item() == 152
    assert result["Recency_in_days"].item() == 30
    assert "first_seen" not in result.columns
    assert "last_seen" not in result.columns


def test_should_use_global_settings_when_settings_is_not_provided_for_date_features(
    custom_dataset_settings: config.Settings,
    patch_global_settings: Callable[[config.Settings], None],
) -> None:
    """Verifica que se use la configuración global si no se recibe una explícita.

    ## Args:
        custom_dataset_settings: Configuración con las fechas alternativas.
        patch_global_settings: Helper para reemplazar la configuración global.
    """
    patch_global_settings(custom_dataset_settings)
    data = pl.DataFrame(
        {
            "customer_id": [7],
            "first_seen": [date(2024, 1, 1)],
            "last_seen": [date(2024, 6, 1)],
        },
        schema={
            "customer_id": pl.Int64,
            "first_seen": pl.Date,
            "last_seen": pl.Date,
        },
    )

    result = feature_engineering.transform_date_features(data, date(2024, 6, 1))

    assert result["Days_between_first_and_last_product"].item() == 152
    assert result["Recency_in_days"].item() == 30


def test_should_replace_zero_denominator_with_one() -> None:
    """Verifica que un denominador cero se reemplace por uno."""
    data = pl.DataFrame({"denominator": [0.0, 2.0, 0.0]})

    result = data.select(feature_engineering._safe_denominator("denominator"))

    assert result["denominator"].to_list() == [1.0, 2.0, 1.0]


def test_should_compute_percentage_from_expressions() -> None:
    """Verifica que el porcentaje acepte expresiones como numerador."""
    data = pl.DataFrame({"numerator": [10.0], "denominator": [4.0]})

    result = data.select(
        feature_engineering._compute_percentage(
            pl.col("numerator") * 2, pl.col("denominator")
        ).alias("percentage")
    )

    assert result["percentage"].item() == 500.0


def test_should_return_zero_for_constant_column_in_min_max_normalization() -> None:
    """Verifica que una columna constante se normalice a cero."""
    data = pl.DataFrame({"value": [5.0, 5.0]})

    result = data.select(
        feature_engineering._min_max_normalize("value").alias("normalized")
    )

    assert result["normalized"].to_list() == [0.0, 0.0]


def test_should_multiply_normalized_value_by_weight() -> None:
    """Verifica que la normalización ponderada multiplique por el peso."""
    data = pl.DataFrame({"value": [1.0, 2.0, 3.0], "weight": [10.0, 20.0, 30.0]})

    result = data.select(
        feature_engineering._min_max_normalize_weighted("value", "weight").alias(
            "weighted"
        )
    )

    assert result["weighted"].to_list() == [0.0, 10.0, 30.0]


def test_should_generate_expected_monthly_transformations() -> None:
    """Verifica todas las variables mensuales contra una salida esperada.

    Usa una fila con valores distintos por canal, tipo y producto para detectar
    columnas mal calculadas, fórmulas intercambiadas y columnas faltantes o extra.
    """
    data = _build_float_frame(
        MONTHLY_COLUMNS,
        [
            {
                "Operations_Bank": 1.0,
                "Operations_Terminal": 2.0,
                "Operations_HomeBanking": 3.0,
                "Operations_Mobile": 4.0,
                "Operations_Ivr": 5.0,
                "Operations_Telemarketer": 6.0,
                "Operations_ATM": 7.0,
                "CreditCard_Payment_Aut_Debit": 1.0,
                "CreditCard_Payment_External": 2.0,
                "CreditCard_Payment_Cash": 3.0,
                "CreditCard_Payment_Web": 4.0,
                "CreditCard_Payment_ATM": 5.0,
                "CreditCard_Payment_TAS": 6.0,
                "SavingAccount_Balance_LastDate": 150.0,
                "SavingAccount_Balance_FirstDate": 100.0,
                "SavingAccount_Days_with_Debits": 4.0,
                "SavingAccount_Days_with_use": 10.0,
                "SavingAccount_Days_with_Credits": 6.0,
                "SavingAccount_Credits_Transactions": 8.0,
                "SavingAccount_Debits_Transactions": 2.0,
                "SavingAccount_Transactions_Transactions": 10.0,
                "SavingAccount_Salary_Payment_Transactions": 10.0,
                "SavingAccount_Transfer_In_Transactions": 20.0,
                "SavingAccount_ATM_Extraction_Transactions": 20.0,
                "SavingAccount_Service_Payment_Transactions": 15.0,
                "SavingAccount_CreditCard_Payment_Transactions": 5.0,
                "SavingAccount_Transfer_Out_Transactions": 10.0,
                "SavingAccount_DebitCard_Spend_Transactions": 10.0,
                "SavingAccount_Credits_Amounts": 400.0,
                "SavingAccount_Debits_Amounts": 600.0,
                "SavingAccount_Total_Amount": 1000.0,
                "SavingAccount_Salary_Payment_Amount": 100.0,
                "SavingAccount_Transfer_In_Amount": 200.0,
                "SavingAccount_ATM_Extraction_Amount": 200.0,
                "SavingAccount_Service_Payment_Amount": 150.0,
                "SavingAccount_CreditCard_Payment_Amount": 50.0,
                "SavingAccount_Transfer_Out_Amount": 100.0,
                "SavingAccount_DebitCard_Spend_Amount": 100.0,
                "CreditCard_Balance_ARG": 100.0,
                "CreditCard_Balance_DOLLAR": 200.0,
                "CreditCard_Total_Limit": 1000.0,
                "CreditCard_Total_Spending": 500.0,
                "CreditCard_Spending_1_Installment": 100.0,
                "CreditCard_Spending_Installments": 200.0,
                "CreditCard_Spending_CrossBoarder": 50.0,
                "CreditCard_Spending_Aut_Debits": 100.0,
                "CreditCard_Revolving": 50.0,
                "CreditCard_Premium": 1.0,
                "CreditCard_Active": 1.0,
                "Loan_Active": 1.0,
                "Mortgage_Active": 0.0,
                "SavingAccount_Active_ARG_Salary": 1.0,
                "SavingAccount_Active_ARG": 1.0,
                "SavingAccount_Active_DOLLAR": 1.0,
                "DebitCard_Active": 1.0,
                "Investment_Active": 0.0,
                "Insurance_Life": 1.0,
            }
        ],
    )

    result = feature_engineering.with_transformations(data)

    expected_values: dict[str, list[float]] = {
        "Operations_total": [28.0],
        "Operations_remote": [18.0],
        "Operations_in_person": [10.0],
        "CreditCard_Payment_total": [21.0],
        "CreditCard_Payment_remote": [5.0],
        "CreditCard_Payment_in_person": [16.0],
        "SavingAccount_Balance_last_minus_first_date": [50.0],
        "SavingAccount_Balance_last_minus_first_date_pct": [150.0],
        "SavingAccount_Days_with_Debits_pct": [40.0],
        "SavingAccount_Days_with_Credits_pct": [60.0],
        "SavingAccount_Credits_Transactions_pct": [80.0],
        "SavingAccount_Debits_Transactions_pct": [20.0],
        "SavingAccount_Transactions_Transactions_DAYS_prom": [1.0],
        "SavingAccount_Credits_Transactions_DAYS_prom": [8 / 6],
        "SavingAccount_Debits_Transactions_DAYS_prom": [0.5],
        "SavingAccount_Salary_Payment_Transactions_pct": [100.0],
        "SavingAccount_Transfer_In_Transactions_pct": [200.0],
        "SavingAccount_ATM_Extraction_Transactions_pct": [200.0],
        "SavingAccount_Service_Payment_Transactions_pct": [150.0],
        "SavingAccount_CreditCard_Payment_Transactions_pct": [50.0],
        "SavingAccount_Transfer_Out_Transactions_pct": [100.0],
        "SavingAccount_DebitCard_Spend_Transactions_pct": [100.0],
        "SavingAccount_Salary_Payment_Transactions_CR_pct": [125.0],
        "SavingAccount_Transfer_In_Transactions_CR_pct": [250.0],
        "SavingAccount_ATM_Extraction_Transactions_DE_pct": [1000.0],
        "SavingAccount_Service_Payment_Transactions_DE_pct": [750.0],
        "SavingAccount_CreditCard_Payment_Transactions_DE_pct": [250.0],
        "SavingAccount_Transfer_Out_Transactions_DE_pct": [500.0],
        "SavingAccount_DebitCard_Spend_Transactions_DE_pct": [500.0],
        "SavingAccount_Credits_Amounts_pct": [40.0],
        "SavingAccount_Debits_Amounts_pct": [60.0],
        "SavingAccount_Salary_Payment_Amount_pct": [10.0],
        "SavingAccount_Transfer_In_Amount_pct": [20.0],
        "SavingAccount_ATM_Extraction_Amount_pct": [20.0],
        "SavingAccount_Service_Payment_Amount_pct": [15.0],
        "SavingAccount_CreditCard_Payment_Amount_pct": [5.0],
        "SavingAccount_Transfer_Out_Amount_pct": [10.0],
        "SavingAccount_DebitCard_Spend_Amount_pct": [10.0],
        "SavingAccount_Salary_Payment_Amount_CR_pct": [25.0],
        "SavingAccount_Transfer_In_Amount_CR_pct": [50.0],
        "SavingAccount_ATM_Extraction_Amount_DE_pct": [200 / 600 * 100],
        "SavingAccount_Service_Payment_Amount_DE_pct": [25.0],
        "SavingAccount_CreditCard_Payment_Amount_DE_pct": [50 / 600 * 100],
        "SavingAccount_Transfer_Out_Amount_DE_pct": [100 / 600 * 100],
        "SavingAccount_DebitCard_Spend_Amount_DE_pct": [100 / 600 * 100],
        "Operations_remote_pct": [18 / 28 * 100],
        "Operations_in_person_pct": [10 / 28 * 100],
        "Operations_Bank_pct": [1 / 28 * 100],
        "Operations_Terminal_pct": [2 / 28 * 100],
        "Operations_HomeBanking_pct": [3 / 28 * 100],
        "Operations_Mobile_pct": [4 / 28 * 100],
        "Operations_Ivr_pct": [5 / 28 * 100],
        "Operations_Telemarketer_pct": [6 / 28 * 100],
        "Operations_ATM_pct": [7 / 28 * 100],
        "Operations_Bank_IP_pct": [10.0],
        "Operations_Terminal_IP_pct": [20.0],
        "Operations_HomeBanking_R_pct": [3 / 18 * 100],
        "Operations_Mobile_R_pct": [4 / 18 * 100],
        "Operations_Ivr_R_pct": [5 / 18 * 100],
        "Operations_Telemarketer_R_pct": [6 / 18 * 100],
        "Operations_ATM_IP_pct": [70.0],
        "CreditCard_Payment_remote_pct": [5 / 21 * 100],
        "CreditCard_Payment_in_person_pct": [16 / 21 * 100],
        "CreditCard_Payment_Aut_Debit_pct": [1 / 21 * 100],
        "CreditCard_Payment_External_pct": [2 / 21 * 100],
        "CreditCard_Payment_Cash_pct": [3 / 21 * 100],
        "CreditCard_Payment_Web_pct": [4 / 21 * 100],
        "CreditCard_Payment_ATM_pct": [5 / 21 * 100],
        "CreditCard_Payment_TAS_pct": [6 / 21 * 100],
        "CreditCard_Payment_Aut_Debit_R_pct": [20.0],
        "CreditCard_Payment_External_IP_pct": [12.5],
        "CreditCard_Payment_Cash_IP_pct": [18.75],
        "CreditCard_Payment_Web_R_pct": [80.0],
        "CreditCard_Payment_ATM_IP_pct": [31.25],
        "CreditCard_Payment_TAS_IP_pct": [37.5],
        "CreditCard_Balance_ARG_limit_pct": [10.0],
        "CreditCard_Balance_DOLLAR_limit_pct": [20.0],
        "CreditCard_Total_Spending_limit_pct": [50.0],
        "CreditCard_Spending_1_Installment_limit_pct": [10.0],
        "CreditCard_Spending_Installments_limit_pct": [20.0],
        "CreditCard_Spending_CrossBoarder_limit_pct": [5.0],
        "CreditCard_Spending_Aut_Debits_limit_pct": [10.0],
        "CreditCard_Revolving_limit_pct": [5.0],
        "CreditCard_Balance_ARG_SP_pct": [20.0],
        "CreditCard_Balance_DOLLAR_SP_pct": [40.0],
        "CreditCard_Spending_1_Installment_SP_pct": [20.0],
        "CreditCard_Spending_Installments_SP_pct": [40.0],
        "CreditCard_Spending_CrossBoarder_SP_pct": [10.0],
        "CreditCard_Spending_Aut_Debits_SP_pct": [20.0],
        "CreditCard_Revolving_SP_pct": [10.0],
        "Quantity_Active_Products": [8.0],
        "Quantity_Common_Active_Product": [4.0],
        "Quantity_Most_Common_Active_Product": [3.0],
    }
    expected = pl.DataFrame(
        expected_values,
        schema={column: pl.Float64 for column in expected_values},
    )

    assert result.height == 1
    assert result.columns == [*MONTHLY_COLUMNS, *expected.columns]
    polars.testing.assert_frame_equal(result.select(expected.columns), expected)


def test_should_protect_divisions_by_zero_in_monthly_transformations() -> None:
    """Verifica que las divisiones por cero no produzcan valores infinitos."""
    data = _build_float_frame(MONTHLY_COLUMNS, [{}])

    result = feature_engineering.with_transformations(data)

    percentage_columns = [
        column for column in result.columns if column.endswith(("_pct", "_prom"))
    ]
    assert percentage_columns
    for column in percentage_columns:
        assert result[column].null_count() == 0
        assert result[column].is_finite().all()


@pytest.fixture
def monthly_history() -> pl.DataFrame:
    """Devuelve historia mensual desordenada para dos clientes.

    ## Returns:
        DataFrame con tres meses del cliente 1 y dos del cliente 2.
    """
    return pl.DataFrame(
        {
            "client_id": [1, 1, 1, 2, 2],
            "Month": [
                date(2024, 3, 1),
                date(2024, 1, 1),
                date(2024, 2, 1),
                date(2024, 1, 1),
                date(2024, 2, 1),
            ],
            "Operations_total": [30.0, 10.0, 20.0, 5.0, 5.0],
            "Balance": [3000.0, 1000.0, 2000.0, 1000.0, 1000.0],
        },
        schema={
            "client_id": pl.Int64,
            "Month": pl.Date,
            "Operations_total": pl.Float64,
            "Balance": pl.Float64,
        },
    )


def test_should_aggregate_monthly_history_and_compute_quantity_statistics(
    monthly_history: pl.DataFrame,
    settings: config.Settings,
) -> None:
    """Verifica la agregación a una fila por cliente y sus estadísticas mensuales.

    ## Args:
        monthly_history: Historia mensual de dos clientes.
        settings: Configuración con los nombres de las columnas.
    """
    result = feature_engineering.aggregate_monthly_to_client(
        monthly_history,
        columns_with_monetary_values=["Balance"],
        columns_with_quantities=["Operations_total"],
        settings=settings,
    ).sort("client_id")
    client_1 = result.filter(pl.col("client_id") == 1)
    client_2 = result.filter(pl.col("client_id") == 2)

    assert result.height == 2
    assert result["client_id"].to_list() == [1, 2]

    assert client_1["Operations_total_min"].item() == 10.0
    assert client_1["Operations_total_max"].item() == 30.0
    assert client_1["Operations_total_mean"].item() == 20.0
    assert client_1["Operations_total_median"].item() == 20.0
    assert client_1["Operations_total_sum"].item() == 60.0
    assert client_1["Operations_total_count_nonzero"].item() == 3
    assert client_1["Operations_total_var"].item() == 100.0
    assert client_1["Operations_total_std"].item() == 10.0
    assert client_1["Operations_total_nunique"].item() == 3
    assert client_1["Operations_total_ptp"].item() == 20.0
    assert client_1["Operations_total_diff"].item() == 20.0

    assert client_2["Operations_total_min"].item() == 5.0
    assert client_2["Operations_total_max"].item() == 5.0
    assert client_2["Operations_total_sum"].item() == 10.0
    assert client_2["Operations_total_count_nonzero"].item() == 2
    assert client_2["Operations_total_var"].item() == 0.0
    assert client_2["Operations_total_std"].item() == 0.0
    assert client_2["Operations_total_nunique"].item() == 1
    assert client_2["Operations_total_diff"].item() == 0.0


def test_should_round_monetary_values_before_counting_unique_values(
    settings: config.Settings,
) -> None:
    """Verifica que los montos se redondeen antes de contar valores únicos.

    ## Args:
        settings: Configuración con los nombres de las columnas.
    """
    data = pl.DataFrame(
        {
            "client_id": [1, 1, 1],
            "Month": [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)],
            "Operations_total": [1.0, 2.0, 3.0],
            "Balance": [1000.0, 1400.0, 2400.0],
        },
        schema={
            "client_id": pl.Int64,
            "Month": pl.Date,
            "Operations_total": pl.Float64,
            "Balance": pl.Float64,
        },
    )

    result = feature_engineering.aggregate_monthly_to_client(
        data,
        columns_with_monetary_values=["Balance"],
        columns_with_quantities=["Operations_total"],
        settings=settings,
    )

    assert result["Balance_rounded_nunique"].item() == 2
    assert result["Operations_total_nunique"].item() == 3
    assert "Balance_nunique" not in result.columns
    assert "Operations_total_rounded_nunique" not in result.columns


def test_should_use_custom_identifier_and_month_columns_in_aggregation(
    monthly_history: pl.DataFrame,
    custom_dataset_settings: config.Settings,
) -> None:
    """Verifica que se usen el identificador y el mes configurados.

    ## Args:
        monthly_history: Historia mensual de dos clientes.
        custom_dataset_settings: Configuración con identificador y mes alternativos.
    """
    data = monthly_history.rename(
        {"client_id": "customer_id", "Month": "snapshot_month"}
    )

    result = feature_engineering.aggregate_monthly_to_client(
        data,
        columns_with_monetary_values=["Balance"],
        columns_with_quantities=["Operations_total"],
        settings=custom_dataset_settings,
    ).sort("customer_id")

    assert result["customer_id"].to_list() == [1, 2]
    client_1 = result.filter(pl.col("customer_id") == 1)
    assert client_1["Operations_total_diff"].item() == 20.0


def test_should_use_global_settings_when_settings_is_not_provided_for_aggregation(
    monthly_history: pl.DataFrame,
    custom_dataset_settings: config.Settings,
    patch_global_settings: Callable[[config.Settings], None],
) -> None:
    """Verifica que se use la configuración global si no se recibe una explícita.

    ## Args:
        monthly_history: Historia mensual de dos clientes.
        custom_dataset_settings: Configuración con identificador y mes alternativos.
        patch_global_settings: Helper para reemplazar la configuración global.
    """
    patch_global_settings(custom_dataset_settings)
    data = monthly_history.rename(
        {"client_id": "customer_id", "Month": "snapshot_month"}
    )

    result = feature_engineering.aggregate_monthly_to_client(
        data,
        columns_with_monetary_values=["Balance"],
        columns_with_quantities=["Operations_total"],
    ).sort("customer_id")

    client_1 = result.filter(pl.col("customer_id") == 1)
    assert client_1["Operations_total_diff"].item() == 20.0


def test_should_use_default_column_groups_when_columns_are_not_provided(
    settings: config.Settings,
) -> None:
    """Verifica que se usen las columnas monetarias y de cantidades por defecto.

    ## Args:
        settings: Configuración con los nombres de las columnas.
    """
    data = pl.DataFrame(
        {
            "client_id": [1, 1],
            "Month": [date(2024, 1, 1), date(2024, 2, 1)],
            "Operations_total": [10.0, 20.0],
            **{
                column: [0.0, 0.0]
                for column in column_groups.COLUMNS_WITH_MONETARY_VALUES
            },
        }
    )

    result = feature_engineering.aggregate_monthly_to_client(data, settings=settings)

    assert result["Operations_total_min"].item() == 10.0
    assert result["Operations_total_diff"].item() == 10.0
    assert result["SavingAccount_Balance_Average_min"].item() == 0.0


@pytest.fixture
def extra_transformations_data() -> pl.DataFrame:
    """Devuelve dos filas con indicadores de uso en cero y diez.

    ## Returns:
        DataFrame con todas las columnas requeridas por las transformaciones extra.
    """
    return _build_float_frame(
        EXTRA_TRANSFORMATION_COLUMNS,
        [
            {column: 0.0 for column in EXTRA_TRANSFORMATION_COLUMNS},
            {column: 10.0 for column in EXTRA_TRANSFORMATION_COLUMNS},
        ],
    )


def test_should_sum_normalized_use_indicators_and_binary_flags(
    extra_transformations_data: pl.DataFrame,
) -> None:
    """Verifica la suma de indicadores normalizados y banderas binarias.

    ## Args:
        extra_transformations_data: Datos con indicadores de uso en cero y diez.
    """
    result = feature_engineering.with_extra_transformations(extra_transformations_data)

    assert result["SUM_OF_USES"].to_list() == pytest.approx([0.0, 9.0])


def test_should_weight_normalized_amount_by_operation_and_transaction_counts(
    extra_transformations_data: pl.DataFrame,
) -> None:
    """Verifica el monto normalizado ponderado por operaciones y transacciones.

    ## Args:
        extra_transformations_data: Datos con indicadores de uso en cero y diez.
    """
    result = feature_engineering.with_extra_transformations(extra_transformations_data)

    assert result["Amount_operations"].to_list() == pytest.approx([0.0, 10.0])
    assert result["Amount_transactions"].to_list() == pytest.approx([0.0, 10.0])
    assert result["Amount_payment"].to_list() == pytest.approx([0.0, 10.0])


def test_should_return_zero_for_constant_columns_in_sum_of_uses() -> None:
    """Verifica que las columnas constantes no aporten a la suma de usos."""
    data = _build_float_frame(
        EXTRA_TRANSFORMATION_COLUMNS,
        [{column: 5.0 for column in EXTRA_TRANSFORMATION_COLUMNS}] * 2,
    ).with_columns(
        pl.lit(0.0).alias("Operations_in_person_pct_max"),
        pl.lit(0.0).alias("CreditCard_Payment_Aut_Debit_max"),
        pl.lit(0.0).alias("CreditCard_Payment_TAS_max"),
    )

    result = feature_engineering.with_extra_transformations(data)

    assert result["SUM_OF_USES"].to_list() == [0.0, 0.0]


def test_should_standardize_numeric_columns_with_population_standard_deviation(
    settings: config.Settings,
) -> None:
    """Verifica la estandarización con desvío estándar poblacional.

    ## Args:
        settings: Configuración con los nombres de identificador y target.
    """
    data = pl.DataFrame(
        {
            "client_id": [1, 2, 3],
            "Target": [0.0, 1.0, 0.0],
            "Feature": [1.0, 2.0, 3.0],
            "Other": [10.0, 20.0, 30.0],
        },
        schema={
            "client_id": pl.Int64,
            "Target": pl.Float64,
            "Feature": pl.Float64,
            "Other": pl.Float64,
        },
    )

    result = feature_engineering.standardize(data, ddof=0, settings=settings)

    expected = [-1.224744871, 0.0, 1.224744871]
    assert result["Feature"].to_list() == pytest.approx(expected)
    assert result["Other"].to_list() == pytest.approx(expected)
    assert result["client_id"].to_list() == [1, 2, 3]
    assert result["Target"].to_list() == [0.0, 1.0, 0.0]


def test_should_standardize_with_sample_standard_deviation_when_ddof_is_one(
    settings: config.Settings,
) -> None:
    """Verifica la estandarización con desvío estándar de la muestra.

    ## Args:
        settings: Configuración con los nombres de identificador y target.
    """
    data = pl.DataFrame(
        {
            "client_id": [1, 2, 3],
            "Target": [0.0, 1.0, 0.0],
            "Feature": [1.0, 2.0, 3.0],
            "Other": [10.0, 20.0, 30.0],
        },
        schema={
            "client_id": pl.Int64,
            "Target": pl.Float64,
            "Feature": pl.Float64,
            "Other": pl.Float64,
        },
    )

    result = feature_engineering.standardize(data, ddof=1, settings=settings)

    assert result["Feature"].to_list() == pytest.approx([-1.0, 0.0, 1.0])
    assert result["Other"].to_list() == pytest.approx([-1.0, 0.0, 1.0])


def test_should_use_custom_identifier_and_target_columns_in_standardization(
    custom_dataset_settings: config.Settings,
) -> None:
    """Verifica que se excluyan el identificador y el target configurados.

    ## Args:
        custom_dataset_settings: Configuración con identificador y target alternativos.
    """
    data = pl.DataFrame(
        {
            "customer_id": [10, 20, 30],
            "churn": [1.0, 0.0, 1.0],
            "Feature": [1.0, 2.0, 3.0],
        },
        schema={
            "customer_id": pl.Int64,
            "churn": pl.Float64,
            "Feature": pl.Float64,
        },
    )

    result = feature_engineering.standardize(
        data, ddof=0, settings=custom_dataset_settings
    )

    assert result["customer_id"].to_list() == [10, 20, 30]
    assert result["churn"].to_list() == [1.0, 0.0, 1.0]
    assert result["Feature"].to_list() == pytest.approx(
        [-1.224744871, 0.0, 1.224744871]
    )


def test_should_use_global_settings_when_settings_is_not_provided_for_standardization(
    custom_dataset_settings: config.Settings,
    patch_global_settings: Callable[[config.Settings], None],
) -> None:
    """Verifica que se use la configuración global si no se recibe una explícita.

    ## Args:
        custom_dataset_settings: Configuración con identificador y target alternativos.
        patch_global_settings: Helper para reemplazar la configuración global.
    """
    patch_global_settings(custom_dataset_settings)
    data = pl.DataFrame(
        {
            "customer_id": [10, 20, 30],
            "churn": [1.0, 0.0, 1.0],
            "Feature": [1.0, 2.0, 3.0],
        },
        schema={
            "customer_id": pl.Int64,
            "churn": pl.Float64,
            "Feature": pl.Float64,
        },
    )

    result = feature_engineering.standardize(data, ddof=0)

    assert result["customer_id"].to_list() == [10, 20, 30]
    assert result["churn"].to_list() == [1.0, 0.0, 1.0]
    assert result["Feature"].to_list() == pytest.approx(
        [-1.224744871, 0.0, 1.224744871]
    )
