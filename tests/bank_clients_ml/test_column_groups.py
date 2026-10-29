"""Tests unitarios para el módulo de agrupación de columnas por fuente."""

import types

import polars as pl
import pytest

from bank_clients_ml import column_groups, config


@pytest.fixture
def custom_settings() -> config.Settings:
    """Devuelve una configuración con nombres de columnas personalizados.

    ## Returns:
        Configuración con identificador, target, mes y fechas no estándar.
    """
    return config.Settings(
        dataset=config.DatasetColumns(
            id="customer_id",
            target="churn",
            month="snapshot_month",
            first_product_date="first_product_dt",
            last_product_date="last_product_dt",
        )
    )


def _build_generated_groups(
    groups: dict[column_groups.ValidColumnGroup, tuple[str, ...]],
) -> column_groups.GeneratedColumnGroups:
    """Construye un agrupamiento completando con tuplas vacías los grupos ausentes.

    ## Args:
        groups: Grupos con las columnas que se desean validar.

    ## Returns:
        Agrupamiento generado con todos los grupos del enum.
    """
    complete_groups: dict[column_groups.ValidColumnGroup, tuple[str, ...]] = {
        group: () for group in column_groups.ValidColumnGroup
    }
    complete_groups.update(groups)
    return column_groups.GeneratedColumnGroups(types.MappingProxyType(complete_groups))


def test_should_return_identity_features_in_expected_order(
    settings: config.Settings,
) -> None:
    """Verifica que las identity features se compongan en el orden esperado."""
    expected = (
        settings.dataset.id,
        settings.dataset.target,
        *column_groups.CATEGORICAL_COLS,
        settings.dataset.first_product_date,
        settings.dataset.last_product_date,
        *column_groups.BINARY_IDENTITY_FEATURES_COLS,
    )

    result = column_groups.get_identity_features_cols(settings)

    assert result == expected


def test_should_use_custom_dataset_names_in_identity_features(
    custom_settings: config.Settings,
) -> None:
    """Verifica que se respeten los nombres personalizados del dataset."""
    expected = (
        "customer_id",
        "churn",
        *column_groups.CATEGORICAL_COLS,
        "first_product_dt",
        "last_product_dt",
        *column_groups.BINARY_IDENTITY_FEATURES_COLS,
    )

    result = column_groups.get_identity_features_cols(custom_settings)

    assert result == expected


def test_should_use_global_settings_when_none_is_provided(
    custom_settings: config.Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifica que se use la configuración global si no se recibe una explícita."""
    expected = (
        "customer_id",
        "churn",
        *column_groups.CATEGORICAL_COLS,
        "first_product_dt",
        "last_product_dt",
        *column_groups.BINARY_IDENTITY_FEATURES_COLS,
    )
    monkeypatch.setattr(config, "get_settings", lambda: custom_settings)

    result = column_groups.get_identity_features_cols()

    assert result == expected


def test_should_return_columns_with_quantities_in_dataframe_order(
    settings: config.Settings,
) -> None:
    """Verifica que se excluyan las columnas sin cantidades y se conserve el orden."""
    raw_columns = [
        settings.dataset.id,
        "Operations_total_mean",
        settings.dataset.target,
        "SavingAccount_Balance_Average",
        settings.dataset.month,
        "custom_quantity",
        settings.dataset.first_product_date,
        "SavingAccount_Balance_last_minus_first_date",
        settings.dataset.last_product_date,
        "CreditCard_Balance_ARG",
        "Region",
        "CreditCard_Premium",
        "another_quantity",
    ]
    dataframe = pl.DataFrame({column: [1] for column in raw_columns})
    expected = ("Operations_total_mean", "custom_quantity", "another_quantity")

    result = column_groups.get_columns_with_quantities(dataframe, settings)

    assert result == expected


def test_should_use_custom_settings_to_exclude_columns_without_quantities(
    custom_settings: config.Settings,
) -> None:
    """Verifica que las exclusiones respeten los nombres personalizados del dataset."""
    dataframe = pl.DataFrame(
        {
            "customer_id": [1],
            "churn": [0],
            "snapshot_month": [1],
            "first_product_dt": [1],
            "last_product_dt": [1],
            "Operations_total_mean": [1],
            "Client_Age_grp": ["young"],
            "Sex": [1],
            "SavingAccount_Balance_Average": [1.0],
        }
    )

    result = column_groups.get_columns_with_quantities(dataframe, custom_settings)

    assert result == ("Operations_total_mean",)


def test_should_return_no_quantities_when_dataframe_is_empty(
    settings: config.Settings,
) -> None:
    """Verifica que un DataFrame vacío no produzca columnas con cantidades."""
    result = column_groups.get_columns_with_quantities(pl.DataFrame(), settings)

    assert result == ()


def test_should_return_all_valid_groups_in_definition_order() -> None:
    """Verifica que se devuelvan los grupos válidos en el orden declarado."""
    expected = (
        column_groups.ValidColumnGroup.ALL,
        column_groups.ValidColumnGroup.SAVING_ACCOUNT_DAYS_TRANSACTIONS,
        column_groups.ValidColumnGroup.SAVING_ACCOUNT_MONETARY,
        column_groups.ValidColumnGroup.OPERATIONS,
        column_groups.ValidColumnGroup.CREDIT_CARD_PAYMENT,
        column_groups.ValidColumnGroup.CREDIT_CARD_MONETARY,
        column_groups.ValidColumnGroup.OTHERS,
    )

    result = column_groups.ValidColumnGroup.members()

    assert result == expected


@pytest.mark.parametrize(
    ("group", "expected"),
    [
        pytest.param(
            column_groups.ValidColumnGroup.ALL,
            "all_columns",
            id="all",
        ),
        pytest.param(
            column_groups.ValidColumnGroup.SAVING_ACCOUNT_DAYS_TRANSACTIONS,
            "saving_account_days_transactions",
            id="saving-account-days-transactions",
        ),
        pytest.param(
            column_groups.ValidColumnGroup.SAVING_ACCOUNT_MONETARY,
            "saving_account_monetary",
            id="saving-account-monetary",
        ),
        pytest.param(
            column_groups.ValidColumnGroup.OPERATIONS,
            "operations",
            id="operations",
        ),
        pytest.param(
            column_groups.ValidColumnGroup.CREDIT_CARD_PAYMENT,
            "credit_card_payment",
            id="credit-card-payment",
        ),
        pytest.param(
            column_groups.ValidColumnGroup.CREDIT_CARD_MONETARY,
            "credit_card_monetary",
            id="credit-card-monetary",
        ),
        pytest.param(
            column_groups.ValidColumnGroup.OTHERS,
            "others",
            id="others",
        ),
    ],
)
def test_should_convert_pretty_name_to_snake_case(
    group: column_groups.ValidColumnGroup,
    expected: str,
) -> None:
    """Verifica la normalización del nombre legible de cada grupo."""
    result = group.snake_case_name

    assert result == expected


@pytest.mark.parametrize(
    ("column", "expected_group"),
    [
        pytest.param(
            "SavingAccount_Days_with_Transactions",
            column_groups.ValidColumnGroup.SAVING_ACCOUNT_DAYS_TRANSACTIONS,
            id="saving-account-days-with",
        ),
        pytest.param(
            "SavingAccount_Transactions_Count",
            column_groups.ValidColumnGroup.SAVING_ACCOUNT_DAYS_TRANSACTIONS,
            id="saving-account-transactions",
        ),
        pytest.param(
            "SavingAccount_Balance_Average",
            column_groups.ValidColumnGroup.SAVING_ACCOUNT_MONETARY,
            id="saving-account-monetary",
        ),
        pytest.param(
            "SavingAccount_Active_ARG",
            column_groups.ValidColumnGroup.OTHERS,
            id="saving-account-active",
        ),
        pytest.param(
            "Operations_total_mean",
            column_groups.ValidColumnGroup.OPERATIONS,
            id="operations",
        ),
        pytest.param(
            "CreditCard_Payment_total_mean",
            column_groups.ValidColumnGroup.CREDIT_CARD_PAYMENT,
            id="credit-card-payment",
        ),
        pytest.param(
            "CreditCard_Balance_ARG",
            column_groups.ValidColumnGroup.CREDIT_CARD_MONETARY,
            id="credit-card-monetary",
        ),
        pytest.param(
            "CreditCard_Premium",
            column_groups.ValidColumnGroup.OTHERS,
            id="credit-card-excluded-premium",
        ),
        pytest.param(
            "CreditCard_Active",
            column_groups.ValidColumnGroup.OTHERS,
            id="credit-card-excluded-active",
        ),
        pytest.param(
            "CreditCard_CoBranding",
            column_groups.ValidColumnGroup.OTHERS,
            id="credit-card-excluded-co-branding",
        ),
        pytest.param(
            "CreditCard_Product",
            column_groups.ValidColumnGroup.OTHERS,
            id="credit-card-excluded-product",
        ),
        pytest.param(
            "Unclassified_feature",
            column_groups.ValidColumnGroup.OTHERS,
            id="unclassified",
        ),
    ],
)
def test_should_assign_each_column_to_its_business_group(
    settings: config.Settings,
    column: str,
    expected_group: column_groups.ValidColumnGroup,
) -> None:
    """Verifica que cada columna se asigne exclusivamente al grupo esperado."""
    dataframe = pl.DataFrame({column: [1]})

    groups = dict(column_groups.group_columns_by_source(dataframe, settings).items())

    assert groups[expected_group] == (column,)
    assert groups[column_groups.ValidColumnGroup.ALL] == (column,)
    for group, columns in groups.items():
        if group not in (column_groups.ValidColumnGroup.ALL, expected_group):
            assert columns == ()


def test_should_group_all_columns_by_source_and_exclude_id_and_target(
    settings: config.Settings,
) -> None:
    """Verifica el agrupamiento completo y la exclusión de identificador y target."""
    raw_columns = [
        settings.dataset.id,
        "SavingAccount_Days_with_Transactions",
        "SavingAccount_Balance_Average",
        "Operations_total_mean",
        "CreditCard_Payment_total_mean",
        "CreditCard_Balance_ARG",
        "CreditCard_Premium",
        "Other_feature",
        settings.dataset.target,
    ]
    dataframe = pl.DataFrame({column: [1] for column in raw_columns})

    groups = dict(column_groups.group_columns_by_source(dataframe, settings).items())

    assert groups[column_groups.ValidColumnGroup.ALL] == tuple(raw_columns[1:-1])
    assert groups[column_groups.ValidColumnGroup.SAVING_ACCOUNT_DAYS_TRANSACTIONS] == (
        "SavingAccount_Days_with_Transactions",
    )
    assert groups[column_groups.ValidColumnGroup.SAVING_ACCOUNT_MONETARY] == (
        "SavingAccount_Balance_Average",
    )
    assert groups[column_groups.ValidColumnGroup.OPERATIONS] == (
        "Operations_total_mean",
    )
    assert groups[column_groups.ValidColumnGroup.CREDIT_CARD_PAYMENT] == (
        "CreditCard_Payment_total_mean",
    )
    assert groups[column_groups.ValidColumnGroup.CREDIT_CARD_MONETARY] == (
        "CreditCard_Balance_ARG",
    )
    assert groups[column_groups.ValidColumnGroup.OTHERS] == (
        "CreditCard_Premium",
        "Other_feature",
    )
    grouped_columns = [
        column
        for group, columns in groups.items()
        if group is not column_groups.ValidColumnGroup.ALL
        for column in columns
    ]
    assert len(grouped_columns) == len(groups[column_groups.ValidColumnGroup.ALL])
    assert sorted(grouped_columns) == sorted(groups[column_groups.ValidColumnGroup.ALL])


def test_should_exclude_only_columns_from_custom_settings_id_and_target(
    custom_settings: config.Settings,
) -> None:
    """Verifica que solo se excluyan el identificador y target configurados."""
    dataframe = pl.DataFrame(
        {
            "customer_id": [1],
            "churn": [0],
            "client_id": [1],
            "Target": [1],
        }
    )

    groups = dict(
        column_groups.group_columns_by_source(dataframe, custom_settings).items()
    )

    assert groups[column_groups.ValidColumnGroup.ALL] == ("client_id", "Target")
    assert groups[column_groups.ValidColumnGroup.OTHERS] == ("client_id", "Target")
    for group, columns in groups.items():
        if group not in (
            column_groups.ValidColumnGroup.ALL,
            column_groups.ValidColumnGroup.OTHERS,
        ):
            assert columns == ()


def test_should_return_empty_groups_when_dataframe_has_no_columns(
    settings: config.Settings,
) -> None:
    """Verifica que un DataFrame sin columnas produzca grupos vacíos completos."""
    groups = dict(
        column_groups.group_columns_by_source(pl.DataFrame(), settings).items()
    )

    assert set(groups) == set(column_groups.ValidColumnGroup)
    assert all(columns == () for columns in groups.values())


def test_should_accept_grouping_when_all_columns_are_covered() -> None:
    """Verifica que un agrupamiento completo y sin sobrantes sea válido."""
    groups = _build_generated_groups(
        {
            column_groups.ValidColumnGroup.ALL: ("alpha", "beta"),
            column_groups.ValidColumnGroup.OPERATIONS: ("alpha",),
            column_groups.ValidColumnGroup.OTHERS: ("beta",),
        }
    )

    result = dict(groups.items())

    assert result[column_groups.ValidColumnGroup.ALL] == ("alpha", "beta")
    assert result[column_groups.ValidColumnGroup.OPERATIONS] == ("alpha",)
    assert result[column_groups.ValidColumnGroup.OTHERS] == ("beta",)


def test_should_raise_value_error_when_grouping_is_missing_columns() -> None:
    """Verifica que se rechace un agrupamiento al que le faltan columnas."""
    with pytest.raises(ValueError) as exc_info:
        _build_generated_groups(
            {
                column_groups.ValidColumnGroup.ALL: ("alpha", "beta", "gamma"),
                column_groups.ValidColumnGroup.OTHERS: ("alpha", "beta"),
            }
        )

    message = str(exc_info.value)

    assert "expected 3 columns, got 2 grouped" in message
    assert "Missing (1): ['gamma']" in message
    assert "Extra" not in message


def test_should_raise_value_error_when_grouping_has_extra_columns() -> None:
    """Verifica que se rechace un agrupamiento con columnas sobrantes."""
    with pytest.raises(ValueError) as exc_info:
        _build_generated_groups(
            {
                column_groups.ValidColumnGroup.ALL: ("alpha", "beta"),
                column_groups.ValidColumnGroup.OTHERS: (
                    "alpha",
                    "beta",
                    "gamma",
                ),
            }
        )

    message = str(exc_info.value)

    assert "expected 2 columns, got 3 grouped" in message
    assert "Extra (1): ['gamma']" in message
    assert "Missing" not in message


def test_should_report_missing_and_extra_columns_in_error_message() -> None:
    """Verifica que el error detalle faltantes y sobrantes cuando coexisten."""
    with pytest.raises(ValueError) as exc_info:
        _build_generated_groups(
            {
                column_groups.ValidColumnGroup.ALL: ("alpha", "beta"),
                column_groups.ValidColumnGroup.OTHERS: ("alpha", "gamma"),
            }
        )

    message = str(exc_info.value)

    assert "Missing (1): ['beta']" in message
    assert "Extra (1): ['gamma']" in message
