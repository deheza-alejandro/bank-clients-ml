"""Constructores de datos usados por los tests de feature engineering."""

from datetime import date

import polars as pl

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


def build_float_frame(
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


def build_identity_features() -> pl.DataFrame:
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


def build_training_data() -> pl.DataFrame:
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


def build_monthly_history() -> pl.DataFrame:
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


def build_extra_transformations_data() -> pl.DataFrame:
    """Devuelve dos filas con indicadores de uso en cero y diez.

    ## Returns:
        DataFrame con todas las columnas requeridas por las transformaciones extra.
    """
    return build_float_frame(
        EXTRA_TRANSFORMATION_COLUMNS,
        [
            {column: 0.0 for column in EXTRA_TRANSFORMATION_COLUMNS},
            {column: 10.0 for column in EXTRA_TRANSFORMATION_COLUMNS},
        ],
    )
