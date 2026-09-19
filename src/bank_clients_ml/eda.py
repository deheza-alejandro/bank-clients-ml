import operator
from collections.abc import Sequence
from typing import Literal

import marimo as mo
import polars as pl
import polars.selectors as cs

from bank_clients_ml.config import Settings, get_settings


def print_describe(df: pl.DataFrame) -> None:
    return mo.output.append(
        df.describe().transpose(include_header=True, column_names="statistic")
    )


def inspect_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """
    Devuelve un DataFrame de diagnóstico con el "shape" del dataframe
    y las columnas que presentan nulos, NaNs, Infs, valores no numéricos
    o sufijos de joins de pandas (_x, _y) o sufijos de joins de polars (_right).
    """
    rows_count, columns_count = df.shape
    df_shape = [f"{rows_count} rows", f"{columns_count} columns"]

    schema = df.schema
    all_cols = list(schema.keys())
    float_cols = [col for col, dt in schema.items() if dt.is_float()]
    non_numeric_cols = [
        col for col, dt in schema.items() if not (dt.is_numeric() or dt == pl.Null)
    ]

    cols_x = [c for c in all_cols if c.endswith("_x")]
    cols_y_right = [c for c in all_cols if c.endswith(("_y", "_right"))]

    exprs = [pl.col(c).is_null().any().alias(f"null_{c}") for c in all_cols]
    for c in float_cols:
        exprs.append(pl.col(c).is_nan().any().alias(f"nan_{c}"))
        exprs.append(pl.col(c).is_infinite().any().alias(f"inf_{c}"))

    if exprs:
        results = df.select(exprs).row(0, named=True)
        null_cols = [c for c in all_cols if results[f"null_{c}"]]
        nan_cols = [c for c in float_cols if results[f"nan_{c}"]]
        inf_cols = [c for c in float_cols if results[f"inf_{c}"]]
    else:
        null_cols, nan_cols, inf_cols = [], [], []

    metrics = [
        ("Shape of the DataFrame", df_shape),
        ("Columns with null, None or NaT", null_cols),
        ("Columns with NaN", nan_cols),
        ("Columns with inf", inf_cols),
        ("Columns with non-numeric values", non_numeric_cols),
        ("Columns ended with _x (pandas join)", cols_x),
        ("Columns ended with _y (pandas join) or _right (polars join)", cols_y_right),
    ]

    return pl.DataFrame(
        {
            "metric": [m[0] for m in metrics],
            "total": [len(m[1]) for m in metrics],
            "values": [m[1] for m in metrics],
        }
    )


ConditionSymbol = Literal["<", ">", "<=", ">=", "==", "!="]

OPERATORS = {
    "<": operator.lt,
    ">": operator.gt,
    "<=": operator.le,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


def _get_operator(condition: ConditionSymbol):
    if condition not in OPERATORS:
        raise ValueError(f"Condición no válida. Usa una de: {list(OPERATORS.keys())}")

    return OPERATORS[condition]


def count_row_matches(
    df: pl.DataFrame,
    columns: Sequence[str] | str,
    threshold: float,
    condition: ConditionSymbol = "<",
) -> pl.DataFrame:
    """Retorna un DataFrame con la cantidad de registros que cumplen la condición por columna.

    Parámetros:
    df: El DataFrame con los datos a analizar.
    columns: El nombre de la columna o columnas a evaluar.
    threshold: El valor límite para la comparación.
    condition: Tipo de comparación.
        "<" para menor que,
        ">" para mayor que,
        "<=" para menor o igual que,
        ">=" para mayor o igual que,
        "==" para igual que,
        "!=" para distinto que.
        Por defecto es 'lt'.
    """
    if isinstance(columns, str):
        columns = [columns]

    op_func = _get_operator(condition)

    return df.select(op_func(pl.col(columns), threshold).sum()).unpivot(
        variable_name="Column name",
        value_name=f"Number of rows {condition} {threshold}",
    )


def filter_columns_by_cardinality(
    df: pl.DataFrame, condition: ConditionSymbol = ">", threshold: int = 10
) -> pl.DataFrame:
    """Devuelve un DataFrame con las columnas con una cantidad de valores únicos
    que cumplen la condición, indicando la cantidad de valores únicos de cada columna."""
    op_func = _get_operator(condition)

    return (
        df.select(pl.all().n_unique())
        .unpivot(variable_name="column", value_name="unique_values")
        .filter(op_func(pl.col("unique_values"), threshold))
    )


def low_cardinality_value_counts(
    df: pl.DataFrame, max_unique_values: int = 10
) -> pl.DataFrame:
    """Calcula el value_counts de las columnas con una cantidad de
    valores únicos <= max_unique_values
    y devuelve un único DataFrame en formato largo (column, value, count).
    """
    cols_to_keep = filter_columns_by_cardinality(df, "<=", max_unique_values)[
        "column"
    ].to_list()

    if not cols_to_keep:
        return pl.DataFrame(
            schema={"column": pl.String, "value": pl.String, "count": pl.UInt32}
        )

    return (
        df.select(pl.col(cols_to_keep).cast(pl.String))
        .unpivot(variable_name="column", value_name="value")
        .group_by(["column", "value"])
        .len("count")
        .sort(["column", "count"], descending=[False, True])
    )


def mins_in_range(df: pl.DataFrame, low: float = -1, high: float = 1) -> pl.DataFrame:
    """Calcula las columnas numéricas cuyos valores mínimos se encuentran entre un rango dado.

    Args:
        df: DataFrame de Polars a analizar.
        low: Límite inferior del rango (exclusivo).
        high: Límite superior del rango (exclusivo).

    Returns:
        DataFrame de Polars con las columnas 'columna' y 'mínimo'.
    """
    return (
        df.select(cs.numeric().min())
        .unpivot(variable_name="column", value_name="minimum")
        .filter(
            pl.col("minimum").is_between(low, high, closed="none")
            & (pl.col("minimum") != 0)
        )
    )


def columns_with_zeros(df: pl.DataFrame) -> pl.DataFrame:
    """Devuelve un DataFrame con las columnas numéricas que contienen ceros y su cantidad.

    Args:
        df: DataFrame de Polars a inspeccionar.

    Returns:
        DataFrame con columnas 'column' y 'zeros_quantity'.
    """
    return (
        df.select((cs.numeric() == 0).sum())
        .unpivot(variable_name="column", value_name="zeros_quantity")
        .filter(pl.col("zeros_quantity") > 0)
    )


def filter_nonzero(
    df: pl.DataFrame, columns: list[str], settings: Settings | None = None
) -> pl.DataFrame:
    """Filtra el DataFrame devolviendo las filas donde todas las
    columnas de columns son distintas de cero.

    Args:
        df: DataFrame de origen.
        columns: Columnas a evaluar

    Returns:
        Subconjunto de df donde se cumple df[columns] != 0 para todas las
        columnas indicadas (con client_id si está presente).
    """
    if settings is None:
        settings = get_settings()

    keep = [settings.col_id, *columns] if settings.col_id in df.columns else columns

    return df.filter(pl.all_horizontal(pl.col(columns) != 0)).select(keep)
