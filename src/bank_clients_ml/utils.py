import operator
from collections.abc import Sequence
from typing import Literal

import polars as pl
import polars.selectors as cs

from bank_clients_ml.config import Settings, get_settings

ConditionSymbol = Literal["<", ">", "<=", ">=", "==", "!="]

OPERATORS = {
    "<": operator.lt,
    ">": operator.gt,
    "<=": operator.le,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


def print_without_trunc(df: pl.DataFrame) -> None:
    """Imprime un DataFrame de Polars completo en la consola sin truncar
    filas, columnas ni cadenas largas.
    """
    with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=100):
        print(df)


def scan_anomalies(df: pl.DataFrame) -> pl.DataFrame:
    """
    Devuelve un DataFrame de diagnóstico con las columnas que presentan
    nulos, NaNs, Infs o sufijos de joins de pandas (_x, _y)
    o sufijos de joins de polars (_right).
    """
    null_counts = df.null_count()
    null_cols = [c for c in df.columns if null_counts[c][0] > 0]

    float_df = df.select(cs.float())
    if float_df.width > 0:
        nan_flags = float_df.select(pl.all().is_nan().any())
        inf_flags = float_df.select(pl.all().is_infinite().any())
        nan_cols = [c for c in float_df.columns if nan_flags[c][0]]
        inf_cols = [c for c in float_df.columns if inf_flags[c][0]]
    else:
        nan_cols, inf_cols = [], []

    cols_x = [c for c in df.columns if c.endswith("_x")]
    cols_y_right = [c for c in df.columns if c.endswith(("_y", "_right"))]

    return pl.DataFrame(
        {
            "metric": [
                "Columns with null, None or NaT",
                "Columns with NaN",
                "Columns with inf",
                "Columns ended with _x (pandas join)",
                "Columns ended with _y (pandas join) or _right (polars join)",
            ],
            "total": [
                len(null_cols),
                len(nan_cols),
                len(inf_cols),
                len(cols_x),
                len(cols_y_right),
            ],
            "columns": [null_cols, nan_cols, inf_cols, cols_x, cols_y_right],
        }
    )


def get_operator(condition: ConditionSymbol):
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

    op_func = get_operator(condition)

    return df.select(op_func(pl.col(columns), threshold).sum()).unpivot(
        variable_name="Column name",
        value_name=f"Number of rows {condition} {threshold}",
    )


def filter_columns_by_cardinality(
    df: pl.DataFrame, condition: ConditionSymbol = ">", threshold: int = 10
) -> pl.DataFrame:
    """Devuelve un DataFrame con las columnas con una cantidad de valores unicos
    que cumplen la condición, indicando la cantidad de valores unicos de cada columna."""
    op_func = get_operator(condition)

    return (
        df.select(pl.all().n_unique())
        .unpivot(variable_name="column", value_name="n_unique")
        .filter(op_func(pl.col("n_unique"), threshold))
    )


def low_cardinality_value_counts(df: pl.DataFrame, max_n_unique: int = 10) -> pl.DataFrame:
    """Calcula el value_counts de las columnas con una cantidad de valores únicos <= max_n_unique
    y devuelve un único DataFrame en formato largo (column, value, count).
    """
    cols_to_keep = filter_columns_by_cardinality(df, "<=", max_n_unique)["column"].to_list()

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
