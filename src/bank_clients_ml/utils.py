import operator
from collections.abc import Sequence
from typing import Literal

import polars as pl
import polars.selectors as cs

ConditionSymbol = Literal["<", ">", "<=", ">=", "==", "!="]

OPERATORS = {
    "<": operator.lt,
    ">": operator.gt,
    "<=": operator.le,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


def get_true_cols(mask: pl.DataFrame) -> list[str]:
    """
    Helper idiomático: convierte un DF booleano de 1 fila en una lista de columnas True
    """
    if mask.width == 0:
        return []
    return mask.unpivot().filter(pl.col("value"))["variable"].to_list()


def print_df_personalizado(df_name: str, df: pl.DataFrame) -> None:
    """
    Imprime un resumen personalizado del DataFrame incluyendo dimensiones,
    columnas con nulos, NaN, infinitos y sufijos _x / _y (join de pandas) o _right (join de polars).
    """
    print(f"Dataframe: {df_name}, shape: {df.shape}\n")

    null_cols = get_true_cols(df.null_count() > 0)
    print(f"Columnas con null, None o NaT ({len(null_cols)}):\n{null_cols}\n")

    nan_cols = get_true_cols(df.select(cs.float().is_nan().any()))
    print(f"Columnas con NaN ({len(nan_cols)}):\n{nan_cols}\n")

    inf_cols = get_true_cols(df.select(cs.float().is_infinite().any()))
    print(f"Columnas con inf ({len(inf_cols)}):\n{inf_cols}\n")

    cols_x = df.select(cs.ends_with("_x")).columns
    print(f"Columnas terminadas con _x (join de pandas) ({len(cols_x)}):\n{cols_x}\n")

    cols_y = df.select(cs.ends_with("_y", "_right")).columns
    print(f"Columnas terminadas con _y o _right ({len(cols_y)}):\n{cols_y}\n")


def print_threshold_violations(
    df: pl.DataFrame,
    columns: Sequence[str] | str,
    threshold: int,
    condition: ConditionSymbol = "<"
) -> None:
    """Muestra la cantidad de registros que sobrepasan un límite para una o varias columnas.

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

    if condition not in OPERATORS:
        raise ValueError(f"Condición no válida. Usa una de: {list(OPERATORS.keys())}")

    op_func = OPERATORS[condition]

    counts_dict = df.select(op_func(pl.col(columns), threshold).sum()).row(0, named=True)

    for col, count in counts_dict.items():
        print(f"Cantidad de registros {condition} {threshold} en {col}: {count}")
    print("\n")


def print_value_counts(df: pl.DataFrame) -> None:
    """Calcula el value_counts de todas las columnas simultáneamente en paralelo
    y luego los imprime
    """
    counts = df.select(pl.all().value_counts(sort=True).implode())

    for col in counts.columns:
        print(counts[col].explode().struct.unnest())
        print("\n")


def filter_nonzero(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Filtra el DataFrame devolviendo las filas donde todas las
    columnas de columns son distintas de cero.

    Args:
        df: DataFrame de origen.
        columns: Columnas a evaluar

    Returns:
        Subconjunto de df donde se cumple df[columns] != 0 para todas las
        columnas indicadas (con client_id si está presente).
    """
    keep = ["client_id", *columns] if "client_id" in df.columns and "client_id" not in columns else columns

    return (
        df.filter(pl.all_horizontal(pl.col(columns) != 0))
          .select(keep)
    )

