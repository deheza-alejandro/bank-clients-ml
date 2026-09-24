"""Funciones para el análisis exploratorio de datos con Polars y Marimo.

Las funciones están pensadas para uso interactivo en cuadernos Marimo,
donde algunos resultados se anexan directamente a la salida visible.

Funciones exportadas:
    print_describe: Muestra el resumen estadístico del DataFrame en la salida de Marimo.
    inspect_dataframe: Inspecciona la estructura y la calidad general del DataFrame.
    count_row_matches: Cuenta, por columna, las filas que cumplen una condición.
    filter_columns_by_cardinality: Filtra columnas según su cantidad de valores únicos.
    low_cardinality_value_counts: Filtra columnas de baja cardinalidad e indica las ocurrencias.
    mins_in_range: Identifica columnas numéricas cuyo mínimo cae dentro de un rango abierto.
    columns_with_zeros: Identifica columnas numéricas que contienen valores iguales a cero.
    filter_nonzero: Filtra las filas donde todas las columnas indicadas son distintas de cero.

Constantes exportadas:
    OPERATORS: diccionario con símbolos asociados a funciones de comparación del módulo `operator`.
"""

import operator
from collections.abc import Sequence
from typing import Literal

import marimo as mo
import polars as pl
import polars.selectors as cs

from bank_clients_ml.config import Settings, get_settings


def print_describe(df: pl.DataFrame) -> None:
    """Muestra el resumen estadístico del DataFrame en la salida de Marimo.

    Transpone el resultado de `df.describe()` para facilitar la lectura de las
    estadísticas por columna y las anexa a la salida actual del cuaderno.

    Args:
        df: DataFrame a analizar.
    """
    return mo.output.append(
        df.describe().transpose(include_header=True, column_names="statistic")
    )


def inspect_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Inspecciona la estructura y la calidad general del DataFrame.

    Revisa `df.shape`, columnas con valores nulos, NaN, infinitos, columnas
    no numéricas y posibles restos de joins con sufijos `_x`, `_y` o `_right`.

    Args:
        df: DataFrame a inspeccionar.

    Returns:
        Nuevo DataFrame con las columnas `metric`, `total` y `values`, donde cada fila
        describe una métrica encontrada y las columnas involucradas.
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


type ConditionSymbol = Literal["<", ">", "<=", ">=", "==", "!="]

OPERATORS = {
    "<": operator.lt,
    ">": operator.gt,
    "<=": operator.le,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


def _get_operator(condition: ConditionSymbol):
    """Obtiene la función de comparación asociada a un símbolo condicional.

    Args:
        condition: Símbolo de comparación a resolver.

    Returns:
        Función de comparación correspondiente del módulo `operator`.

    Raises:
        ValueError: Si el símbolo no es uno de los valores permitidos.
    """
    if condition not in OPERATORS:
        raise ValueError(
            f"Invalid condition {condition!r}. Must be one of {list(OPERATORS.keys())}"
        )

    return OPERATORS[condition]


def count_row_matches(
    df: pl.DataFrame,
    columns: Sequence[str] | str,
    threshold: float,
    condition: ConditionSymbol = "<",
) -> pl.DataFrame:
    """Cuenta, por columna, las filas que cumplen una condición respecto a un umbral.

    Args:
        df: DataFrame sobre el que se realiza el conteo.
        columns: Columna o secuencia de columnas a evaluar.
        threshold: Valor umbral contra el que se compara cada celda.
        condition: Símbolo de comparación a aplicar. Valores posibles:
            "<" para menor que,
            ">" para mayor que,
            "<=" para menor o igual que,
            ">=" para mayor o igual que,
            "==" para igual que,
            "!=" para distinto que.

    Returns:
        Nuevo DataFrame con el nombre de cada columna y la cantidad de filas que
        cumplen la condición indicada.
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
    """Filtra columnas según su cantidad de valores únicos.

    Args:
        df: DataFrame con las columnas sobre las que se calcula la cardinalidad.
        condition: Símbolo de comparación aplicado sobre el conteo de valores únicos.
        threshold: Umbral de valores únicos para filtrar las columnas.

    Returns:
        Nuevo DataFrame con las columnas que cumplen la condición y su cantidad de
        valores únicos.
    """
    op_func = _get_operator(condition)

    return (
        df.select(pl.all().n_unique())
        .unpivot(variable_name="column", value_name="unique_values")
        .filter(op_func(pl.col("unique_values"), threshold))
    )


def low_cardinality_value_counts(
    df: pl.DataFrame, max_unique_values: int = 10
) -> pl.DataFrame:
    """Filtra columnas de baja cardinalidad e indica las ocurrencias.

    Filtra las columnas cuya cantidad de valores únicos cumple el límite
    indicado y cuenta las ocurrencias de cada valor.

    Args:
        df: DataFrame con las columnas a analizar.
        max_unique_values: Cantidad máxima de valores únicos que debe tener una columna.

    Returns:
        Nuevo DataFrame con las columnas `column`, `value` y `count`, ordenado por
        columna y frecuencia descendente. Si ninguna columna cumple el criterio,
        retorna un DataFrame vacío con dicho esquema.
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
    """Identifica columnas numéricas cuyo mínimo cae dentro de un rango abierto.

    Excluye los límites del intervalo y los mínimos iguales a cero, lo que
    resulta útil para detectar valores cercanos a cero o posibles residuos de
    normalizaciones.

    Args:
        df: DataFrame a analizar.
        low: Límite inferior exclusivo del rango.
        high: Límite superior exclusivo del rango.

    Returns:
        Nuevo DataFrame con las columnas que cumplen el criterio y su valor mínimo.
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
    """Identifica columnas numéricas que contienen valores iguales a cero.

    Args:
        df: DataFrame a analizar.

    Returns:
        Nuevo DataFrame con las columnas que contienen al menos un cero y la cantidad
        de ceros encontrados en cada una.
    """
    return (
        df.select((cs.numeric() == 0).sum())
        .unpivot(variable_name="column", value_name="zeros_quantity")
        .filter(pl.col("zeros_quantity") > 0)
    )


def filter_nonzero(
    df: pl.DataFrame, columns: list[str], settings: Settings | None = None
) -> pl.DataFrame:
    """Filtra las filas donde todas las columnas indicadas son distintas de cero.

    Conserva la columna identificadora configurada, cuando existe en el
    DataFrame, junto con las columnas evaluadas.

    Args:
        df: DataFrame a filtrar.
        columns: Columnas de `df` a analizar.
        settings: Configuración con el nombre de la columna identificadora.
            Si no se indica, se obtiene la configuración global.

    Returns:
        Nuevo DataFrame filtrado con la columna identificadora y las columnas que cumplen la
        condición evaluada.
    """
    if settings is None:
        settings = get_settings()

    keep = [settings.col_id, *columns] if settings.col_id in df.columns else columns

    return df.filter(pl.all_horizontal(pl.col(columns) != 0)).select(keep)
