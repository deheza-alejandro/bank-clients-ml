"""Funciones para el análisis exploratorio de datos con Polars.

## Funciones exportadas:
    describe: Retorna el resumen estadístico del DataFrame en formato transpuesto.
    inspect_dataframe: Inspecciona la estructura y la calidad general del DataFrame.
    count_row_matches: Cuenta, por columna, las filas que cumplen una condición.
    filter_columns_by_cardinality: Filtra columnas según su cantidad de valores únicos.
    low_cardinality_value_counts: Filtra columnas de baja cardinalidad e indica las ocurrencias.
    get_clients_by_months: Resume la cantidad de meses registrados por cliente.
    mins_in_range: Detecta columnas numéricas cuyo valor mínimo pertenece a un intervalo abierto.
    columns_with_zeros: Identifica columnas numéricas que contienen valores iguales a cero.
    filter_nonzero: Filtra las filas donde todas las columnas son distintas de cero.

## Constantes exportadas:
    OPERATORS: funciones del módulo `operator` asociadas a símbolos de comparación.
"""

import operator
import types
from collections.abc import Callable
from typing import Final, Literal

import polars as pl
import polars.selectors as cs

from bank_clients_ml import config


def describe(df: pl.DataFrame) -> pl.DataFrame:
    """Genera el resumen estadístico del DataFrame en formato transpuesto.

    Calcula las estadísticas descriptivas con `df.describe()`, transpone el
    resultado para mostrar una fila por columna original y normaliza la columna
    `null_count` a entero para facilitar su lectura.

    ## Args:
        df: DataFrame a analizar.

    ## Returns:
        Nuevo DataFrame con una fila por columna del DataFrame original y una
        columna `statistic` que identifica cada medida estadística.
    """
    return (
        df.describe()
        .transpose(include_header=True, column_names="statistic")
        .with_columns(pl.col("null_count").cast(pl.Float64).cast(pl.Int64))
    )


def inspect_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Inspecciona la estructura y la calidad general del DataFrame.

    Revisa `df.shape`, columnas con valores nulos, NaN, infinitos, columnas
    no numéricas y posibles restos de joins con sufijos `_x`, `_y` o `_right`.

    ## Args:
        df: DataFrame a inspeccionar.

    ## Returns:
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

type ComparisonFunction = Callable[[pl.Expr, float | int], pl.Expr]

OPERATORS: Final[types.MappingProxyType[ConditionSymbol, ComparisonFunction]] = (
    types.MappingProxyType(
        {
            "<": operator.lt,
            ">": operator.gt,
            "<=": operator.le,
            ">=": operator.ge,
            "==": operator.eq,
            "!=": operator.ne,
        }
    )
)


def _get_operator(condition: ConditionSymbol) -> ComparisonFunction:
    """Obtiene la función de comparación asociada a un símbolo condicional.

    ## Args:
        condition: Símbolo de comparación a resolver.

    ## Returns:
        Función de comparación correspondiente del módulo `operator`.

    ## Raises:
        ValueError: Si el símbolo no es uno de los valores permitidos.
    """
    if condition not in OPERATORS:
        msg = (
            f"Invalid condition {condition!r}. Must be one of {list(OPERATORS.keys())}"
        )
        raise ValueError(msg)

    return OPERATORS[condition]


def count_row_matches(
    df: pl.DataFrame,
    condition: ConditionSymbol = "<",
    threshold: float = 0.0,
    settings: config.Settings | None = None,
) -> pl.DataFrame:
    """Cuenta, por columna, las filas que cumplen una condición respecto a un umbral.

    ## Args:
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
        settings: Configuración del proyecto. Si es None, se obtiene la
            configuración global.

    ## Returns:
        Nuevo DataFrame con el nombre de cada columna y la cantidad de filas que
        cumplen la condición indicada.
    """
    settings = settings or config.get_settings()

    comparison_function = _get_operator(condition)

    columns = df.columns

    return df.select(comparison_function(pl.col(columns), threshold).sum()).unpivot(
        variable_name=settings.analysis.column_name,
        value_name=f"Number of rows {condition} {threshold}",
    )


def filter_columns_by_cardinality(
    df: pl.DataFrame,
    condition: ConditionSymbol = ">",
    threshold: int = 10,
    settings: config.Settings | None = None,
) -> pl.DataFrame:
    """Filtra columnas según su cantidad de valores únicos.

    ## Args:
        df: DataFrame con las columnas sobre las que se calcula la cardinalidad.
        condition: Símbolo de comparación aplicado sobre el conteo de valores únicos.
        threshold: Umbral de valores únicos para filtrar las columnas.
        settings: Configuración del proyecto. Si es None, se obtiene la
            configuración global.

    ## Returns:
        Nuevo DataFrame con las columnas que cumplen la condición y su cantidad de
        valores únicos.
    """
    settings = settings or config.get_settings()

    comparison_function = _get_operator(condition)

    return (
        df.select(pl.all().n_unique())
        .unpivot(
            variable_name=settings.analysis.column_name, value_name="unique_values"
        )
        .filter(comparison_function(pl.col("unique_values"), threshold))
    )


def low_cardinality_value_counts(
    df: pl.DataFrame,
    max_unique_values: int = 10,
    settings: config.Settings | None = None,
) -> pl.DataFrame:
    """Filtra columnas de baja cardinalidad e indica las ocurrencias.

    Filtra las columnas cuya cantidad de valores únicos cumple el límite
    indicado y cuenta las ocurrencias de cada valor.

    ## Args:
        df: DataFrame con las columnas a analizar.
        max_unique_values: Cantidad máxima de valores únicos que debe tener una columna.
        settings: Configuración del proyecto. Si es None, se obtiene la
            configuración global.

    ## Returns:
        Nuevo DataFrame con las columnas `column`, `value` y `count`, ordenado por
        columna y frecuencia descendente. Si ninguna columna cumple el criterio,
        retorna un DataFrame vacío con dicho esquema.
    """
    settings = settings or config.get_settings()

    cols_to_keep = (
        filter_columns_by_cardinality(df, "<=", max_unique_values)
        .get_column(settings.analysis.column_name)
        .to_list()
    )

    if not cols_to_keep:
        return pl.DataFrame(
            schema={
                settings.analysis.column_name: pl.String,
                "value": pl.String,
                "count": pl.UInt32,
            }
        )

    return (
        df.select(pl.col(cols_to_keep).cast(pl.String))
        .unpivot(variable_name=settings.analysis.column_name, value_name="value")
        .group_by([settings.analysis.column_name, "value"])
        .len("count")
        .sort([settings.analysis.column_name, "count"], descending=[False, True])
    )


def get_clients_by_months(
    clean_data: pl.DataFrame, settings: config.Settings | None = None
) -> pl.DataFrame:
    """Resume la cantidad de meses registrados por cliente.

    ## Args:
        clean_data: DataFrame depurado con una fila por cliente y mes.
        settings: Configuración con el nombre de la columna identificadora
            del cliente. Si no se indica, se obtiene la configuración global.

    ## Returns:
        Nuevo DataFrame con las columnas `month_count` y `clients_count`, donde
        cada fila indica cuántos meses de datos se posee de un grupo de
        clientes y cuántos clientes hay en cada grupo, ordenado por frecuencia
        descendente.
    """
    settings = settings or config.get_settings()

    return (
        clean_data.group_by(settings.dataset.id)
        .len(name="month_count")
        .get_column("month_count")
        .value_counts(sort=True)
        .rename({"count": "clients_count"})
    )


def mins_in_range(
    df: pl.DataFrame,
    *,
    low: float = -1,
    high: float = 1,
    settings: config.Settings | None = None,
) -> pl.DataFrame:
    """Detecta columnas numéricas cuyo valor mínimo pertenece a un intervalo abierto.

    Calcula el mínimo de cada columna numérica y conserva solo aquellas cuyo
    mínimo se encuentra estrictamente entre `low` y `high`, excluyendo además
    los valores iguales a cero. Resulta útil para identificar residuos de
    normalización o valores cercanos a cero.

    ## Args:
        df: DataFrame a analizar.
        low: Límite inferior exclusivo del intervalo de búsqueda.
        high: Límite superior exclusivo del intervalo de búsqueda.
        settings: Configuración del proyecto. Si es None, se obtiene la
            configuración global.

    ## Returns:
        Nuevo DataFrame con las columnas que cumplen el criterio y su valor
        mínimo correspondiente. Si ninguna columna cumple la condición, retorna
        un DataFrame vacío con dicho esquema.
    """
    settings = settings or config.get_settings()

    return (
        df.select(cs.numeric().min())
        .unpivot(variable_name=settings.analysis.column_name, value_name="minimum")
        .filter(
            pl.col("minimum").is_between(low, high, closed="none")
            & (pl.col("minimum") != 0)
        )
    )


def columns_with_zeros(
    df: pl.DataFrame, settings: config.Settings | None = None
) -> pl.DataFrame:
    """Identifica columnas numéricas que contienen valores iguales a cero.

    ## Args:
        df: DataFrame a analizar.
        settings: Configuración del proyecto. Si es None, se obtiene la
            configuración global.

    ## Returns:
        Nuevo DataFrame con las columnas que contienen al menos un cero y la cantidad
        de ceros encontrados en cada una.
    """
    settings = settings or config.get_settings()

    return (
        df.select((cs.numeric() == 0).sum())
        .unpivot(
            variable_name=settings.analysis.column_name, value_name="zeros_quantity"
        )
        .filter(pl.col("zeros_quantity") > 0)
    )


def filter_nonzero(df: pl.DataFrame) -> pl.DataFrame:
    """Filtra las filas donde todas las columnas son distintas de cero.

    ## Args:
        df: DataFrame a filtrar.

    ## Returns:
        Nuevo DataFrame filtrado con las columnas que cumplen la condición
        evaluada.
    """
    columns = df.columns

    return df.filter(pl.all_horizontal(pl.col(columns) != 0))
