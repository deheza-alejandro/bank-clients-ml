import polars as pl
import polars.selectors as cs

from bank_clients_ml.config import Settings, get_settings





def safe_denominator(
    denominator: str | pl.Expr, search: int = 0, replace_with: int = 1
) -> pl.Expr:
    """
    Reemplaza valores inseguros para división.

    Args:
        denominator: Expresion o nombre de columna del denominador.
        search: Valor a buscar para reemplazar (por defecto 0).
        replace_with: Valor de reemplazo seguro (por defecto 1).
    """
    expr_denominator = (
        pl.col(denominator) if isinstance(denominator, str) else denominator
    )
    return expr_denominator.replace(search, replace_with)


def compute_percentage(numerator: str | pl.Expr, denominator: str | pl.Expr) -> pl.Expr:
    """Calcula el porcentaje entre dos columnas asegurando división segura.

    Args:
        numerator: Numerador.
        denominator: Denominador.

    Returns:
        Expresión de Polars con el porcentaje calculado.
    """
    numerator_exp = (
        pl.col(numerator) if isinstance(numerator, str) else numerator
    )
    return numerator_exp / safe_denominator(denominator) * 100.0


def target_encode_columns(
    df: pl.DataFrame, columns: list[str], settings: Settings | None = None
) -> pl.DataFrame:
    """Calcula porcentajes respecto al target por columna categorica usando Polars.

    Args:
        df: DataFrame de Polars de entrada.
        columns: Columnas categóricas para agrupar.
        target: Nombre de la columna objetivo (por defecto settings.col_target).

    Returns:
        DataFrame con cada columna categórica y su porcentaje de target.
    """
    if settings is None:
        settings = get_settings()

    expressions = []

    for column in columns:
        count_1 = (pl.col(settings.col_target) == 1.0).sum().over(column)
        total = pl.col(settings.col_target).is_in([0.0, 1.0]).sum().over(column)

        # el pl.col("total") nunca deberia ser 0. si da 0 es porque estoy haciendo algo mal
        if df.select((total == 0).any()).item():
            categories_with_issue = (
                df.filter(total == 0).get_column(column).unique().to_list()
            )
            raise ValueError(
                f"La columna '{column}' tiene categorías con denominador 0: {categories_with_issue}"
            )

        expressions.append(((count_1 / total) * 100).round(3).alias(column))

    return df.with_columns(expressions)


def group_columns_by_source(
    df: pl.DataFrame, settings: Settings | None = None
) -> dict[str, list[str]]:
    """Agrupa las columnas de un DataFrame de Polars según su fuente de negocio.

    Args:
        df: DataFrame estandarizado.

    Returns:
        Diccionario con los grupos de columnas clasificados.
    """
    if settings is None:
        settings = get_settings()

    groups: dict[str, list[str]] = {
        "saving_account_days_transactions": [],
        "saving_account_monetary": [],
        "operations": [],
        "credit_card_payment": [],
        "credit_card_monetary": [],
        "others": [],
    }

    CreditCard_excluded = {
        "CreditCard_Premium",
        "CreditCard_Active",
        "CreditCard_CoBranding",
        "CreditCard_Product",
    }
    ignored = {settings.col_id, settings.col_target}

    for col in df.columns:
        if col in ignored:
            continue

        if col.startswith("SavingAccount_Days_with_") or (
            col.startswith("SavingAccount_") and "Transactions" in col
        ):
            groups["saving_account_days_transactions"].append(col)
        elif col.startswith("SavingAccount_") and not col.startswith(
            "SavingAccount_Active_"
        ):
            groups["saving_account_monetary"].append(col)
        elif col.startswith("Operations_"):
            groups["operations"].append(col)
        elif col.startswith("CreditCard_Payment_"):
            groups["credit_card_payment"].append(col)
        elif col.startswith("CreditCard_") and col not in CreditCard_excluded:
            groups["credit_card_monetary"].append(col)
        else:
            groups["others"].append(col)

    return groups


def min_max_normalize(column: str) -> pl.Expr:
    """Devuelve una expresión de Polars que normaliza una columna al rango [0, 1]
    usando normalización min-max.

    Args:
        column: Columna a normalizar.

    Returns:
        Expresión de Polars normalizada.
    """
    column_expr = pl.col(column)
    min = column_expr.min()
    denominator = column_expr.max() - min

    return (column_expr - min) / safe_denominator(denominator)


def min_max_normalize_weighted(column: str, weight: str) -> pl.Expr:
    """Normaliza una columna usando min-max y la multiplica por un peso dado.

    Args:
        column: Columna a normalizar.
        weight: Columna de ponderación.

    Returns:
        Expresión de Polars ponderada.
    """
    return min_max_normalize(column) * pl.col(weight)


def binning_by_ranges(
    column: str,
    ranges: list[tuple[float, float]],
    values: list[float],
    default: float,
) -> pl.Expr:
    """Construye una cadena de expresiones de Polars para agrupar
    valores en bins definidos por rangos numéricos.

    Args:
        column: Columna a agrupar.
        ranges: Lista de tuplas con los límites de los rangos (low, high).
        values: Lista de valores correspondientes a cada rango.
        debe tener la misma longitud que ``ranges``.
        default: Valor por defecto si una fila no cae en ningún rango.

    Returns:
        pl.Expr: Expresión de Polars con el agrupamiento aplicado.
    """
    expr_column = pl.col(column)

    first_low, first_high = ranges[0]
    expr = pl.when(expr_column.is_between(first_low, first_high)).then(values[0])

    for (low, high), val in zip(ranges[1:], values[1:], strict=True):
        expr = expr.when(expr_column.is_between(low, high)).then(val)

    return expr.otherwise(default)
