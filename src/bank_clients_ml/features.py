from datetime import date

import numpy as np
import polars as pl

from bank_clients_ml.config import Settings, get_settings


def get_date_windows(
    df: pl.DataFrame, date_column: str, prediction_window_size: int
) -> tuple[list[date], list[date]]:
    """el parámetro prediction_window_size se usa para determinar que "offset_by(...)" usar.
        si prediction_window_size es 2, se usa offset_by("-1mo") para el prediction_months,
        si prediction_window_size es 3, se usa offset_by("-2mo") para el prediction_months, y asi.

        la separación entre la ventana de predicción y entrenamiento (Lead Windows
    ) es siempre de 1 mes
    """
    pred_offset = f"-{prediction_window_size - 1}mo"
    train_offset = f"-{prediction_window_size + 1}mo"

    last_month = pl.col(date_column).max()
    first_month = pl.col(date_column).min()

    windows = df.select(
        prediction_months=pl.date_range(
            last_month.dt.offset_by(pred_offset), last_month, interval="1mo"
        ).implode(),
        training_months=pl.date_range(
            first_month, last_month.dt.offset_by(train_offset), interval="1mo"
        ).implode(),
    )

    prediction_months = windows["prediction_months"][0].to_list()
    training_months = windows["training_months"][0].to_list()

    return training_months, prediction_months


def safe_denominator(
    denominator: str | pl.Expr, search: int = 0, replace_with: int = 1
) -> pl.Expr:
    """
    Reemplaza valores inseguros para división.

    Args:
        denominator: Expresión o nombre de columna del denominador.
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
    numerator_exp = pl.col(numerator) if isinstance(numerator, str) else numerator
    return numerator_exp / safe_denominator(denominator) * 100.0


def get_saving_account_cols():
    return [
        "SavingAccount_Balance_Average",
        "SavingAccount_Balance_FirstDate",
        "SavingAccount_Balance_LastDate",
        "SavingAccount_Total_Amount",
        "SavingAccount_Salary_Payment_Amount",
        "SavingAccount_Transfer_In_Amount",
        "SavingAccount_Credits_Amounts",
        "SavingAccount_ATM_Extraction_Amount",
        "SavingAccount_Service_Payment_Amount",
        "SavingAccount_CreditCard_Payment_Amount",
        "SavingAccount_Transfer_Out_Amount",
        "SavingAccount_DebitCard_Spend_Amount",
        "SavingAccount_Debits_Amounts",
    ]


def get_binary_identity_features_cols() -> list[str]:
    return [
        "CreditCard_Premium",
        "CreditCard_Active",
        "CreditCard_CoBranding",
        "Loan_Active",
        "Mortgage_Active",
        "SavingAccount_Active_ARG_Salary",
        "SavingAccount_Active_ARG",
        "SavingAccount_Active_DOLLAR",
        "DebitCard_Active",
        "Investment_Active",
        "Package_Active",
        "Insurance_Life",
        "Insurance_Home",
        "Insurance_Accidents",
        "Insurance_Mobile",
        "Insurance_ATM",
        "Insurance_Unemployment",
        "Sex",
        "Mobile",
        "Email",
    ]


def get_credit_card_cols() -> list[str]:
    return [
        "CreditCard_Balance_ARG",
        "CreditCard_Balance_DOLLAR",
        "CreditCard_Total_Limit",
        "CreditCard_Total_Spending",
        "CreditCard_Spending_1_Installment",
        "CreditCard_Spending_Installments",
        "CreditCard_Spending_CrossBoarder",
        "CreditCard_Spending_Aut_Debits",
        "CreditCard_Revolving",
    ]


def target_encode_columns(
    df: pl.DataFrame, columns: list[str], settings: Settings | None = None
) -> pl.DataFrame:
    """Calcula porcentajes respecto al target por columna categórica usando Polars.

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

        if df.select((total == 0).any()).item():
            categories_with_issue = (
                df.filter(total == 0).get_column(column).unique().to_list()
            )
            raise ValueError(
                f"La columna '{column}' tiene categorías con denominador 0: {categories_with_issue}"
            )

        expressions.append(((count_1 / total) * 100).round(3).alias(column))

    return df.with_columns(expressions)


def get_constant_columns(df: pl.DataFrame) -> list[str]:
    return _get_true_column_names(df.select(pl.all().n_unique() == 1))


def get_imbalanced_binary_columns(
    df: pl.DataFrame,
    threshold: float = 0.10,
    settings: Settings | None = None,
) -> list[str]:
    if settings is None:
        settings = get_settings()

    c_threshold = 1.0 - threshold
    is_binary = pl.all().n_unique() == 2
    first_val_ratio = (pl.all() == pl.all().first()).mean()
    is_imbalanced = (first_val_ratio < threshold) | (first_val_ratio > c_threshold)

    return _get_true_column_names(
        df.drop(settings.col_target).select(is_binary & is_imbalanced)
    )


def _get_true_column_names(df: pl.DataFrame) -> list[str]:
    """Retorna los nombres de las columnas que contienen valores verdaderos"""
    return df.unpivot().filter(pl.col("value")).get_column("variable").to_list()


class CorrelationAnalyzer:
    def __init__(self, df: pl.DataFrame, settings: Settings | None = None):
        if settings is None:
            settings = get_settings()

        self.corr_df = df.drop(settings.col_id, settings.col_target).corr()
        self.columns = self.corr_df.columns

    def get_redundant_correlated_columns(self, threshold: float = 0.80) -> list[str]:
        """Deja siempre la primera columna fuera de la lista.
        Si N columnas están correlacionadas entre sí, devolverá N-1 en la lista.

        El triángulo inferior y la diagonal quedan en 0.0,
        lo que no afecta al max() ya que |r| >= 0"""
        abs_corr_np = np.abs(self.corr_df.to_numpy())
        upper_triangle = np.triu(abs_corr_np, k=1)

        maximums_per_column = upper_triangle.max(axis=0)
        correlated_columns = [
            col
            for col, max_val in zip(self.columns, maximums_per_column, strict=True)
            if max_val > threshold
        ]
        return correlated_columns

    def get_correlations_for(
        self, column: str, threshold: float = 0.80
    ) -> pl.DataFrame:
        return (
            self.corr_df.select(
                pl.Series("feature", self.columns),
                pl.col(column).alias("correlation"),
            )
            .filter(
                (pl.col("feature") != column)
                & (pl.col("correlation").abs() > threshold)
            )
            .sort(pl.col("correlation").abs(), descending=True)
        )


def standardize(
    df: pl.DataFrame, ddof: int = 0, settings: Settings | None = None
) -> pl.DataFrame:
    if settings is None:
        settings = get_settings()

    cols_to_standardize = pl.exclude(settings.col_id, settings.col_target)

    standardized_ABT = df.with_columns(
        (cols_to_standardize - cols_to_standardize.mean())
        / cols_to_standardize.std(ddof=ddof)
    )
    return standardized_ABT


def group_columns_by_source(columns: list[str]) -> dict[str, list[str]]:
    """Agrupa las columnas de un DataFrame de Polars según su fuente de negocio.

    Args:
        columns: columnas de un DataFrame estandarizado.

    Returns:
        Diccionario con los grupos de columnas clasificados.
    """
    groups: dict[str, list[str]] = {
        "saving_account_days_transactions": [],
        "saving_account_monetary": [],
        "operations": [],
        "credit_card_payment": [],
        "credit_card_monetary": [],
        "others": [],
    }

    credit_card_excluded = {
        "CreditCard_Premium",
        "CreditCard_Active",
        "CreditCard_CoBranding",
        "CreditCard_Product",
    }

    for col in columns:
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
        elif col.startswith("CreditCard_") and col not in credit_card_excluded:
            groups["credit_card_monetary"].append(col)
        else:
            groups["others"].append(col)

    all_grouped_cols = set().union(*groups.values())
    expected_cols = set(columns)

    missing = expected_cols - all_grouped_cols
    extra = all_grouped_cols - expected_cols

    if missing or extra:
        msg = []
        total_columns = len(columns)
        total_grouped = sum(map(len, groups.values()))
        if missing:
            msg.append(
                f"Columnas faltantes en los grupos (len = {len(missing)}): {missing} \n"
            )
        if extra:
            msg.append(
                f"Columnas extra/duplicadas en los grupos (len = {len(extra)}): {extra} \n"
            )

        raise ValueError(
            f"La lista original (len = {total_columns}) "
            f"no coincide con los grupos generados (len = {total_grouped}).\n"
            f"Es probable que no estés teniendo en cuenta alguna columna "
            f"y tengas que revisar esta función \n" + " | ".join(msg)
        )

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


def group_bins_by_ranges(
    column: str,
    ranges: list[tuple[int, int]],
    table: pl.DataFrame,
    settings: Settings | None = None,
) -> pl.Expr:
    """Construye una cadena de expresiones de Polars para agrupar
    valores en bins definidos por rangos de Bins (bin_min, bin_max),
    calculando dinámicamente los valores por rango y el valor por defecto desde la tabla resumen.

    Args:
        column: Nombre de la columna a agrupar.
        ranges: Lista de tuplas con los límites de los rangos (bin_min, bin_max).
        table: DataFrame de Polars con las columnas 'Bin', 'Min', 'Max', 'Clients'
        y 'settings.col_target'.

    Returns:
        pl.Expr: Expresión de Polars con el agrupamiento aplicado.
    """
    if settings is None:
        settings = get_settings()

    range_conditions = [
        (pl.col("Bin") >= b_min) & (pl.col("Bin") <= b_max) for b_min, b_max in ranges
    ]

    aggregations = []
    for i, cond in enumerate(range_conditions):
        aggregations.extend(
            [
                pl.col("Min").filter(cond).min().alias(f"min_{i}"),
                pl.col("Max").filter(cond).max().alias(f"max_{i}"),
                pl.col("Clients").filter(cond).sum().alias(f"cli_{i}"),
                pl.col(settings.col_target).filter(cond).sum().alias(f"tgt_{i}"),
            ]
        )

    out_of_range_cond = ~pl.any_horizontal(range_conditions)
    aggregations.extend(
        [
            pl.col("Clients").filter(out_of_range_cond).sum().alias("cli_def"),
            pl.col(settings.col_target)
            .filter(out_of_range_cond)
            .sum()
            .alias("tgt_def"),
        ]
    )

    stats = table.select(aggregations).row(0, named=True)
    cli_def = stats["cli_def"] or 0
    tgt_def = stats["tgt_def"] or 0
    default_val = (float(tgt_def) / float(cli_def) * 100.0) if cli_def > 0 else 0.0

    expr_col = pl.col(column)

    low, high, val = _get_range_data(0, stats)
    expr = pl.when(expr_col.is_between(low, high)).then(val)

    for i in range(1, len(ranges)):
        low, high, val = _get_range_data(i, stats)
        expr = expr.when(expr_col.is_between(low, high)).then(val)

    return expr.otherwise(default_val)


def _get_range_data(i: int, stats) -> tuple[float, float, float]:
    """Función auxiliar para extraer bounds y valor por índice"""
    min_val = stats[f"min_{i}"]
    max_val = stats[f"max_{i}"]
    cli_sum = stats[f"cli_{i}"] or 0
    tgt_sum = stats[f"tgt_{i}"] or 0

    low = (float(min_val) - 0.01) if isinstance(min_val, (int, float)) else 0.0
    high = (float(max_val) + 0.01) if isinstance(max_val, (int, float)) else 0.0
    val = (float(tgt_sum) / float(cli_sum) * 100.0) if cli_sum > 0 else 0.0
    return low, high, val
