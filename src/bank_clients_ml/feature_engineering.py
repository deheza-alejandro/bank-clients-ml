import polars as pl

from bank_clients_ml.config import Settings, get_settings


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


def _safe_denominator(
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


def _compute_percentage(
    numerator: str | pl.Expr, denominator: str | pl.Expr
) -> pl.Expr:
    """Calcula el porcentaje entre dos columnas asegurando división segura.

    Args:
        numerator: Numerador.
        denominator: Denominador.

    Returns:
        Expresión de Polars con el porcentaje calculado.
    """
    numerator_expr = pl.col(numerator) if isinstance(numerator, str) else numerator
    return numerator_expr / _safe_denominator(denominator) * 100.0


def _min_max_normalize(column: str) -> pl.Expr:
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

    return (column_expr - min) / _safe_denominator(denominator)


def _min_max_normalize_weighted(column: str, weight: str) -> pl.Expr:
    """Normaliza una columna usando min-max y la multiplica por un peso dado.

    Args:
        column: Columna a normalizar.
        weight: Columna de ponderación.

    Returns:
        Expresión de Polars ponderada.
    """
    return _min_max_normalize(column) * pl.col(weight)


def with_transformations(df: pl.DataFrame) -> pl.DataFrame:

    result = df.with_columns(
        [
            # OPERATION
            (
                pl.col("Operations_Bank")
                + pl.col("Operations_Terminal")
                + pl.col("Operations_HomeBanking")
                + pl.col("Operations_Mobile")
                + pl.col("Operations_Ivr")
                + pl.col("Operations_Telemarketer")
                + pl.col("Operations_ATM")
            ).alias("Operations_total"),
            (
                pl.col("Operations_HomeBanking")
                + pl.col("Operations_Mobile")
                + pl.col("Operations_Ivr")
                + pl.col("Operations_Telemarketer")
            ).alias("Operations_remote"),
            (
                pl.col("Operations_Bank")
                + pl.col("Operations_Terminal")
                + pl.col("Operations_ATM")
            ).alias("Operations_in_person"),
            # CREDIT CARD
            (
                pl.col("CreditCard_Payment_Aut_Debit")
                + pl.col("CreditCard_Payment_External")
                + pl.col("CreditCard_Payment_Cash")
                + pl.col("CreditCard_Payment_Web")
                + pl.col("CreditCard_Payment_ATM")
                + pl.col("CreditCard_Payment_TAS")
            ).alias("CreditCard_Payment_total"),
            (
                pl.col("CreditCard_Payment_Aut_Debit")
                + pl.col("CreditCard_Payment_Web")
            ).alias("CreditCard_Payment_remote"),
            (
                pl.col("CreditCard_Payment_External")
                + pl.col("CreditCard_Payment_Cash")
                + pl.col("CreditCard_Payment_ATM")
                + pl.col("CreditCard_Payment_TAS")
            ).alias("CreditCard_Payment_in_person"),
            # SAVING ACCOUNT
            (
                pl.col("SavingAccount_Balance_LastDate")
                - pl.col("SavingAccount_Balance_FirstDate")
            ).alias("SavingAccount_Balance_last_minus_first_date"),
            _compute_percentage(
                "SavingAccount_Balance_LastDate", "SavingAccount_Balance_FirstDate"
            ).alias("SavingAccount_Balance_last_minus_first_date_pct"),
            _compute_percentage(
                "SavingAccount_Days_with_Debits", "SavingAccount_Days_with_use"
            ).alias("SavingAccount_Days_with_Debits_pct"),
            _compute_percentage(
                "SavingAccount_Days_with_Credits", "SavingAccount_Days_with_use"
            ).alias("SavingAccount_Days_with_Credits_pct"),
            _compute_percentage(
                "SavingAccount_Credits_Transactions",
                "SavingAccount_Transactions_Transactions",
            ).alias("SavingAccount_Credits_Transactions_pct"),
            _compute_percentage(
                "SavingAccount_Debits_Transactions",
                "SavingAccount_Transactions_Transactions",
            ).alias("SavingAccount_Debits_Transactions_pct"),
            (
                pl.col("SavingAccount_Credits_Transactions")
                / _safe_denominator("SavingAccount_Days_with_use")
            ).alias("SavingAccount_Transactions_Transactions_DAYS_prom"),
            (
                pl.col("SavingAccount_Credits_Transactions")
                / _safe_denominator("SavingAccount_Days_with_Credits")
            ).alias("SavingAccount_Credits_Transactions_DAYS_prom"),
            (
                pl.col("SavingAccount_Debits_Transactions")
                / _safe_denominator("SavingAccount_Days_with_Debits")
            ).alias("SavingAccount_Debits_Transactions_DAYS_prom"),
            _compute_percentage(
                "SavingAccount_Salary_Payment_Transactions",
                "SavingAccount_Transactions_Transactions",
            ).alias("SavingAccount_Salary_Payment_Transactions_pct"),
            _compute_percentage(
                "SavingAccount_Transfer_In_Transactions",
                "SavingAccount_Transactions_Transactions",
            ).alias("SavingAccount_Transfer_In_Transactions_pct"),
            _compute_percentage(
                "SavingAccount_ATM_Extraction_Transactions",
                "SavingAccount_Transactions_Transactions",
            ).alias("SavingAccount_ATM_Extraction_Transactions_pct"),
            _compute_percentage(
                "SavingAccount_Service_Payment_Transactions",
                "SavingAccount_Transactions_Transactions",
            ).alias("SavingAccount_Service_Payment_Transactions_pct"),
            _compute_percentage(
                "SavingAccount_CreditCard_Payment_Transactions",
                "SavingAccount_Transactions_Transactions",
            ).alias("SavingAccount_CreditCard_Payment_Transactions_pct"),
            _compute_percentage(
                "SavingAccount_Transfer_Out_Transactions",
                "SavingAccount_Transactions_Transactions",
            ).alias("SavingAccount_Transfer_Out_Transactions_pct"),
            _compute_percentage(
                "SavingAccount_DebitCard_Spend_Transactions",
                "SavingAccount_Transactions_Transactions",
            ).alias("SavingAccount_DebitCard_Spend_Transactions_pct"),
            _compute_percentage(
                "SavingAccount_Salary_Payment_Transactions",
                "SavingAccount_Credits_Transactions",
            ).alias("SavingAccount_Salary_Payment_Transactions_CR_pct"),
            _compute_percentage(
                "SavingAccount_Transfer_In_Transactions",
                "SavingAccount_Credits_Transactions",
            ).alias("SavingAccount_Transfer_In_Transactions_CR_pct"),
            _compute_percentage(
                "SavingAccount_ATM_Extraction_Transactions",
                "SavingAccount_Debits_Transactions",
            ).alias("SavingAccount_ATM_Extraction_Transactions_DE_pct"),
            _compute_percentage(
                "SavingAccount_Service_Payment_Transactions",
                "SavingAccount_Debits_Transactions",
            ).alias("SavingAccount_Service_Payment_Transactions_DE_pct"),
            _compute_percentage(
                "SavingAccount_CreditCard_Payment_Transactions",
                "SavingAccount_Debits_Transactions",
            ).alias("SavingAccount_CreditCard_Payment_Transactions_DE_pct"),
            _compute_percentage(
                "SavingAccount_Transfer_Out_Transactions",
                "SavingAccount_Debits_Transactions",
            ).alias("SavingAccount_Transfer_Out_Transactions_DE_pct"),
            _compute_percentage(
                "SavingAccount_DebitCard_Spend_Transactions",
                "SavingAccount_Debits_Transactions",
            ).alias("SavingAccount_DebitCard_Spend_Transactions_DE_pct"),
            _compute_percentage(
                "SavingAccount_Credits_Amounts", "SavingAccount_Total_Amount"
            ).alias("SavingAccount_Credits_Amounts_pct"),
            _compute_percentage(
                "SavingAccount_Debits_Amounts", "SavingAccount_Total_Amount"
            ).alias("SavingAccount_Debits_Amounts_pct"),
            _compute_percentage(
                "SavingAccount_Salary_Payment_Amount", "SavingAccount_Total_Amount"
            ).alias("SavingAccount_Salary_Payment_Amount_pct"),
            _compute_percentage(
                "SavingAccount_Transfer_In_Amount", "SavingAccount_Total_Amount"
            ).alias("SavingAccount_Transfer_In_Amount_pct"),
            _compute_percentage(
                "SavingAccount_ATM_Extraction_Amount", "SavingAccount_Total_Amount"
            ).alias("SavingAccount_ATM_Extraction_Amount_pct"),
            _compute_percentage(
                "SavingAccount_Service_Payment_Amount", "SavingAccount_Total_Amount"
            ).alias("SavingAccount_Service_Payment_Amount_pct"),
            _compute_percentage(
                "SavingAccount_CreditCard_Payment_Amount", "SavingAccount_Total_Amount"
            ).alias("SavingAccount_CreditCard_Payment_Amount_pct"),
            _compute_percentage(
                "SavingAccount_Transfer_Out_Amount", "SavingAccount_Total_Amount"
            ).alias("SavingAccount_Transfer_Out_Amount_pct"),
            _compute_percentage(
                "SavingAccount_DebitCard_Spend_Amount", "SavingAccount_Total_Amount"
            ).alias("SavingAccount_DebitCard_Spend_Amount_pct"),
            _compute_percentage(
                "SavingAccount_Salary_Payment_Amount", "SavingAccount_Credits_Amounts"
            ).alias("SavingAccount_Salary_Payment_Amount_CR_pct"),
            _compute_percentage(
                "SavingAccount_Transfer_In_Amount", "SavingAccount_Credits_Amounts"
            ).alias("SavingAccount_Transfer_In_Amount_CR_pct"),
            _compute_percentage(
                "SavingAccount_ATM_Extraction_Amount", "SavingAccount_Debits_Amounts"
            ).alias("SavingAccount_ATM_Extraction_Amount_DE_pct"),
            _compute_percentage(
                "SavingAccount_Service_Payment_Amount", "SavingAccount_Debits_Amounts"
            ).alias("SavingAccount_Service_Payment_Amount_DE_pct"),
            _compute_percentage(
                "SavingAccount_CreditCard_Payment_Amount",
                "SavingAccount_Debits_Amounts",
            ).alias("SavingAccount_CreditCard_Payment_Amount_DE_pct"),
            _compute_percentage(
                "SavingAccount_Transfer_Out_Amount", "SavingAccount_Debits_Amounts"
            ).alias("SavingAccount_Transfer_Out_Amount_DE_pct"),
            _compute_percentage(
                "SavingAccount_DebitCard_Spend_Amount", "SavingAccount_Debits_Amounts"
            ).alias("SavingAccount_DebitCard_Spend_Amount_DE_pct"),
        ]
    )

    result = result.with_columns(
        [
            # OPERATION
            _compute_percentage("Operations_remote", "Operations_total").alias(
                "Operations_remote_pct"
            ),
            _compute_percentage("Operations_in_person", "Operations_total").alias(
                "Operations_in_person_pct"
            ),
            _compute_percentage("Operations_Bank", "Operations_total").alias(
                "Operations_Bank_pct"
            ),
            _compute_percentage("Operations_Terminal", "Operations_total").alias(
                "Operations_Terminal_pct"
            ),
            _compute_percentage("Operations_HomeBanking", "Operations_total").alias(
                "Operations_HomeBanking_pct"
            ),
            _compute_percentage("Operations_Mobile", "Operations_total").alias(
                "Operations_Mobile_pct"
            ),
            _compute_percentage("Operations_Ivr", "Operations_total").alias(
                "Operations_Ivr_pct"
            ),
            _compute_percentage("Operations_Telemarketer", "Operations_total").alias(
                "Operations_Telemarketer_pct"
            ),
            _compute_percentage("Operations_ATM", "Operations_total").alias(
                "Operations_ATM_pct"
            ),
            _compute_percentage("Operations_Bank", "Operations_in_person").alias(
                "Operations_Bank_IP_pct"
            ),
            _compute_percentage("Operations_Terminal", "Operations_in_person").alias(
                "Operations_Terminal_IP_pct"
            ),
            _compute_percentage("Operations_HomeBanking", "Operations_remote").alias(
                "Operations_HomeBanking_R_pct"
            ),
            _compute_percentage("Operations_Mobile", "Operations_remote").alias(
                "Operations_Mobile_R_pct"
            ),
            _compute_percentage("Operations_Ivr", "Operations_remote").alias(
                "Operations_Ivr_R_pct"
            ),
            _compute_percentage("Operations_Telemarketer", "Operations_remote").alias(
                "Operations_Telemarketer_R_pct"
            ),
            _compute_percentage("Operations_ATM", "Operations_in_person").alias(
                "Operations_ATM_IP_pct"
            ),
            # CREDIT CARD
            _compute_percentage(
                "CreditCard_Payment_remote", "CreditCard_Payment_total"
            ).alias("CreditCard_Payment_remote_pct"),
            _compute_percentage(
                "CreditCard_Payment_in_person", "CreditCard_Payment_total"
            ).alias("CreditCard_Payment_in_person_pct"),
            _compute_percentage(
                "CreditCard_Payment_Aut_Debit", "CreditCard_Payment_total"
            ).alias("CreditCard_Payment_Aut_Debit_pct"),
            _compute_percentage(
                "CreditCard_Payment_External", "CreditCard_Payment_total"
            ).alias("CreditCard_Payment_External_pct"),
            _compute_percentage(
                "CreditCard_Payment_Cash", "CreditCard_Payment_total"
            ).alias("CreditCard_Payment_Cash_pct"),
            _compute_percentage(
                "CreditCard_Payment_Web", "CreditCard_Payment_total"
            ).alias("CreditCard_Payment_Web_pct"),
            _compute_percentage(
                "CreditCard_Payment_ATM", "CreditCard_Payment_total"
            ).alias("CreditCard_Payment_ATM_pct"),
            _compute_percentage(
                "CreditCard_Payment_TAS", "CreditCard_Payment_total"
            ).alias("CreditCard_Payment_TAS_pct"),
            _compute_percentage(
                "CreditCard_Payment_Aut_Debit", "CreditCard_Payment_remote"
            ).alias("CreditCard_Payment_Aut_Debit_R_pct"),
            _compute_percentage(
                "CreditCard_Payment_External", "CreditCard_Payment_in_person"
            ).alias("CreditCard_Payment_External_IP_pct"),
            _compute_percentage(
                "CreditCard_Payment_Cash", "CreditCard_Payment_in_person"
            ).alias("CreditCard_Payment_Cash_IP_pct"),
            _compute_percentage(
                "CreditCard_Payment_Web", "CreditCard_Payment_remote"
            ).alias("CreditCard_Payment_Web_R_pct"),
            _compute_percentage(
                "CreditCard_Payment_ATM", "CreditCard_Payment_in_person"
            ).alias("CreditCard_Payment_ATM_IP_pct"),
            _compute_percentage(
                "CreditCard_Payment_TAS", "CreditCard_Payment_in_person"
            ).alias("CreditCard_Payment_TAS_IP_pct"),
            _compute_percentage(
                "CreditCard_Balance_ARG", "CreditCard_Total_Limit"
            ).alias("CreditCard_Balance_ARG_limit_pct"),
            _compute_percentage(
                "CreditCard_Balance_DOLLAR", "CreditCard_Total_Limit"
            ).alias("CreditCard_Balance_DOLLAR_limit_pct"),
            _compute_percentage(
                "CreditCard_Total_Spending", "CreditCard_Total_Limit"
            ).alias("CreditCard_Total_Spending_limit_pct"),
            _compute_percentage(
                "CreditCard_Spending_1_Installment", "CreditCard_Total_Limit"
            ).alias("CreditCard_Spending_1_Installment_limit_pct"),
            _compute_percentage(
                "CreditCard_Spending_Installments", "CreditCard_Total_Limit"
            ).alias("CreditCard_Spending_Installments_limit_pct"),
            _compute_percentage(
                "CreditCard_Spending_CrossBoarder", "CreditCard_Total_Limit"
            ).alias("CreditCard_Spending_CrossBoarder_limit_pct"),
            _compute_percentage(
                "CreditCard_Spending_Aut_Debits", "CreditCard_Total_Limit"
            ).alias("CreditCard_Spending_Aut_Debits_limit_pct"),
            _compute_percentage("CreditCard_Revolving", "CreditCard_Total_Limit").alias(
                "CreditCard_Revolving_limit_pct"
            ),
            _compute_percentage(
                "CreditCard_Balance_ARG", "CreditCard_Total_Spending"
            ).alias("CreditCard_Balance_ARG_SP_pct"),
            _compute_percentage(
                "CreditCard_Balance_DOLLAR", "CreditCard_Total_Spending"
            ).alias("CreditCard_Balance_DOLLAR_SP_pct"),
            _compute_percentage(
                "CreditCard_Spending_1_Installment", "CreditCard_Total_Spending"
            ).alias("CreditCard_Spending_1_Installment_SP_pct"),
            _compute_percentage(
                "CreditCard_Spending_Installments", "CreditCard_Total_Spending"
            ).alias("CreditCard_Spending_Installments_SP_pct"),
            _compute_percentage(
                "CreditCard_Spending_CrossBoarder", "CreditCard_Total_Spending"
            ).alias("CreditCard_Spending_CrossBoarder_SP_pct"),
            _compute_percentage(
                "CreditCard_Spending_Aut_Debits", "CreditCard_Total_Spending"
            ).alias("CreditCard_Spending_Aut_Debits_SP_pct"),
            _compute_percentage(
                "CreditCard_Revolving", "CreditCard_Total_Spending"
            ).alias("CreditCard_Revolving_SP_pct"),
            # OTHERS
            (
                pl.col("CreditCard_Premium")
                + pl.col("CreditCard_Active")
                + pl.col("Loan_Active")
                + pl.col("Mortgage_Active")
                + pl.col("SavingAccount_Active_ARG_Salary")
                + pl.col("SavingAccount_Active_ARG")
                + pl.col("SavingAccount_Active_DOLLAR")
                + pl.col("DebitCard_Active")
                + pl.col("Investment_Active")
                + pl.col("Insurance_Life")
                + pl.col("Insurance_Home")
                + pl.col("Insurance_Accidents")
                + pl.col("Insurance_Mobile")
                + pl.col("Insurance_ATM")
                + pl.col("Insurance_Unemployment")
            ).alias("Quantity_Active_Products"),
            (
                pl.col("CreditCard_Active")
                + pl.col("SavingAccount_Active_ARG")
                + pl.col("SavingAccount_Active_DOLLAR")
                + pl.col("DebitCard_Active")
            ).alias("Quantity_Common_Active_Product"),
            (
                pl.col("CreditCard_Active")
                + pl.col("SavingAccount_Active_ARG")
                + pl.col("DebitCard_Active")
            ).alias("Quantity_Most_Common_Active_Product"),
        ]
    )

    return result


def aggregate_monthly_to_client(
    saving_account_cols: list[str],
    credit_card_cols: list[str],
    training_data: pl.DataFrame,
    identity_features: pl.DataFrame,
    settings: Settings | None = None,
) -> pl.DataFrame:
    """
    Antes de la agregación se ordenan los registros de cada cliente por mes para que
    luego funcionen "first" y "last" correctamente

    diff_rel (Diferencia relativa): (último / primero)

    pct_var (Variación porcentual): 1 - diferencia relativa
    """
    if settings is None:
        settings = get_settings()

    columns_with_monetary_values = [
        *saving_account_cols,
        "SavingAccount_Balance_last_minus_first_date",
        *[c for c in credit_card_cols if c != "CreditCard_Total_Limit"],
    ]

    columns_with_quantities = [
        c
        for c in training_data.columns
        if c
        not in {
            *columns_with_monetary_values,
            *identity_features.columns,
            "Month",
            "First_product_dt",
            "Last_product_dt",
            settings.col_id,
            settings.col_target,
        }
    ]

    cols = pl.col([*columns_with_quantities, *columns_with_monetary_values])

    agg_exprs = [
        cols.min().name.suffix("_min"),
        cols.max().name.suffix("_max"),
        cols.mean().name.suffix("_mean"),
        cols.median().name.suffix("_median"),
        cols.sum().name.suffix("_sum"),
        (cols != 0).sum().name.suffix("_count_nonzero"),
        cols.var().name.suffix("_var"),
        cols.std().name.suffix("_std"),
        pl.col(columns_with_quantities).n_unique().name.suffix("_nunique"),
        (pl.col(columns_with_monetary_values) / 1000.0)
        .round()
        .n_unique()
        .name.suffix("_rounded_nunique"),
        (cols.max() - cols.min()).name.suffix("_ptp"),
        (cols.last() - cols.first()).name.suffix("_diff"),
        _compute_percentage(cols.last(), cols.first()).name.suffix("_diff_rel"),
        (_compute_percentage(cols.last(), cols.first()) - 100.0).name.suffix(
            "_percent_var"
        ),
    ]

    data_agg = (
        training_data.sort([settings.col_id, "Month"])
        .group_by(settings.col_id)
        .agg(agg_exprs)
    )

    return data_agg


def with_extra_transformations(df: pl.DataFrame) -> pl.DataFrame:
    result = df.with_columns(
        (
            _min_max_normalize("SavingAccount_Days_with_use_count_nonzero")
            + _min_max_normalize("SavingAccount_Days_with_use_min")
            + _min_max_normalize(
                "SavingAccount_CreditCard_Payment_Transactions_count_nonzero"
            )
            + _min_max_normalize("Operations_total_count_nonzero")
            + _min_max_normalize("CreditCard_Payment_total_max")
            + _min_max_normalize("CreditCard_Payment_in_person_max")
            + (pl.col("Operations_in_person_pct_max") > 0).cast(pl.Float64)
            + (pl.col("CreditCard_Payment_Aut_Debit_max") > 0).cast(pl.Float64)
            + (pl.col("CreditCard_Payment_TAS_max") > 0).cast(pl.Float64)
        ).alias("SUM_OF_USES"),
        _min_max_normalize_weighted(
            "SavingAccount_CreditCard_Payment_Amount_max",
            "Operations_total_count_nonzero",
        ).alias("Amount_operations"),
        _min_max_normalize_weighted(
            "SavingAccount_CreditCard_Payment_Amount_max",
            "SavingAccount_CreditCard_Payment_Transactions_count_nonzero",
        ).alias("Amount_transactions"),
        _min_max_normalize_weighted(
            "SavingAccount_CreditCard_Payment_Amount_max",
            "CreditCard_Payment_total_max",
        ).alias("Amount_payment"),
        _min_max_normalize_weighted(
            "CreditCard_Total_Limit_diff_rel", "Operations_total_count_nonzero"
        ).alias("Limit_operations"),
        _min_max_normalize_weighted(
            "CreditCard_Total_Limit_diff_rel",
            "SavingAccount_CreditCard_Payment_Transactions_count_nonzero",
        ).alias("Limit_transactions"),
        _min_max_normalize_weighted(
            "CreditCard_Total_Limit_diff_rel", "CreditCard_Payment_total_max"
        ).alias("Limit_payment"),
    )

    return result


def standardize(
    df: pl.DataFrame, ddof: int = 0, settings: Settings | None = None
) -> pl.DataFrame:
    if settings is None:
        settings = get_settings()

    cols_to_standardize = pl.exclude(settings.col_id, settings.col_target)

    standardized_df = df.with_columns(
        (cols_to_standardize - cols_to_standardize.mean())
        / cols_to_standardize.std(ddof=ddof)
    )
    return standardized_df
