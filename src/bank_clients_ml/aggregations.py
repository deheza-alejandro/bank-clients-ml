import polars as pl

from bank_clients_ml.config import Settings, get_settings
from bank_clients_ml.features import compute_percentage


def generate_aggregations(
    saving_account_cols: list[str],
    credit_card_cols: list[str],
    training_data_6: pl.DataFrame,
    identity_features_2: pl.DataFrame,
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
        for c in training_data_6.columns
        if c
        not in {
            *columns_with_monetary_values,
            *identity_features_2.columns,
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
        compute_percentage(cols.last(), cols.first()).name.suffix("_diff_rel"),
        (compute_percentage(cols.last(), cols.first()) - 100.0).name.suffix(
            "_percent_var"
        ),
    ]

    data_agg = (
        training_data_6.sort([settings.col_id, "Month"])
        .group_by(settings.col_id)
        .agg(agg_exprs)
    )

    return data_agg
