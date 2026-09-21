from typing import Final

import polars as pl

from bank_clients_ml.config import Settings, get_settings


def get_saving_account_cols() -> list[str]:
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


CREDIT_CARD_EXCLUDED: Final[frozenset[str]] = frozenset(
    {
        "CreditCard_Premium",
        "CreditCard_Active",
        "CreditCard_CoBranding",
        "CreditCard_Product",
    }
)

GROUP_WEIGHTS: Final[dict[str, int]] = {
    "saving_account_days_transactions": 1,
    "saving_account_monetary": 1,
    "operations": 1,
    "credit_card_payment": 1,
    "credit_card_monetary": 2,
    "others": 3,
}


def _classify_column(col: str) -> str:
    """Aplica las reglas de negocio para determinar el grupo de una columna."""
    if col.startswith("SavingAccount_Days_with_") or (
        col.startswith("SavingAccount_") and "Transactions" in col
    ):
        group = "saving_account_days_transactions"
    elif col.startswith("SavingAccount_") and not col.startswith(
        "SavingAccount_Active_"
    ):
        group = "saving_account_monetary"
    elif col.startswith("Operations_"):
        group = "operations"
    elif col.startswith("CreditCard_Payment_"):
        group = "credit_card_payment"
    elif col.startswith("CreditCard_") and col not in CREDIT_CARD_EXCLUDED:
        group = "credit_card_monetary"
    else:
        group = "others"

    return group


def _validate_groups(columns: list[str], groups: dict[str, list[str]]) -> None:
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


def group_columns_by_source(
    df: pl.DataFrame, settings: Settings | None = None
) -> dict[str, tuple[list[str], int]]:
    """Agrupa las columnas de un DataFrame de Polars según su fuente de negocio.

    Args:
        df: DataFrame estandarizado.

    Returns:
        Diccionario con los grupos de columnas clasificados.
    """
    if settings is None:
        settings = get_settings()

    excluded_cols = {settings.col_id, settings.col_target}
    all_columns = [col for col in df.columns if col not in excluded_cols]
    grouped: dict[str, list[str]] = {group: [] for group in GROUP_WEIGHTS}

    for col in all_columns:
        group = _classify_column(col)
        grouped[group].append(col)

    _validate_groups(all_columns, grouped)

    result: dict[str, tuple[list[str], int]] = {"all_columns": (all_columns, 0)}
    for group_name, weight in GROUP_WEIGHTS.items():
        result[group_name] = (grouped[group_name], weight)

    return result
