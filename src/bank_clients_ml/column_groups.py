"""Agrupa columnas según su fuente de negocio.

Centraliza las reglas de agrupamiento utilizadas durante la exploración y
la selección de variables con LightGBM.

Funciones exportadas:
    get_saving_account_cols: Retorna las columnas monetarias asociadas a la caja de ahorro.
    get_binary_identity_features_cols: Retorna las identity features binarias.
    get_credit_card_cols: Retorna las columnas monetarias asociadas a la tarjeta de crédito.
    group_columns_by_source: Agrupa las columnas de un dataframe según su fuente de negocio.

Constantes exportadas:
    CREDIT_CARD_EXCLUDED: Columnas asociadas a la tarjeta de crédito excluidas del grupo monetario.
    GROUP_WEIGHTS: Ponderación aplicada a cada grupo en la selección.
"""

from collections.abc import Mapping
from typing import Final

import polars as pl

from bank_clients_ml.config import Settings, get_settings


def get_saving_account_cols() -> list[str]:
    """Retorna las columnas monetarias asociadas a la caja de ahorro del cliente."""
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
    """Retorna las identity features binarias del cliente.

    Representan la foto del cliente en el último mes de entrenamiento.

    Returns:
        Nombres de las columnas binarias.
    """
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
    """Retorna las columnas monetarias asociadas a la tarjeta de crédito del cliente."""
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

GROUP_WEIGHTS: Final[Mapping[str, int]] = {
    "saving_account_days_transactions": 1,
    "saving_account_monetary": 1,
    "operations": 1,
    "credit_card_payment": 1,
    "credit_card_monetary": 2,
    "others": 3,
}


def _classify_column(col: str) -> str:
    """Aplica reglas para determinar el grupo de una columna.

    Clasifica una columna según su prefijo y su contenido en uno de los
    grupos definidos en GROUP_WEIGHTS. Las columnas incluidas
    en CREDIT_CARD_EXCLUDED se derivan al grupo residual "others".

    Args:
        col: Nombre de la columna a clasificar.

    Returns:
        Nombre del grupo al que pertenece la columna.

    Example:
        group = _classify_column("Operations_total_mean")
        # group == "operations"
        group = _classify_column("CreditCard_Balance_ARG")
        # group == "credit_card_monetary"
    """
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


def _validate_groups(columns: list[str], groups: Mapping[str, list[str]]) -> None:
    """Valida que el agrupamiento cubra exactamente las columnas esperadas.

    Compara el conjunto de columnas agrupadas con el conjunto de columnas
    originales y verifica que no existan faltantes ni sobrantes. Este control
    garantiza que ninguna variable se pierda o se duplique antes de la
    selección por grupos.

    Args:
        columns: Columnas que debieron agruparse.
        groups: Agrupamiento propuesto, donde cada clave es un grupo y cada
            valor contiene sus columnas asignadas.

    Raises:
        ValueError: Si existen columnas faltantes o sobrantes en el
            agrupamiento, con el detalle de ambos conjuntos y los conteos.
    """
    all_grouped_cols = set().union(*groups.values())
    expected_cols = set(columns)

    missing = expected_cols - all_grouped_cols
    extra = all_grouped_cols - expected_cols

    if missing or extra:
        msg = []
        total_columns = len(columns)
        total_grouped = sum(map(len, groups.values()))
        if missing:
            msg.append(f"Missing ({len(missing)}): {sorted(missing)} \n")
        if extra:
            msg.append(f"Extra ({len(extra)}): {sorted(extra)} \n")

        raise ValueError(
            f"Column grouping mismatch: expected {total_columns} columns, "
            f"got {total_grouped} grouped. \n" + " | ".join(msg)
        )


def group_columns_by_source(
    df: pl.DataFrame, settings: Settings | None = None
) -> dict[str, tuple[list[str], int]]:
    """Agrupa las columnas de un DataFrame según su fuente de negocio.

    Excluye el identificador y la variable target definidos en la
    configuración, clasifica cada columna restante según su fuente de negocio
    y valida la cobertura total. Agrega además una entrada especial con
    todas las columnas consideradas y ponderación cero como referencia para
    el entrenamiento base.

    Args:
        df: DataFrame de entrada cuyas columnas se desean agrupar.
        settings: Configuración con los nombres del identificador y del
            target. Si no se indica, se obtiene la configuración global.

    Returns:
        Agrupamiento por fuente, donde cada grupo se asocia con sus columnas
        y su ponderación para la selección. Incluye la clave `all_columns`
        con todas las variables consideradas con ponderación cero.

    Raises:
        ValueError: Si la validación detecta columnas faltantes o sobrantes.

    Example:
        columns_groups = group_columns_by_source(df, settings)
        operation_cols, weight = columns_groups["operations"]
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
