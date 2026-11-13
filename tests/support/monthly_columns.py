"""Columnas mensuales compartidas por los tests.

Define los grupos que componen las columnas de entrada del pipeline mensual.
`MONTHLY_TRANSFORMATION_INPUT_COLS` contiene las columnas que consume
`feature_engineering.with_transformations`; `MONTHLY_PIPELINE_INPUT_COLS`
agrega `SavingAccount_Balance_Average`, usada en la agregación mensual.
"""

from typing import Final

from bank_clients_ml import column_groups

_OPERATIONS_COLS: Final[tuple[str, ...]] = (
    "Operations_Bank",
    "Operations_Terminal",
    "Operations_HomeBanking",
    "Operations_Mobile",
    "Operations_Ivr",
    "Operations_Telemarketer",
    "Operations_ATM",
)

_CREDIT_CARD_PAYMENT_COLS: Final[tuple[str, ...]] = (
    "CreditCard_Payment_Aut_Debit",
    "CreditCard_Payment_External",
    "CreditCard_Payment_Cash",
    "CreditCard_Payment_Web",
    "CreditCard_Payment_ATM",
    "CreditCard_Payment_TAS",
)

_SAVING_ACCOUNT_DAYS_TRANSACTIONS_COLS: Final[tuple[str, ...]] = (
    "SavingAccount_Days_with_Debits",
    "SavingAccount_Days_with_use",
    "SavingAccount_Days_with_Credits",
    "SavingAccount_Credits_Transactions",
    "SavingAccount_Debits_Transactions",
    "SavingAccount_Transactions_Transactions",
    "SavingAccount_Salary_Payment_Transactions",
    "SavingAccount_Transfer_In_Transactions",
    "SavingAccount_ATM_Extraction_Transactions",
    "SavingAccount_Service_Payment_Transactions",
    "SavingAccount_CreditCard_Payment_Transactions",
    "SavingAccount_Transfer_Out_Transactions",
    "SavingAccount_DebitCard_Spend_Transactions",
)

_MONTHLY_EXCLUDED_IDENTITY_COLS: Final[frozenset[str]] = frozenset(
    {
        "CreditCard_CoBranding",
        "Package_Active",
        "Sex",
        "Mobile",
        "Email",
    }
)

_MONTHLY_SAVING_ACCOUNT_COLS: Final[tuple[str, ...]] = tuple(
    column
    for column in column_groups.SAVING_ACCOUNT_COLS
    if column != "SavingAccount_Balance_Average"
)

_MONTHLY_IDENTITY_FEATURES_COLS: Final[tuple[str, ...]] = tuple(
    column
    for column in column_groups.BINARY_IDENTITY_FEATURES_COLS
    if column not in _MONTHLY_EXCLUDED_IDENTITY_COLS
)

MONTHLY_TRANSFORMATION_INPUT_COLS: Final[tuple[str, ...]] = (
    *_OPERATIONS_COLS,
    *_CREDIT_CARD_PAYMENT_COLS,
    *_MONTHLY_SAVING_ACCOUNT_COLS,
    *_SAVING_ACCOUNT_DAYS_TRANSACTIONS_COLS,
    *column_groups.CREDIT_CARD_COLS,
    *_MONTHLY_IDENTITY_FEATURES_COLS,
)
"""Columnas mensuales de entrada requeridas por `with_transformations`."""

MONTHLY_PIPELINE_INPUT_COLS: Final[tuple[str, ...]] = (
    *MONTHLY_TRANSFORMATION_INPUT_COLS,
    "SavingAccount_Balance_Average",
)
"""Columnas mensuales de entrada usadas para construir la historia mensual."""
