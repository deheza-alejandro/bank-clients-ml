"""Agrupa columnas según su fuente de negocio.

Centraliza las reglas de agrupamiento utilizadas durante la exploración y
la selección de variables con LightGBM.

## Funciones exportadas:
    get_saving_account_cols: Retorna las columnas monetarias asociadas a la caja de ahorro.
    get_categorical_cols: Retorna los nombres de las categorical features del cliente.
    get_binary_identity_features_cols: Retorna las identity features binarias.
    get_identity_features_cols: Retorna las identity features del cliente.
    get_credit_card_cols: Retorna las columnas monetarias asociadas a la tarjeta de crédito.
    group_columns_by_source: Agrupa las columnas de un dataframe según su fuente de negocio.

## Constantes exportadas:
    CREDIT_CARD_EXCLUDED: Columnas asociadas a la tarjeta de crédito excluidas del grupo monetario.

## Clases exportadas:
    ColumnGroups: Grupos de columnas validos para la selección.
"""

import dataclasses
import enum
import types
from collections.abc import ItemsView
from typing import Final, Self

import polars as pl

from bank_clients_ml import config

"""Retorna las columnas monetarias asociadas a la caja de ahorro del cliente."""
SAVING_ACCOUNT_COLS: Final[tuple[str, ...]] = (
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
)

"""Retorna los nombres de las categorical features del cliente."""
CATEGORICAL_COLS: Final[tuple[str, ...]] = (
    "Client_Age_grp",
    "Region",
    "CreditCard_Product",
)


"""Retorna las identity features binarias del cliente.

Representan la foto del cliente en el último mes de entrenamiento.
"""
BINARY_IDENTITY_FEATURES_COLS: Final[tuple[str, ...]] = (
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
)


def get_identity_features_cols(
    settings: config.Settings | None = None,
) -> tuple[str, ...]:
    """Retorna las identity features del cliente.

    Representan la foto del cliente en el último mes de entrenamiento.

    ## Returns:
        Nombres de las columnas binarias.
    """
    settings = settings or config.get_settings()

    return (
        settings.dataset.id,
        settings.dataset.target,
        *CATEGORICAL_COLS,
        settings.dataset.first_product_date,
        settings.dataset.last_product_date,
        *BINARY_IDENTITY_FEATURES_COLS,
    )


_CREDIT_CARD_COLS_MINUS_1: Final[tuple[str, ...]] = (
    "CreditCard_Balance_ARG",
    "CreditCard_Balance_DOLLAR",
    "CreditCard_Total_Spending",
    "CreditCard_Spending_1_Installment",
    "CreditCard_Spending_Installments",
    "CreditCard_Spending_CrossBoarder",
    "CreditCard_Spending_Aut_Debits",
    "CreditCard_Revolving",
)

"""Retorna las columnas monetarias asociadas a la tarjeta de crédito del cliente."""
CREDIT_CARD_COLS: Final[tuple[str, ...]] = (
    "CreditCard_Total_Limit",
    *_CREDIT_CARD_COLS_MINUS_1,
)

COLUMNS_WITH_MONETARY_VALUES: Final[tuple[str, ...]] = (
    *SAVING_ACCOUNT_COLS,
    "SavingAccount_Balance_last_minus_first_date",
    *_CREDIT_CARD_COLS_MINUS_1,
)


def _get_columns_without_quantities(settings: config.Settings) -> set[str]:
    return {
        *COLUMNS_WITH_MONETARY_VALUES,
        *get_identity_features_cols(settings),
        settings.dataset.month,
        settings.dataset.first_product_date,
        settings.dataset.last_product_date,
    }


def get_columns_with_quantities(
    transformed_training_data: pl.DataFrame, settings: config.Settings
) -> tuple[str, ...]:
    return tuple(
        [
            c
            for c in transformed_training_data.columns
            if c not in _get_columns_without_quantities(settings)
        ]
    )


@dataclasses.dataclass(frozen=True, slots=True)
class ColumnGroup:
    top_n: int
    pretty_name: str


class ValidColumnGroup(enum.Enum):
    """Grupos validos.

    Incluye la clave ALL con todas las variables consideradas con top_n cero
    como referencia para el entrenamiento base.
    """

    ALL = ColumnGroup(0, "All Columns")

    SAVING_ACCOUNT_DAYS_TRANSACTIONS = ColumnGroup(
        1, "Saving Account (Days transactions)"
    )

    SAVING_ACCOUNT_MONETARY = ColumnGroup(1, "Saving Account (Monetary)")

    OPERATIONS = ColumnGroup(1, "Operations")

    CREDIT_CARD_PAYMENT = ColumnGroup(1, "Credit Card (Payment)")

    CREDIT_CARD_MONETARY = ColumnGroup(2, "Credit Card (Monetary)")

    OTHERS = ColumnGroup(3, "Others")

    @classmethod
    def members(cls) -> tuple[Self, ...]:
        return tuple([member for member in cls])

    @property
    def top_n(self) -> int:
        return self.value.top_n

    @property
    def pretty_name(self) -> str:
        return self.value.pretty_name

    @property
    def snake_case_name(self) -> str:
        """Formatea.

        Toma un string, lo pasa a minúsculas, elimina los paréntesis
        y reemplaza los espacios por guiones bajos.
        """
        return (
            self.pretty_name.lower().replace("(", "").replace(")", "").replace(" ", "_")
        )


CREDIT_CARD_EXCLUDED: Final[frozenset[str]] = frozenset(
    {
        "CreditCard_Premium",
        "CreditCard_Active",
        "CreditCard_CoBranding",
        "CreditCard_Product",
    }
)


def _classify_column(col: str) -> ValidColumnGroup:
    """Aplica reglas para determinar el grupo de una columna.

    Clasifica una columna según su prefijo y su contenido en uno de los
    grupos definidos en ValidColumnGroup. Las columnas incluidas
    en CREDIT_CARD_EXCLUDED se derivan al grupo residual "others".

    ## Args:
        col: Nombre de la columna a clasificar.

    ## Returns:
        Nombre del grupo al que pertenece la columna.

    ## Example:
        group = _classify_column("Operations_total_mean")
        # group == "operations"
        group = _classify_column("CreditCard_Balance_ARG")
        # group == "credit_card_monetary"
    """
    match col:
        case _ if col.startswith("SavingAccount_Days_with_") or (
            col.startswith("SavingAccount_") and "Transactions" in col
        ):
            return ValidColumnGroup.SAVING_ACCOUNT_DAYS_TRANSACTIONS

        case _ if col.startswith("SavingAccount_") and not col.startswith(
            "SavingAccount_Active_"
        ):
            return ValidColumnGroup.SAVING_ACCOUNT_MONETARY

        case _ if col.startswith("Operations_"):
            return ValidColumnGroup.OPERATIONS

        case _ if col.startswith("CreditCard_Payment_"):
            return ValidColumnGroup.CREDIT_CARD_PAYMENT

        case _ if col.startswith("CreditCard_") and col not in CREDIT_CARD_EXCLUDED:
            return ValidColumnGroup.CREDIT_CARD_MONETARY

        case _:
            return ValidColumnGroup.OTHERS


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedColumnGroups:
    _groups: types.MappingProxyType[ValidColumnGroup, tuple[str, ...]]

    def __post_init__(self) -> None:
        """Valida que el agrupamiento cubra exactamente las columnas esperadas.

        Compara el conjunto de columnas agrupadas con el conjunto de columnas
        originales y verifica que no existan faltantes ni sobrantes. Este control
        garantiza que ninguna variable se pierda o se duplique antes de la
        selección por grupos.

        ## Args:
            columns: Columnas que debieron agruparse.
            groups: Agrupamiento propuesto, donde cada clave es un grupo y cada
                valor contiene sus columnas asignadas.

        ## Raises:
            ValueError: Si existen columnas faltantes o sobrantes en el
                agrupamiento, con el detalle de ambos conjuntos y los conteos.
        """
        all_grouped_cols = []
        for group, columns in self.items():
            if group == ValidColumnGroup.ALL:
                continue
            all_grouped_cols.extend(columns)
        all_grouped_cols_set = set(all_grouped_cols)

        all_cols = self._groups[ValidColumnGroup.ALL]
        expected_cols_set = set(all_cols)

        missing = expected_cols_set - all_grouped_cols_set
        extra = all_grouped_cols_set - expected_cols_set

        if missing or extra:
            msg = []
            total_columns = len(all_cols)
            total_grouped = len(all_grouped_cols)
            if missing:
                msg.append(f"Missing ({len(missing)}): {sorted(missing)} \n")
            if extra:
                msg.append(f"Extra ({len(extra)}): {sorted(extra)} \n")

            raise ValueError(
                f"Column grouping mismatch: expected {total_columns} columns, "
                f"got {total_grouped} grouped. \n" + " | ".join(msg)
            )

    def items(self) -> ItemsView[ValidColumnGroup, tuple[str, ...]]:
        return self._groups.items()


def group_columns_by_source(
    df: pl.DataFrame, settings: config.Settings
) -> GeneratedColumnGroups:
    """Agrupa las columnas de un DataFrame según su fuente de negocio.

    Excluye el identificador y la variable target definidos en la
    configuración, clasifica cada columna restante según su fuente de negocio
    y valida la cobertura total.

    ## Args:
        df: DataFrame de entrada cuyas columnas se desean agrupar.
        settings: Configuración con los nombres del identificador y del
            target. Si no se indica, se obtiene la configuración global.

    ## Returns:
        Agrupamiento por fuente, donde cada grupo se asocia con sus columnas
        y su top_n para la selección.

    ## Raises:
        ValueError: Si la validación detecta columnas faltantes o sobrantes.

    ## Example:
        columns_groups = group_columns_by_source(df, settings)
        operation_cols, weight = columns_groups["operations"]
    """
    excluded_cols = {settings.dataset.id, settings.dataset.target}
    all_columns = [col for col in df.columns if col not in excluded_cols]
    grouped: dict[ValidColumnGroup, list[str]] = {
        group: [] for group in ValidColumnGroup
    }

    for col in all_columns:
        group = _classify_column(col)
        grouped[group].append(col)

    grouped[ValidColumnGroup.ALL] = all_columns

    return GeneratedColumnGroups(
        types.MappingProxyType(
            {column_group: tuple(columns) for column_group, columns in grouped.items()}
        )
    )
