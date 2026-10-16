"""Agrupa columnas según su fuente de negocio.

Centraliza las reglas de agrupamiento utilizadas durante la exploración y
la selección de variables con LightGBM.

## Funciones exportadas:
    get_identity_features_cols: Retorna las identity features del cliente.
    get_columns_with_quantities: Retorna las columnas con cantidades.
    group_columns_by_source: Agrupa las columnas de un DataFrame según su fuente de negocio.

## Constantes exportadas:
    SAVING_ACCOUNT_COLS: Columnas monetarias asociadas a la caja de ahorro.
    CATEGORICAL_COLS: Nombres de las categorical features del cliente.
    BINARY_IDENTITY_FEATURES_COLS: Identity features binarias del cliente.
    CREDIT_CARD_COLS: Columnas monetarias asociadas a la tarjeta de crédito.
    COLUMNS_WITH_MONETARY_VALUES: Columnas con valores monetarios.
    CREDIT_CARD_EXCLUDED: Columnas de tarjeta de crédito excluidas del grupo monetario.

## Clases exportadas:
    ColumnGroup: Configuración de un grupo de columnas.
    ValidColumnGroup: Grupos válidos para la selección por fuente.
    GeneratedColumnGroups: Agrupamiento validado de columnas por fuente.
"""

import dataclasses
import enum
import types
from collections.abc import ItemsView
from typing import Final, Self

import polars as pl

from bank_clients_ml import config

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
"""Columnas monetarias asociadas a la caja de ahorro del cliente."""

CATEGORICAL_COLS: Final[tuple[str, ...]] = (
    "Client_Age_grp",
    "Region",
    "CreditCard_Product",
)
"""Nombres de las categorical features del cliente."""

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
"""Identity features binarias del cliente."""


def get_identity_features_cols(
    settings: config.Settings | None = None,
) -> tuple[str, ...]:
    """Retorna las identity features del cliente.

    Representan la foto del cliente en el último mes de entrenamiento.
    Combinan identificador, target, categorical features, fechas de
    producto e identity features binarias.

    ## Args:
        settings: Configuración con los nombres de columnas. Si no se indica,
            se obtiene la configuración global.

    ## Returns:
        Nombres de las columnas que componen las identity features.
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
"""Columnas monetarias de tarjeta de crédito sin "CreditCard_Total_Limit"."""

CREDIT_CARD_COLS: Final[tuple[str, ...]] = (
    "CreditCard_Total_Limit",
    *_CREDIT_CARD_COLS_MINUS_1,
)
"""Columnas monetarias asociadas a la tarjeta de crédito del cliente."""

COLUMNS_WITH_MONETARY_VALUES: Final[tuple[str, ...]] = (
    *SAVING_ACCOUNT_COLS,
    "SavingAccount_Balance_last_minus_first_date",
    *_CREDIT_CARD_COLS_MINUS_1,
)
"""Columnas con valores monetarios para agregación diferenciada."""


def _get_columns_without_quantities(settings: config.Settings) -> set[str]:
    """Retorna los nombres de columnas sin cantidades.

    Reúne columnas monetarias, identity features, mes y fechas de producto
    para excluirlas de la detección de columnas con cantidades.

    ## Args:
        settings: Configuración con los nombres de columnas del dataset.

    ## Returns:
        Conjunto con los nombres de columnas sin cantidades.
    """
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
    """Retorna las columnas con cantidades.

    Identifica por exclusión, conservando el orden original del DataFrame.

    ## Args:
        transformed_training_data: DataFrame transformado cuyas columnas se
            desean clasificar.
        settings: Configuración con los nombres de columnas del dataset.

    ## Returns:
        Nombres de las columnas con cantidades en orden de aparición.
    """
    return tuple(
        [
            c
            for c in transformed_training_data.columns
            if c not in _get_columns_without_quantities(settings)
        ]
    )


@dataclasses.dataclass(frozen=True, slots=True)
class ColumnGroup:
    """Configuración de un grupo de columnas.

    ## Attributes:
        top_n: Cantidad de variables a seleccionar dentro del grupo.
        pretty_name: Nombre legible del grupo para reportes y gráficos.
    """

    top_n: int
    pretty_name: str


class ValidColumnGroup(enum.Enum):
    """Grupos válidos para la selección por fuente de negocio.

    Incluye la clave ALL con todas las variables consideradas y top_n cero
    como referencia para la selección de variables.
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
        """Retorna los miembros del enum.

        ## Returns:
            Tupla con todos los grupos válidos en orden de definición.
        """
        return tuple([member for member in cls])

    @property
    def top_n(self) -> int:
        """Cantidad de variables a seleccionar dentro del grupo."""
        return self.value.top_n

    @property
    def pretty_name(self) -> str:
        """Nombre legible del grupo para reportes y gráficos."""
        return self.value.pretty_name

    @property
    def snake_case_name(self) -> str:
        """Retorna el nombre del grupo en formato snake case.

        Convierte el nombre legible a minúsculas, elimina paréntesis y
        reemplaza espacios por guiones bajos.

        ## Returns:
            Nombre normalizado para uso en archivos y columnas.
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
"""Columnas de tarjeta de crédito excluidas del grupo monetario."""


def _classify_column(col: str) -> ValidColumnGroup:
    """Clasifica una columna según su fuente de negocio.

    Aplica reglas por prefijo y contenido para asignar la columna a uno de
    los grupos definidos en ValidColumnGroup. Las columnas incluidas en
    CREDIT_CARD_EXCLUDED se derivan al grupo residual OTHERS.

    ## Args:
        col: Nombre de la columna a clasificar.

    ## Returns:
        Grupo al que pertenece la columna.

    ## Example:
        group = _classify_column("Operations_total_mean")
        group = _classify_column("CreditCard_Balance_ARG")
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
    """Agrupamiento validado de columnas por fuente de negocio.

    ## Attributes:
        _groups: Mapeo inmutable de cada grupo a sus columnas asignadas.
    """

    _groups: types.MappingProxyType[ValidColumnGroup, tuple[str, ...]]

    def __post_init__(self) -> None:
        """Valida que el agrupamiento cubra exactamente las columnas esperadas.

        Compara el conjunto de columnas agrupadas con el conjunto de columnas
        originales y verifica que no existan faltantes ni sobrantes. Este control
        garantiza que ninguna variable se pierda o se duplique antes de la
        selección por grupos.

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
        """Retorna los pares de grupo y columnas asignadas.

        ## Returns:
            Vista con los grupos y sus tuplas de columnas correspondientes para
            poder iterar los datos como un diccionario.
        """
        return self._groups.items()


def group_columns_by_source(
    df: pl.DataFrame, settings: config.Settings
) -> GeneratedColumnGroups:
    """Agrupa las columnas de un DataFrame según su fuente de negocio.

    Excluye identificador y target definidos en la configuración, clasifica
    cada columna restante según su fuente de negocio y valida la cobertura
    total del agrupamiento.

    ## Args:
        df: DataFrame de entrada cuyas columnas se desean agrupar.
        settings: Configuración con los nombres del identificador y del target.

    ## Returns:
        Agrupamiento por fuente, donde cada grupo se asocia con sus columnas.

    ## Raises:
        ValueError: Si la validación detecta columnas faltantes o sobrantes.

    ## Example:
        columns_groups = group_columns_by_source(df, settings)
        for group, columns in columns_groups.items():
            print(group.pretty_name, len(columns))
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
