import numpy as np
import pandas as pd


def columnas_minimo_entre(df, low=-1, high=1):
    df_num = df.select_dtypes(include=["number"])  # Tomar solo columnas numéricas
    minimos = df_num.min()
    columnas_filtradas = minimos[(minimos > low) & (minimos < high) & (minimos != 0)]
    return columnas_filtradas


def columnas_con_ceros(df):
    # Cuenta los ceros en todas las columnas del dataframe a la vez
    ceros_por_columna = (df == 0).sum()
    # Filtra y muestra solo las columnas que tienen ceros
    columnas_filtradas = ceros_por_columna[ceros_por_columna > 0]
    for col, cant_ceros in columnas_filtradas.items():
        print(f"{col} : {cant_ceros}")


def a_datetime(x):
    return pd.to_datetime(x, format="%Y-%m-%d")


def safe_denominator(data, search=0, replace_with=1):
    """
    Reemplaza el valor indicado (por defecto 0) por un valor seguro (por defecto 1)
    para evitar divisiones por cero.

    "data" puede ser un DataFrame o una Serie de Pandas.

    Versión optimizada: Devuelve un array de NumPy para evitar la sobrecarga de Pandas.

    Importante: al usar esta funcion se pierden los nombres de las filas y columnas del "data"
    """
    # usar .to_numpy() es mas rapido, pero se pierden los nombres de las filas y columnas
    values = data.to_numpy()
    return np.where(values == search, replace_with, values)


def compute_percentage(num, den):
    """
    Calcula el porcentaje entre dos series o columnas evitando división por cero.
    """
    return 100 * num / safe_denominator(den)


def replace_null_with_value(df, col, value):
    """
    Reemplaza los valores nulos de una columna con un valor específico.
    """
    return np.where(df[col].isnull(), value, df[col])


def calculate_target_percentage_by_category(df, col, target="Target"):
    """
    Calcula porcentajes respecto al target por columna categorica.
    """
    tabla = df.groupby([col, target]).size().unstack(fill_value=0)
    denominador_por_categoria = tabla[0.0] + tabla[1.0]

    if (denominador_por_categoria == 0).any():
        categorias_con_problema = denominador_por_categoria[denominador_por_categoria == 0].index.tolist()
        # el denominador_por_categoria nunca deberia ser 0. si da 0 es porque estoy haciendo algo mal
        raise ValueError(
            f"Error: La columna '{col}' tiene categorías con denominador 0: {categorias_con_problema}"
        )

    return (100 * tabla[1.0] / denominador_por_categoria).round(3)


def group_columns_by_source(df):
    """Agrupa las columnas de ``df`` según su fuente de negocio.

    Args:
        df (pandas.DataFrame): Analytical base table estandarizada donde las
            columnas siguen las convenciones de nombres ``SavingAccount_``,
            ``Operations_``, ``CreditCard_``.

    Returns:
        dict[str, list[str]]: Mapeo con las claves
        ``saving_account_days_transactions``, ``saving_account_monetary``,
        ``operations``, ``credit_card_payment``, ``credit_card_monetary`` y
        ``others``; la unión de todas las listas equivale a ``df.columns``
        menos ``client_id`` y ``Target``.
    """
    saving_account_days_transactions = [
        x for x in df.columns
        if x.startswith("SavingAccount_Days_with_")
        or (x.startswith("SavingAccount_") and "Transactions" in x)
    ]
    saving_account_monetary = [
        x for x in df.columns
        if x.startswith("SavingAccount_")
        and not x.startswith("SavingAccount_Active_")
        and x not in saving_account_days_transactions
    ]
    operations = [x for x in df.columns if x.startswith("Operations_")]
    credit_card_payment = [
        x for x in df.columns if x.startswith("CreditCard_Payment_")
    ]
    credit_card_monetary = [
        x for x in df.columns
        if x.startswith("CreditCard_")
        and x not in [
            "CreditCard_Premium",
            "CreditCard_Active",
            "CreditCard_CoBranding",
            "CreditCard_Product",
        ]
        and x not in credit_card_payment
    ]
    already_assigned = (
        saving_account_days_transactions
        + saving_account_monetary
        + operations
        + credit_card_payment
        + credit_card_monetary
        + ["client_id", "Target"]
    )
    others = [x for x in df.columns if x not in already_assigned]
    return {
        "saving_account_days_transactions": saving_account_days_transactions,
        "saving_account_monetary": saving_account_monetary,
        "operations": operations,
        "credit_card_payment": credit_card_payment,
        "credit_card_monetary": credit_card_monetary,
        "others": others,
    }


def min_max_normalize(series):
    """Escala ``series`` al rango ``[0, 1]`` usando normalización min-max.

    Args:
        series (pandas.Series): Serie numérica a normalizar.

    Returns:
        pandas.Series: ``(series - min) / (max - min)``. Cuando ``max == min``
        todos los valores darían división por cero, por eso se devuelve una
        serie llena de ceros para mantener segura la operación.
    """
    s_min = series.min()
    s_max = series.max()
    denom = s_max - s_min
    if denom == 0:
        return series * 0
    return (series - s_min) / denom


def min_max_normalize_weighted(series, weight):
    """Normaliza ``series`` con min-max y la multiplica elemento a elemento por ``weight``.

    Args:
        series (pandas.Series): Serie numérica a normalizar.
        weight (pandas.Series | scalar): Peso aplicado elemento a elemento
            después de la normalización.

    Returns:
        pandas.Series: ``min_max_normalize(series) * weight``.
    """
    return min_max_normalize(series) * weight


def apply_binning_by_ranges(series, ranges, values, default):
    """Agrupa ``series`` en bins definidos por rangos numéricos.

    Cada entrada de ``ranges`` es una tupla ``(low, high)`` mapeada al valor
    correspondiente en ``values`` usando ``series.between(low, high)``; las
    filas que no caen en ningún rango reciben ``default``.

    Args:
        series (pandas.Series): Serie numérica a agrupar.
        ranges (list[tuple[float, float]]): Lista de rangos
            ``(low, high)``.
        values (list): Valor asignado a cada rango; debe tener la misma
            longitud que ``ranges``.
        default (scalar): Valor asignado a las filas que no caen en ningún rango.

    Returns:
        numpy.ndarray: Valores agrupados con los nuevos bins.
    """
    conditions = [series.between(low, high) for low, high in ranges]
    return np.select(conditions, values, default=default)

