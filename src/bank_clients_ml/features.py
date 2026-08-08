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

