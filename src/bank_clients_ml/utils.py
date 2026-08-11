import numpy as np
import pandas as pd


def print_df_personalizado(nombre_df, df):
    """
    print personalizado
    """
    print(nombre_df, "shape:", df.shape)
    print("")
    nulos = df.columns[df.isnull().any()].tolist()
    print("Columnas con nulos:", len(nulos))
    print(nulos)
    print("")
    c_x = [x for x in df.columns if x.endswith("_x")]
    print("Columnas terminadas con _x:", len(c_x))
    print(c_x)
    print("")
    c_y = [x for x in df.columns if x.endswith("_y")]
    print("Columnas terminadas con _y:", len(c_y))
    print(c_y)
    print("")
    nan = df.columns[df.isna().any()].tolist()
    print("Columnas con NaN:", len(nan))
    print(nan)
    print("")
    inf = df.columns[(df == np.inf).any() | (df == -np.inf).any()].tolist()
    print("Columnas con inf:", len(inf))
    print(inf)


def print_threshold_violations(
    df: pd.DataFrame, column_name: str, threshold: int, condition: str = "lt"
) -> None:
    """Cuenta y muestra la cantidad de registros en una columna que superan o están por debajo de un umbral determinado.

    Parámetros:
    df (pd.DataFrame): El DataFrame con los datos a analizar.
    column_name (str): El nombre de la columna a evaluar.
    threshold (int): El valor límite para la comparación.
    condition (str): Tipo de comparación. 'lt' para menor que (<) o 'gt' para
    mayor que (>). Por defecto es 'lt'.
    """
    # Validar que la condición sea una de las permitidas por Pandas
    if condition not in ["lt", "gt"]:
        raise ValueError("La condición debe ser 'lt' (menor que) o 'gt' (mayor que).")

    # Asignar el símbolo correcto para el mensaje en consola
    symbol = "<" if condition == "lt" else ">"

    # Ejecutar dinámicamente el método de pandas (.lt o .gt) y sumar los True
    total_count = getattr(df[column_name], condition)(threshold).sum()

    # Imprimir el resultado con el formato requerido
    print(f"Cantidad de registros {symbol} {threshold} en {column_name}: {total_count}")


def describe_full(df):
    """
    Muestra la descripción completa del DataFrame sin truncar filas.
    Para que la consola no corte el output.
    """
    pd.set_option('display.max_rows', None)
    result = df.describe(include='all').T
    pd.reset_option('display.max_rows')
    return result


def print_value_counts(df, col):
    """
    Imprime el value_counts de una columna seguido de una línea en blanco.
    """
    print(df[col].value_counts())
    print("")


def filter_nonzero(df, columns):
    """Filtra el ``df`` devolviendo las filas donde todas las columnas de ``columns`` son distintas de cero.

    Args:
        df (pandas.DataFrame): DataFrame de origen.
        columns (list[str]): Columnas a evaluar; cada una debe ser != 0
            para que la fila se mantenga.

    Returns:
        pandas.DataFrame: Subconjunto de ``df`` donde se cumple ``df[columns] != 0`` para todas las
        columnas indicadas (con ``client_id`` si está presente).
    """
    mask = (df[columns] != 0).all(axis='columns')
    keep = columns
    if "client_id" in df.columns and "client_id" not in keep:
        keep = ["client_id", *columns]
    return df.loc[mask, keep]

