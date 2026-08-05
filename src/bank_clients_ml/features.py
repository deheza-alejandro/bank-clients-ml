import numpy as np
import pandas as pd

def columnas_minimo_entre(df, low=-1, high=1):
    df_num = df.select_dtypes(include=['number']) # Tomar solo columnas numéricas
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
    return pd.to_datetime(x, format='%Y-%m-%d')

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
