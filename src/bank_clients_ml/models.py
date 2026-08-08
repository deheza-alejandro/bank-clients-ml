import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import uniform as sp_uniform
from sklearn.model_selection import (
    RandomizedSearchCV,
    train_test_split,
    StratifiedKFold,
)

param_test = {
    "n_estimators": np.arange(6, 50, 1),
    "max_depth": np.arange(
        4, 10, 1
    ),  # "arange" genera un array de este tipo [4, 5, 6, 7, 8, 9] de 4 a (10-1) aumentando de a 1
    "num_leaves": np.arange(3, 20, 1),
    "subsample": sp_uniform(loc=0.2, scale=0.8),
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "min_child_samples": np.arange(1000, 3000, 100),
}


def generar_split(
    df,
    target="Target",
    test_size=0.3,  # 70% en training y 30% en test
    random_state=42,
):

    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[
            target
        ],  # mantiene la proporcion de buenos y malos en ambos sets de train y test
    )

    return train, test


def generar_modelo_y_buscador(n_iter=100):

    modelo_LightGBM_clasificador = lgb.LGBMClassifier(
        random_state=314,
        n_jobs=1,
        # verbosity = 2, # para debug
        metric="auc",
    )

    buscador_mejores_hiperparametros = RandomizedSearchCV(
        estimator=modelo_LightGBM_clasificador,
        param_distributions=param_test,
        n_iter=n_iter,  # Va a probar 100 combinaciones diferentes al azar
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
        cv=StratifiedKFold(n_splits=3),  # K-FOLD CROSS-VALIDATION con k = 3
        verbose=1,  # 4 para debug
        random_state=314,
    )

    return modelo_LightGBM_clasificador, buscador_mejores_hiperparametros


def process_model_results(
    df: pd.DataFrame, probabilities: np.ndarray, bins: list[int] = []
) -> pd.DataFrame:
    """
    Combina los datos del cliente con sus probabilidades de predicción,
    calcula los deciles y, opcionalmente, el rango porcentual.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame original que contiene las columnas 'Target' y 'client_id'.
    probabilities : np.ndarray
        Array bidimensional con las probabilidades predichas por el modelo.
    bins : list[int], opcional
        Si no esta vacia, usa los bins y calcula la columna de porcentaje 'porc' para tests (por defecto esta vacia).

    Retorna:
    --------
    pd.DataFrame
        DataFrame procesado con las columnas unidas, deciles y porcentajes si aplica.
    """
    # Seleccionar columnas clave y reiniciar el índice
    selected_features = df[["Target", "client_id"]].reset_index()

    # Extraer la segunda columna de probabilidades (clase 1)
    probabilities_df = pd.DataFrame(probabilities[:, 1], columns=["Prob1"])

    # Concatenar las características con las probabilidades
    combined_df = pd.concat([selected_features, probabilities_df], axis="columns")

    decile_labels = ["10", "9", "8", "7", "6", "5", "4", "3", "2", "1"]

    if bins:
        # Si tiene bins, calcular el rango porcentual y usar los bins
        combined_df["porc"] = combined_df["Prob1"].rank(pct=True) * 100
        combined_df["decil"] = pd.cut(
            combined_df["Prob1"], bins=bins, labels=decile_labels
        )
    else:
        # Sino calcular los deciles del 10 al 1
        combined_df["decil"] = pd.qcut(combined_df["Prob1"], q=10, labels=decile_labels)

    return combined_df


def train_and_get_feature_importances(X_train, columns, n_iter=5, target="Target"):
    """
    Entrena un modelo LightGBM usando RandomizedSearchCV y devuelve el clasificador, el buscador y las importancias de variables.
    """
    model, searcher = generar_modelo_y_buscador(n_iter=n_iter)
    searcher.fit(X_train[columns], X_train[target])
    importances = pd.Series(
        searcher.best_estimator_.feature_importances_,
        index=columns
    )
    print("Best score: ", searcher.best_score_)
    print(searcher)
    return model, searcher, importances

