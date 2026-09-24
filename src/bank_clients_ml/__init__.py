"""Paquete principal para el modelado de "bank_clients_ml".

Reúne los módulos de configuración, exploración de datos, feature engineering,
muestreo, filtrado de variables redundantes, entrenamiento
con LightGBM y visualización utilizados en el proyecto.

Módulos disponibles:
    config: Configuración global y acceso a la instancia singleton.
    column_groups: Agrupamiento de columnas según su fuente de negocio.
    eda: Análisis exploratorio de datos con Polars y Marimo.
    feature_engineering: Creación y transformación de variables.
    sampling: Ventanas temporales y muestreo estratificado.
    redundant_column_filter: Filtrado secuencial de columnas redundantes.
    training: Entrenamiento y evaluación de modelos LightGBM.
    visualization: Generación y guardado de gráficos del análisis.
"""
