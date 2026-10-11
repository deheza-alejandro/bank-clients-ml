"""Limpieza inicial de datos crudos de bank_clients_ml.

## Funciones exportadas:
    clean_up_data: Limpia y normaliza el DataFrame crudo de clientes.
"""

import polars as pl

from bank_clients_ml import config


def clean_up_data(
    raw_data: pl.DataFrame, settings: config.Settings | None = None
) -> pl.DataFrame:
    """Limpia y normaliza el DataFrame crudo de clientes.

    Filtra los registros sin target y normaliza los tipos de las
    columnas de fechas e identificador según la configuración vigente.

    Args:
        raw_data: Datos crudos de clientes por depurar.
        settings: Configuración del proyecto. Si es None, se obtiene
            la configuración global.

    Returns:
        Datos limpios con fechas convertidas, identificador
            normalizado y sin valores nulos en la columna target.
    """
    settings = settings or config.get_settings()

    clean_data = raw_data.filter(
        pl.col(settings.dataset.target).is_not_null()
    ).with_columns(
        pl.col(
            settings.dataset.month,
            settings.dataset.first_product_date,
            settings.dataset.last_product_date,
        ).str.to_date(),
        pl.col(settings.dataset.id).cast(pl.Int64),
    )
    return clean_data
