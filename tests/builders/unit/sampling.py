"""Constructores de datos usados por los tests de muestreo."""

from datetime import date

import polars as pl

from bank_clients_ml import config, sampling


def build_clean_data() -> pl.DataFrame:
    """Devuelve datos limpios con clientes válidos e inválidos para las ventanas.

    ## Returns:
        DataFrame con diez meses de historia y siete clientes que cubren los
        criterios de cantidad de meses y banderas de producto.
    """
    months = [date(2024, month, 1) for month in range(1, 11)]
    clients = {
        1: ((date(2024, 2, 1),), "No", "No", 1),
        2: ((date(2024, 3, 1),), "No", "No", 0),
        3: ((date(2024, 4, 1),), "Yes", "No", 1),
        4: ((date(2024, 5, 1),), "No", "Yes", 0),
        5: ((), "No", "No", 1),
        6: ((date(2024, 7, 1),), "No", "No", 0),
        7: ((date(2024, 2, 1), date(2024, 3, 1)), "No", "No", 1),
    }
    rows = []
    for client_id, client in clients.items():
        missing_months, package_active, co_branding, target = client
        for month in months:
            if month in missing_months:
                continue
            is_last_training_month = month == date(2024, 7, 1)
            rows.append(
                {
                    "client_id": client_id,
                    "Month": month,
                    "Package_Active": (
                        package_active if is_last_training_month else "No"
                    ),
                    "CreditCard_CoBranding": (
                        co_branding if is_last_training_month else "No"
                    ),
                    "Target": target,
                    "Balance": float(client_id * 100 + month.month),
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "client_id": pl.Int64,
            "Month": pl.Date,
            "Package_Active": pl.String,
            "CreditCard_CoBranding": pl.String,
            "Target": pl.Int64,
            "Balance": pl.Float64,
        },
    )


def build_renamed_clean_data(clean_data: pl.DataFrame) -> pl.DataFrame:
    """Renombra las columnas de los datos limpios según la configuración alternativa.

    ## Args:
        clean_data: Datos limpios con los nombres por defecto.

    ## Returns:
        DataFrame con los nombres de columna alternativos.
    """
    return clean_data.rename(
        {
            "client_id": "customer_id",
            "Target": "churn",
            "Month": "snapshot_month",
            "Package_Active": "has_package",
            "CreditCard_CoBranding": "has_branding",
        }
    )


def build_abt_data() -> pl.DataFrame:
    """Devuelve una tabla balanceada para probar la partición estratificada.

    ## Returns:
        DataFrame con veinte clientes por cada clase del target.
    """
    return pl.DataFrame(
        {
            "client_id": list(range(1, 41)),
            "Target": [0] * 20 + [1] * 20,
            "Feature": [float(value) for value in range(40)],
        }
    )


def build_imbalanced_train() -> pl.DataFrame:
    """Devuelve un conjunto desbalanceado con mayoría de target cero.

    ## Returns:
        DataFrame con treinta clientes de target cero y cinco de target uno.
    """
    return pl.DataFrame(
        {
            "client_id": list(range(1, 36)),
            "Target": [0] * 30 + [1] * 5,
            "Feature": [float(value) for value in range(35)],
        }
    )


def build_time_window_split(
    clean_data: pl.DataFrame,
    settings: config.Settings,
) -> sampling.TimeWindowSplit:
    """Devuelve la partición temporal por defecto de los datos limpios.

    ## Args:
        clean_data: Datos limpios con diez meses de historia.
        settings: Configuración con los nombres de las columnas.

    ## Returns:
        Ventana temporal con dos meses de predicción y un mes de separación.
    """
    return sampling.TimeWindowSplit.generate(clean_data, 2, settings=settings)
