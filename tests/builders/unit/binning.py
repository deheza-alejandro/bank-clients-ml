"""Constructores de objetos usados por los tests de binning."""

import polars as pl

from bank_clients_ml import binning


def build_bivariate_table() -> pl.DataFrame:
    """Construye una tabla bivariada canónica para los tests.

    La tabla permite comprobar el cálculo ponderado del porcentaje de target:
    el grupo de bines [1, 2] representa el 25 %, el grupo [4, 5] el 42.5 %
    y el bin residual [3] el 10 %.

    ## Returns:
        Tabla con las columnas de bin, mínimo, máximo, clientes y target.
    """
    return pl.DataFrame(
        {
            "Bin": [1, 2, 3, 4, 5],
            "Min": [0.0, 11.0, 21.0, 31.0, 41.0],
            "Max": [10.0, 20.0, 30.0, 40.0, 50.0],
            "Clients": [100, 300, 200, 300, 100],
            "Target": [10, 90, 20, 120, 50],
            "Target_pct": [10.0, 30.0, 10.0, 40.0, 50.0],
        }
    )


def build_age_mapping() -> binning.NameMapping:
    """Devuelve el mapeo de la variable de edad usado en los tests.

    ## Returns:
        Mapeo entre el nombre interno y el nombre legible de la edad.
    """
    return binning.NameMapping.from_dict({"age": "Age"})
