"""Constructores de datos usados por los tests de cleaning."""

import polars as pl


def build_raw_data() -> pl.DataFrame:
    """Construye un DataFrame crudo con registros válidos, nulos y un target cero.

    ## Returns:
        DataFrame con identificadores de texto, fechas de texto y target con
            valores nulos que deben descartarse.
    """
    return pl.DataFrame(
        {
            "client_id": ["1", "2", "3", "4"],
            "Month": ["2024-01-01", "2024-02-01", None, "2024-03-01"],
            "First_product_dt": [
                "2020-01-15",
                "2020-02-15",
                None,
                "2020-03-15",
            ],
            "Last_product_dt": [
                "2024-01-01",
                "2024-02-01",
                None,
                "2024-03-01",
            ],
            "Target": [1, None, 0, None],
            "Balance": [100.0, 200.0, 300.0, 400.0],
        },
        schema={
            "client_id": pl.String,
            "Month": pl.String,
            "First_product_dt": pl.String,
            "Last_product_dt": pl.String,
            "Target": pl.Int64,
            "Balance": pl.Float64,
        },
    )
