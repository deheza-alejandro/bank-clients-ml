"""Fixtures compartidos para los tests del paquete."""

from collections.abc import Callable

import pytest

from bank_clients_ml import config


@pytest.fixture
def settings() -> config.Settings:
    """Devuelve una configuración independiente para cada test.

    Returns:
        Configuración con los valores por defecto del proyecto.
    """
    return config.Settings()


@pytest.fixture
def custom_dataset_settings() -> config.Settings:
    """Devuelve una configuración con nombres de columnas personalizados.

    ## Returns:
        Configuración con identificador, target, mes, fechas y banderas
            alternativos.
    """
    return config.Settings(
        dataset=config.DatasetColumns(
            id="customer_id",
            target="churn",
            month="snapshot_month",
            first_product_date="first_seen",
            last_product_date="last_seen",
            package_active="has_package",
            credit_card_co_branding="has_branding",
        )
    )


@pytest.fixture
def patch_global_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[config.Settings], None]:
    """Devuelve un helper que reemplaza la configuración global.

    ## Args:
        monkeypatch: Utilidad para reemplazar `get_settings`.

    ## Returns:
        Función que recibe la configuración que debe devolver `get_settings`.
    """

    def _patch(settings: config.Settings) -> None:
        monkeypatch.setattr(config, "get_settings", lambda: settings)

    return _patch
