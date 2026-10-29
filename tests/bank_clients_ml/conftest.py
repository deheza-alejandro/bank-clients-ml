"""Fixtures compartidos para los tests del paquete."""

import pytest

from bank_clients_ml import config


@pytest.fixture
def settings() -> config.Settings:
    """Devuelve una configuración independiente para cada test.

    Returns:
        Configuración con los valores por defecto del proyecto.
    """
    return config.Settings()
