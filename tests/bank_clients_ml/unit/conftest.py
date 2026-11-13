"""Fixtures compartidos para los tests unitarios del proyecto."""

from collections.abc import Callable

import pytest

import tests.support.settings
from bank_clients_ml import config


@pytest.fixture
def settings() -> config.Settings:
    """Devuelve una configuración nueva para cada test.

    Se construye con `build_settings_for_tests`, sin pasar por `get_settings`
    ni compartir la instancia cacheada. No lee el archivo `.env`, pero sí
    respeta las variables `ML_*` del entorno; la fixture autouse
    `clean_ml_environment` las elimina en los tests no marcados con
    `requires_real_env`.

    No sustituye la configuración global: si el código bajo prueba llama a
    `config.get_settings()`, hay que inyectarla con `patch_global_settings`.

    ## Returns:
        Configuración con los valores por defecto del proyecto, salvo las
        variables `ML_*` definidas en el entorno.
    """
    return tests.support.settings.build_settings_for_tests()


@pytest.fixture
def custom_dataset_settings() -> config.Settings:
    """Devuelve una configuración con nombres de columnas alternativos.

    Reemplaza solo `Settings.dataset`; las rutas y el resto de los parámetros
    conservan sus valores por defecto. Permite comprobar que el código obtiene
    los nombres de columna desde la configuración en lugar de tenerlos
    codificados.

    Devuelve una instancia nueva, independiente de la configuración global cacheada.

    ## Returns:
        Configuración con estos nombres de columna:
            - `id="customer_id"`
            - `target="churn"`
            - `month="snapshot_month"`
            - `first_product_date="first_seen"`
            - `last_product_date="last_seen"`
            - `package_active="has_package"`
            - `credit_card_co_branding="has_branding"`
    """
    return tests.support.settings.build_settings_for_tests(
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
    """Devuelve un helper que sustituye `config.get_settings` durante el test.

    El helper recibe la configuración que debe devolver `get_settings` y la
    instala con `monkeypatch.setattr`. El parche se revierte automáticamente
    al terminar el test.

    Solo afecta a las llamadas que resuelven `config.get_settings` en tiempo
    de ejecución; no modifica referencias importadas previamente ni el
    entorno o el archivo `.env`. Tampoco limpia la caché, porque de eso se
    encarga la fixture autouse `clean_ml_environment`.

    ## Args:
        monkeypatch: Utilidad para reemplazar `get_settings`.

    ## Returns:
        Función que recibe la configuración que debe devolver
        `config.get_settings`.
    """

    def _patch(settings: config.Settings) -> None:
        monkeypatch.setattr(config, "get_settings", lambda: settings)

    return _patch
