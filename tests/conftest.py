"""Configuración compartida por toda la suite de tests."""

import os
import sys
from collections.abc import Iterator

import pytest

from bank_clients_ml import config


@pytest.fixture(autouse=True)
def clean_ml_environment(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Aísla `Settings` de las variables `ML_*` y limpia su caché entre tests.

    Es autouse, así que se aplica a todos los tests. Antes y después de cada
    test limpia la caché de `Settings.settings_cached()` para que no se
    reutilice una instancia construida con otra configuración.

    Salvo que el test esté marcado con `@pytest.mark.requires_real_env`,
    elimina las variables de entorno `ML_*`; `monkeypatch` las restaura al
    terminar el test. De este modo, y siempre que no haya un `.env` local,
    los tests parten de los valores por defecto de `config.py`.

    El marker se consulta con `get_closest_marker`, por lo que se puede
    aplicar al test, a la clase o al módulo.

    ## Args:
        request: Acceso al test para consultar el marker.
        monkeypatch: Utilidad para modificar el entorno y la configuración.
            Revierte sus cambios al finalizar el test.

    ## Yields:
        Control al test una vez aplicado el aislamiento.

    ## Note:
        Esta fixture no desactiva el archivo `.env` que `Settings` lee por
        defecto: solo elimina variables del proceso. Si existe un `.env`
        local, sus valores siguen aplicándose, incluso en tests no marcados
        con `requires_real_env`.
    """
    uses_real_env = request.node.get_closest_marker("requires_real_env") is not None

    if not uses_real_env:
        for name in list(os.environ):
            if name.upper().startswith("ML_"):
                monkeypatch.delenv(name, raising=False)

    # Save the original reference: some tests replace settings_cached using
    # monkeypatch, and the teardown for this fixture occurs before it is restored.
    clear_settings_cache = config.Settings.settings_clear_cache
    clear_settings_cache()
    yield
    clear_settings_cache()


@pytest.fixture(autouse=True)
def close_matplotlib_figures() -> Iterator[None]:
    """Cierra todas las figuras de Matplotlib al finalizar cada test."""
    yield
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is not None:
        plt.close("all")
